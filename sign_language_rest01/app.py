#가상환경에 pip install soundfile
#가상환경에 whisper 설치 한 후 실행
from flask import Flask
from flask import request
import numpy as np
import io
import soundfile as sf
import whisper
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import numpy as np
import pickle
from tqdm import tqdm
import random
import math
import cv2
import os
import torch
import shutil
from werkzeug.utils import secure_filename
import datetime
import subprocess
import json


app = Flask(__name__)




STA_INDEX = 1
END_INDEX = 2

MAX_FRAME_NUM = int ( (24 / 5) * 60 )


with open('data/index_to_word.pickle','rb') as f:
    index_to_word = pickle.load(f)

HID_DIM = 512
OUTPUT_DIM = len(index_to_word)

N_LAYER = 2
DEC_DROPOUT = 0.5
EMB_DIM = 128
BATCH_SIZE = 32
N_EPOCH = 1000
CLIP = 1
learning_rate = 0.0001
model_save_path = 'data/'
save_model_name = 'model1.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#np.load('data/X_data.npy') : keypoint가 저장죈 X_data.npy 읽어서 리턴
X_data = np.load('./data/X_data.npy')

#np.load('data/y_data.npy) : 수어 스크립트가 저장된 y_data.npy 읽어서 리턴
y_data = np.load('data/y_data.npy')

input_size = X_data.shape[2]


#인코더 연산을 클래스로 선언
class Encoder(nn.Module):
    def __init__(self, input_size, hid_dim, n_layer):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.gru = nn.GRU(
            input_size,
            hid_dim,
            n_layer,
            batch_first = True,
            bidirectional = True
        )

        self.fc = nn.Linear(
            hid_dim * 4,
            hid_dim
        )

    def forward(self, x):
        h0 = torch.zeros(self.n_layer*2, x.shape[0], self.hid_dim).cuda().float()
        encoder_output, encoder_hidden = self.gru(x,h0)    #<-- 수정

        encoder_hidden = torch.cat((
            encoder_hidden[-4,:,:],
            encoder_hidden[-3,:,:],
            encoder_hidden[-2,:,:],
            encoder_hidden[-1,:,:]
        ), dim = 1)

        encoder_hidden = torch.tanh(self.fc(encoder_hidden))
        return encoder_output, encoder_hidden

#Attention을 클래스로 구현

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()

        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_output):

        batch_size = encoder_output.shape[0]
        src_len = encoder_output.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_output), dim = 2)))

        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim = 1)

#디코더 연산을 클래스로 구현
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layer, attention, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.attention = attention
        self.n_layer = n_layer
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.gru = nn.GRU(hid_dim*2+emb_dim,hid_dim)
        self.fc_out = nn.Linear((hid_dim*2)+hid_dim+emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, encoder_hidden, encoder_output):

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        attention_output = self.attention(encoder_hidden, encoder_output)
        attention_output = attention_output.unsqueeze(1)

        weighted = torch.bmm(attention_output, encoder_output)

        weighted = weighted.permute(1,0,2)

        rnn_input = torch.cat((embedded, weighted), dim = 2)

        decoder_output, decoder_hidden = self.gru(rnn_input, encoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        decoder_output = decoder_output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((decoder_output, weighted, embedded), dim = 1))

        return prediction,decoder_hidden.squeeze(0)

#SequenceToSequence 구현

class GRU_AT_Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    #딥러닝 학습시 호출되는 함수
    #src : 수어 keypoint
    #trg : 수어 스크립트 문장이 저장
    def forward(self, src, trg, teacher_forcing_ration = 0.5):
        #trg.shape[0] : trg에 저장된 문장의 개수 (16)
        batch_size = trg.shape[0]

        #trg.shape[1] : 수어 스크립트 단어의 개수 (30)
        trg_len = trg.shape[1]

        #decoder.output_dim : 수어 스크립트를 구성하는 단어의 개수 (162)
        trg_vocab_size = self.decoder.output_dim

        #torch.zeros(trg_len, batch_size, trg_vocab_size): 0으로 초기화된 배열 생성
                                                        # 배열 shape[trg_len(30), batch_size(16), trg_vocab_size(162)]
        output = torch.zeros(trg_len, batch_size, trg_vocab_size).cuda()

        #self.encoder(src) : 인코더 연산 실행
        encoder_output, hidden = self.encoder(src)

        #trg[:,0] :수어스크립트 trg 0번째 칸 리턴 <START>
        input = trg[:,0]

        for t in range(1, trg_len):
            #self.decoder(input, hidden, encoder_output) :디코더 연산 실행
            decoder_output, hidden = self.decoder(input, hidden, encoder_output)

            output [t] = decoder_output        #decoder_output: 디코더 연산 결과

            #random.random() < teacher_forcing_ration : 0~1 사이 난수를 생성해서 난수가 0.5 미만이면 난수 리턴
                                                     # 난수가 0.5이상이면 None 리턴

            teacher_force = random.random() < teacher_forcing_ration

            #decoder_output.argmax(1) : decoder_output에서 가장 확률이 높은 단어 대입
            top1 = decoder_output.argmax(1)

            #trg[:,t] if teacher_force : teacher_force 가 None이 아니면 수어스크립트가 저장된 trg[:,t] 리턴
            #  else top1:                teacher_force 가 None이면 decoder의 예측값 top1리턴
            input = trg[:,t] if teacher_force else top1

        return output

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

enc = Encoder(input_size, HID_DIM, N_LAYER).to(device)
att = Attention(HID_DIM).to(device)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYER, att, DEC_DROPOUT).to(device)
model = GRU_AT_Seq2Seq(enc,dec,device).to(device)

model.apply(init_weights)
model.load_state_dict(torch.load("./data/model1.pt"))



whisper_model = whisper.load_model("medium")
#word_to_index.pickle : 수어 스크립트 단어를 어떤 숫자로 변환 해야하는지 저장한 파일



@app.route('/')
def hello_flask():
    return 'Hello, flask'

@app.route("/wave01", methods=['POST'])
def wave2text():
    now = int(datetime.datetime.utcnow().timestamp())
    WAVE_PATH = "./data/wave/" + str(now)
    os.makedirs(WAVE_PATH, exist_ok=True)

    file = request.files['file']

    filename = secure_filename(file.filename)

    file.save(os.path.join(WAVE_PATH, filename))
    """
    # Read the file content as bytes
    audio_data = file.read()

    # Convert the bytes to a NumPy array using soundfile
    with io.BytesIO(audio_data) as f:
        data, samplerate = sf.read(f)

    # Check if the audio has multiple channels
    if len(data.shape) > 1:
        # Convert to mono by averaging channels
        data = np.mean(data, axis=1)

    # Resample the audio to 16kHz if necessary
    if samplerate != 16000:
        # You may need to install the resampy package for this
        import resampy
        data = resampy.resample(data, samplerate, 16000)

    # Ensure the data is in the correct dtype
    data = data.astype(np.float32)

    # Now you have your audio data in a NumPy array
    print(data.shape)  # Shape of the array
    print(samplerate)  # Sample rate of the audio
    """

    text = whisper_model.transcribe(WAVE_PATH+"/"+filename)
    print("text['text']=",text["text"])
    return text["text"]

@app.route("/sign_language", methods=['POST'])
def sign_language2text():
    now = int(datetime.datetime.utcnow().timestamp())

    VIDEO_PATH = "./data/mp4/"+str(now)
    os.makedirs(VIDEO_PATH, exist_ok=True)

    FRAME_PATH = "./data/frame/"+str(now)
    os.makedirs(FRAME_PATH, exist_ok=True)

    file = request.files['file']
    filename = secure_filename(file.filename)

    file.save(os.path.join(VIDEO_PATH, filename))

    video_path = VIDEO_PATH + "/" + filename
    capture = cv2.VideoCapture(video_path)

    frame_num = 0
    jpg_num = 0

    while (True):

        success, frame = capture.read()

        if success & (frame_num % 3 == 0):
            cv2.imwrite(f'{FRAME_PATH}/frame_{jpg_num}.jpg', frame)
            jpg_num = jpg_num + 1

        if not success:
            break

        frame_num = frame_num + 1

    capture.release()

    KEYPOINT_PATH = "./data/keypoint/"+str(now)
    os.makedirs(KEYPOINT_PATH, exist_ok=True)
    command = f".\\openpose\\bin\\OpenPoseDemo.exe   --image_dir  {FRAME_PATH} --hand --model_folder .\\openpose\\models --write_json {KEYPOINT_PATH}  --display 0 --render_pose 0 --num_gpu 1 --num_gpu_start 0"
    subprocess.run(command)

    keypoint_file_list = os.listdir(KEYPOINT_PATH)
    keypoint_num = len(keypoint_file_list)

    all_keypoint_list = []

    for i in range(keypoint_num):
        json_file_path = f"{KEYPOINT_PATH}/frame_{i}_keypoints.json"

        with open(json_file_path, "r") as keypoint_json_file:
            keypoint_json = json.load(keypoint_json_file)
            pose_keypoints_2d = keypoint_json["people"][0]["pose_keypoints_2d"]
            pose_keypoint_arr = np.array(pose_keypoints_2d, dtype="float32").reshape(-1, 3)
            hand_left_keypoints_2d = keypoint_json["people"][0]["hand_left_keypoints_2d"]
            hand_left_keypoint_arr = np.array(hand_left_keypoints_2d, dtype="float32").reshape(-1, 3)
            hand_right_keypoints_2d = keypoint_json["people"][0]["hand_right_keypoints_2d"]
            hand_right_keypoint_arr = np.array(hand_right_keypoints_2d, dtype="float32").reshape(-1, 3)

            keypoint_arr = np.concatenate(
                (pose_keypoint_arr[:, :2], hand_left_keypoint_arr[:, :2], hand_right_keypoint_arr[:, :2]))
            all_keypoint_list.append(keypoint_arr.tolist())

    test_keypoint_arr = np.array(all_keypoint_list, dtype="float32")
    print("test_keypoint_arr.shape=",test_keypoint_arr.shape)

    frame_keypoint_list = []

    for arr_row in test_keypoint_arr:
        x = arr_row[:, 0]
        y = arr_row[:, 1]

        mean_x = np.mean(x)
        mean_y = np.mean(y)

        std_x = np.std(x)
        std_y = np.std(y)

        normal_x = (x - mean_x) / std_x
        normal_y = (y - mean_y) / std_y

        arr_row[:, 0] = normal_x
        arr_row[:, 1] = normal_y

        frame_keypoint_list.append(arr_row.flatten().tolist())

    if len(frame_keypoint_list) < MAX_FRAME_NUM:
        random_choice_frame = np.random.choice(len(frame_keypoint_list), MAX_FRAME_NUM - len(frame_keypoint_list))

        random_choice_frame.sort()

        # for random_idx in random_choice_frame:
        frame_keypoint_list[:0] = [[0] * 134] * (MAX_FRAME_NUM - len(frame_keypoint_list))
    else:
        frame_keypoint_list = frame_keypoint_list[: MAX_FRAME_NUM]

    X_test = np.array([frame_keypoint_list], dtype="float32")
    X_test = torch.tensor(X_test)
    print("X_test.shape=", X_test.shape)

    if torch.cuda.is_available():
        model.cuda()
        X_test = X_test.cuda().float()

    model.eval()

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(X_test)

    sentence_word_list = []
    last_word = None
    for index in range(30):
        start_tensor = torch.tensor([STA_INDEX], dtype=torch.long).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(start_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        pred_word= index_to_word.get(pred_token)


        if pred_token == END_INDEX:
            break
        if last_word != pred_word:
            sentence_word_list.append(index_to_word.get(pred_token))

        last_word = pred_word

    return " ".join(sentence_word_list)




if __name__ == '__main__':
    app.run()