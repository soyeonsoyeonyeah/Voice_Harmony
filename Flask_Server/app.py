from flask import Flask
from flask import request
from flask_cors import CORS
from flask import send_file
import numpy as np
import io
#import soundfile as sf
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
from gtts import gTTS


# 인코더 연산을 클래스로 선언
class Encoder(nn.Module):
    def __init__(self, input_size, hid_dim, n_layer):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.gru = nn.GRU(
            input_size,
            hid_dim,
            n_layer,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hid_dim * 4,
            hid_dim
        )

    def forward(self, x):
        h0 = torch.zeros(self.n_layer * 2, x.shape[0], self.hid_dim).cuda().float()
        encoder_output, encoder_hidden = self.gru(x, h0)

        encoder_hidden = torch.cat((
            encoder_hidden[-4, :, :],
            encoder_hidden[-3, :, :],
            encoder_hidden[-2, :, :],
            encoder_hidden[-1, :, :]
        ), dim=1)

        encoder_hidden = torch.tanh(self.fc(encoder_hidden))
        return encoder_output, encoder_hidden


# Attention을 클래스로 구현

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()

        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_output):
        batch_size = encoder_output.shape[0]
        src_len = encoder_output.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_output), dim=2)))

        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


# 디코더 연산을 클래스로 구현
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layer, attention, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.attention = attention
        self.n_layer = n_layer
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.gru = nn.GRU(hid_dim * 2 + emb_dim, hid_dim)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, encoder_hidden, encoder_output):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        attention_output = self.attention(encoder_hidden, encoder_output)
        attention_output = attention_output.unsqueeze(1)

        weighted = torch.bmm(attention_output, encoder_output)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        decoder_output, decoder_hidden = self.gru(rnn_input, encoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        decoder_output = decoder_output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((decoder_output, weighted, embedded), dim=1))

        return prediction, decoder_hidden.squeeze(0)


# SequenceToSequece 구현
class GRU_AT_Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # 딥러닝 학습시 호출되는 함수
    # src : 수어 keypoint
    # trg : 수어 스크립트 문장이 저장
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # trg.shape[0] : trg에 저장된 문장의  개수 (16)
        batch_size = trg.shape[0]

        # trg.shape[1] : 수어 스크립트 단어의 개수 (30)
        trg_len = trg.shape[1]

        # decoder.output_dim : 수어 스크립트를 구성하는 단어의 개수 (162)
        trg_vocab_size = self.decoder.output_dim

        # torch.zeros(trg_len, batch_size,trg_vocab_size) : 0으로 초기화된 배열 생성
        #                                                  배열 shape [trg_len(30) , batch_size(16) ,trg_vocab_size(162) ]
        output = torch.zeros(trg_len, batch_size, trg_vocab_size).cuda()

        # self.encoder(src) : 인코더 연산 실행
        encoder_output, encoder_hidden = self.encoder(src)

        # trg[:,0] :수어스크립트 trg 0번째 칸 리턴 <START>
        input = trg[:, 0]

        for t in range(1, trg_len):
            # self.decoder(input, encoder_hidden, encoder_output) : 디코더 연산 실행
            decoder_output, decoder_hidden = self.decoder(input, encoder_hidden, encoder_output)

            output[t] = decoder_output  # decoder_output : 디코더 연산 결과

            # random.random() < teacher_forcing_ratio : 0~1 사이 난수를 생성해서 난수가 0.5 미만이면 난수리턴
            #                                                                 난수가 0.5 이상이면 None 리턴
            teacher_force = random.random() < teacher_forcing_ratio

            # decoder_output.argmax(1) : decoder_output에서 가장 확률이 높은 단어 대입
            top1 = decoder_output.argmax(1)

            # trg[:,t] if teacher_force  : teacher_force가 None이 아니면 수어스크립트가 저장된 trg[: ,t] 리턴
            #     else top1 :              teacher_force가 None이면 decoder의 예측값 top1리턴
            input = trg[:, t] if teacher_force else top1

        return output

#-0.08~0.08 사이 난수 리턴
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


#수어 스크립트 시작 <START> 인덱스
STA_INDEX = 1

#수어 스크립트 끝 <END> 인덱스
END_INDEX = 2

#동영상의 3의 배수 프레임 마다 프레임 저장
SKIP_NUM = 3

#수어 동영상 keypoint 개수
MAX_FRAME_NUM = 240

#index_to_word.pickle : Decoder가 예측한 숫자를 어떤 단어로 변환 해야 하는지 저장한 파일
with open('data/index_to_word.pickle','rb') as f:
    #Decoder가 예측한 숫자를 어떤 단어로 변환 해야 하는지 저장한 파일을 읽어서 index_to_word 에 저장
    index_to_word = pickle.load(f)

# len(index_to_word) : 저장된 단어의 개수
OUTPUT_DIM = len(index_to_word)

#torch.cuda.is_available() : GPU 사용가능하면 True 아니면 False

#"cuda:0" if torch.cuda.is_available() else "cpu" : GPU 사용가능하면 "cuda:0" 리탄 아니면 cpu 리턴
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_LAYER = 2
DEC_DROPOUT =  0.5
EMB_DIM =  128


#np.load('data/X_data.npy') : keypoint가 저장죈 X_data.npy 읽어서 리턴
X_data = np.load('./data/X_data.npy')

#np.load('data/y_data.npy) : 수어 스크립트가 저장된 y_data.npy 읽어서 리턴
y_data = np.load('data/y_data.npy')

input_size = X_data.shape[2]
HID_DIM = 512


enc = Encoder(input_size, HID_DIM, N_LAYER).to(device)
att = Attention(HID_DIM).to(device)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYER, att, DEC_DROPOUT).to(device)
model = GRU_AT_Seq2Seq(enc,dec,device).to(device)

##-0.08~0.08 사이 난수로 초기화
model.apply(init_weights)
#학습 결과가 저장된 data/model1.pt 를 읽어서 model에 저장
model.load_state_dict(torch.load("data/model1.pt"))


app = Flask(__name__)
CORS(app)
whisper_model = whisper.load_model("medium")

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
    #수어 동영상을 저장할 폴더
    VIDEO_PATH = "./data/mp4/"+str(now)
    #수어 동영상을 저장 할 폴더가 없으면 생성
    os.makedirs(VIDEO_PATH, exist_ok=True)

    #프레임을 저장 할 폴더
    FRAME_PATH = "./data/frame/"+str(now)
    #프레임을 저장 할 폴더가 없으면 생성
    os.makedirs(FRAME_PATH, exist_ok=True)

    #전송한 수어 동영상 파일의 정보를 리턴
    file = request.files['file']
    #전송한 수어 동영상 파일의 이름 리턴
    filename = secure_filename(file.filename)
    #전송한 수어 동영상 파일 저장
    file.save(os.path.join(VIDEO_PATH, filename))
    #저장한 수어 동영상의 경로(VIDEO_PATH) 전송한 수어 동영상 파일의 이름 (filename)
    video_path = VIDEO_PATH + "/" + filename


    # 동영상의 프레임 번호
    frame_num = 0
    # 저장된 jpg의 순서
    jpg_num = 0

    # cv2.VideoCapture(video_path ) : 수어 동영상 파일의 정보를 가져올 객체 생성
    capture = cv2.VideoCapture(video_path)

    while (True):

        # capture.read() : 동영상의 프레임을 읽어서 리턴

        # success : 프레임을 읽어 올수 있으면 True 더이상 읽을 프레임이 없으면 False
        # frame : 동영상의 프레임 저장
        success, frame = capture.read()

        # success : 프레임을 읽어 올수 있으면 True 더이상 읽을 프레임이 없으면 False

        # (frame_num % SKIP_NUM  == 0) : frame_num (동영상 프레임 순서) 를 SKIP_NUM으로 나눈 나머지가 0
        #                               SKIP_NUM이 3이므로 3의 배수번째 프레임

        if success & (frame_num % SKIP_NUM == 0):
            # 저장 할 프레임 경로
            jpg_path = f'{FRAME_PATH}/frame_{jpg_num}.jpg'
            # 동영상 프레임 저장
            cv2.imwrite(jpg_path, frame)
            # 저장한 jpg 개수 1증가
            jpg_num = jpg_num + 1

        if not success:
            break

        # 프레임 개수 1증가
        frame_num = frame_num + 1

    # 동영상 닫기
    capture.release()

    #키포인트 저장 할 폴더 경로
    KEYPOINT_PATH = "./data/keypoint/"+str(now)
    #키포인트 저장
    os.makedirs(KEYPOINT_PATH, exist_ok=True)

    # openpose\\bin\\OpenPoseDemo.exe : keypoint 생성할 파일 경로
    # --image_dir data\\test_frame : 이미지 프레임 경로
    # --hand : 손의 좌표 키포인트도 추가
    # --model_folder openpose\\models : 모델 파일 경로
    # --write_json data\\test_keypoint : 키포인트 json 파일 저장 경로
    # --display 0 --render_pose 0 : 키포인트 결과를 파일로 저장
    # --num_gpu 1 --num_gpu_start 0 : GPU 설정

    command = f".\\openpose\\bin\\OpenPoseDemo.exe   --image_dir  {FRAME_PATH} --hand --model_folder .\\openpose\\models --write_json {KEYPOINT_PATH}  --display 0 --render_pose 0 --num_gpu 1 --num_gpu_start 0"
    # command 실행
    subprocess.run(command)
    # os.listdir(KEYPOINT_PATH) : KEYPOINT_PATH 에 저장된 JSON 파일 리스트 리턴 (수어 동작의 손과 몸의 좌표 저장)
    keypoint_file_list = os.listdir(KEYPOINT_PATH)
    # len(keypoint_file_list) : 폴더에 저장된 JSON 파일의 개수
    keypoint_num = len(keypoint_file_list)
    print(keypoint_num)

    # JSON 파일의 내용을 저장 할 리스트
    all_keypoint_list = []

    for i in range(keypoint_num):
        # JSON 파일의 경로와 파일명
        json_file_path = f"{KEYPOINT_PATH}/frame_{i}_keypoints.json"

        with open(json_file_path, "r") as keypoint_json_file:
            # json.load(keypoint_json_file) : 키포인트가 저장된 JSON 파일의 내용 리턴
            keypoint_json = json.load(keypoint_json_file)

            if "people" in keypoint_json and len(keypoint_json["people"]) > 0:
                # keypoint_json["people"][0]["pose_keypoints_2d"] : 포즈의 키포인트 리턴
                pose_keypoints_2d = keypoint_json["people"][0]["pose_keypoints_2d"]

                # keypoint_json["people"][0]["hand_left_keypoints_2d"] : 왼손의 키포인트 리턴
                hand_left_keypoints_2d = keypoint_json["people"][0]["hand_left_keypoints_2d"]

                # keypoint_json["people"][0]["hand_right_keypoints_2d"] : 오른손 키포인트 리턴
                hand_right_keypoints_2d = keypoint_json["people"][0]["hand_right_keypoints_2d"]

                # pose_keypoints_2d : 포즈의 키포인트가 저장된 리스트 각 keypoint 마다 x좌표 y좌표 확률 3개씩 저장
                # np.array(pose_keypoints_2d) : numpy 배열로 변환
                # reshape(-1,3) : 리스트를 3열로 변환 행은 생략 (-1)
                pose_keypoint_arr = np.array(pose_keypoints_2d).reshape(-1, 3)

                # hand_left_keypoints_2d : 왼손의 키포인트가 저장된 리스트 각 keypoint 마다 x좌표 y좌표 확률 3개씩 저장
                # np.array(hand_left_keypoints_2d) : numpy 배열로 변환
                # reshape(-1,3) : 리스트를 3열로 변환 행은 생략 (-1)

                hand_left_keypoint_arr = np.array(hand_left_keypoints_2d).reshape(-1, 3)

                # hand_right_keypoints_2d: 오른손의 키포인트가 저장된 리스트 각 keypoint 마다 x좌표 y좌표 확률 3개씩 저장
                # np.array(hand_right_keypoints_2d) : numpy 배열로 변환
                # reshape(-1,3) : 리스트를 3열로 변환 행은 생략 (-1)

                hand_right_keypoint_arr = np.array(hand_right_keypoints_2d).reshape(-1, 3)

                # pose_keypoint_arr[: , :2] : pose_keypoint_arr (포즈 키포인트의 x좌표 y좌표 확률 저장의
                #                            모든 행 (:)  2열 미만 ( : 2   ) 리턴 => x,y 좌표 리턴

                # hand_left_keypoint_arr[: , :2] : hand_left_keypoint_arr (왼손 키포인트의 x좌표 y좌표 확률 저장의
                #                                 모든 행 (:)  2열 미만 ( : 2   ) 리턴 => x,y 좌표 리턴

                # hand_right_keypoint_arr[: , :2] : hand_right_keypoint_arr (오른손 키포인트의 x좌표 y좌표 확률 저장의
                #                                 모든 행 (:)  2열 미만 ( : 2   ) 리턴 => x,y 좌표 리턴

                # np.concatenate () : 배열들을 합쳐서 새로운 배열 생성
                # pose_keypoint_arr[: , :2] : 포즈 키포인트의 x,y 좌표
                # hand_left_keypoint_arr[: , :2] : 왼손 키포인트의 x,y 좌표
                # hand_right_keypoint_arr[: , :2] : 오른손 키포인트의 x,y 좌표

                # np.concatenate((pose_keypoint_arr[: , :2], hand_left_keypoint_arr[: , :2],hand_right_keypoint_arr[: , :2]))
                # : 포즈 왼손 오른손 키포인트의 x,y 좌표를 합쳐서 새로운 배열 생성
                keypoint_arr = np.concatenate(
                    (pose_keypoint_arr[:, :2], hand_left_keypoint_arr[:, :2], hand_right_keypoint_arr[:, :2]))

                # keypoint_arr을 all_keypoint_list에 추가
                all_keypoint_list.append(keypoint_arr.tolist())

    # np.array(all_keypoint_list , dtype="float32") : all_keypoint_list를 실수("float32") 타입 배열로 변환
    test_keypoint_arr = np.array(all_keypoint_list, dtype="float32")
    print("test_keypoint_arr.shape=",test_keypoint_arr.shape)

    frame_keypoint_list = []

    # test_keypoint_arr 의 1행 씩 arr_row에 대입
    for arr_row in test_keypoint_arr:
        # arr_row[:,0] : arr_row의 0열 -> keypoint 의 x좌표
        x = arr_row[:, 0]
        # arr_row[:,1] : arr_row의 1열 -> keypoint 의 y좌표
        y = arr_row[:, 1]

        # np.mean(x) : keypoint 의 x좌표 평균
        mean_x = np.mean(x)
        # np.mean(y) : keypoint 의 y좌표 평균
        mean_y = np.mean(y)

        # np.std(x) : keypoint 의 x좌표 표준편차
        std_x = np.std(x)
        # np.std(y) : keypoint 의 y좌표 표준편차
        std_y = np.std(y)

        # keypoint 의 x,y 좌표에서 평균을 빼고 표준편차로 나눔
        # 평균 => 0  표준편차 =>1
        normal_x = (x - mean_x) / std_x
        normal_y = (y - mean_y) / std_y

        # 0번째 열에 normal_x 대입
        arr_row[:, 0] = normal_x
        # 1번째 열에 normal_y 대입
        arr_row[:, 1] = normal_y

        # arr_row.flatten() : arr_row(2차원배열) 을 1차원 배열로 변환
        # tolist() : 배열을 리스트로 변환
        # frame_keypoint_list.append() : frame_keypoint에 추가
        frame_keypoint_list.append(arr_row.flatten().tolist())

    # len(frame_keypoint_list)  가 MAX_FRAME_NUM (240) 미만
    if len(frame_keypoint_list) < MAX_FRAME_NUM:
        # np.random.choice(max, num) : 0~max 미만의 난수 num 개 리턴

        # np.random.choice(len(frame_keypoint_list) , MAX_FRAME_NUM - len(frame_keypoint_list) ) :
        #     len(frame_keypoint_list) : frame_keypoint_list의 행의 개수 미만의 난수
        #     MAX_FRAME_NUM - len(frame_keypoint_list)  개 리턴
        random_choice_frame = np.random.choice(len(frame_keypoint_list), MAX_FRAME_NUM - len(frame_keypoint_list))

        # random_choice_frame 에 저장된 난수를 정렬
        random_choice_frame.sort()

        # random_choice_frame에 저장된 난수를 random_idx에 대입
        for random_idx in random_choice_frame:
            # frame_keypoint_list[random_idx] :frame_keypoint에서 random_idx번째 프레임
            # frame_keypoint_list.append() : frame_keypoint_list에 추가
            # => 240 행(프레임 키포인트 x y 좌표)으로 길이를 맞춤
            frame_keypoint_list.append(frame_keypoint_list[random_idx])
    else:
        # frame_keypoint_list[: MAX_FRAME_NUM] : MAX_FRAME_NUM 행(프레임 키포인트 x y 좌표) 리턴
        frame_keypoint_list = frame_keypoint_list[: MAX_FRAME_NUM]

    # frame_keypoint_list[::-1] : frame_keypoint의 행을 역순으로 변환
    frame_keypoint_list = frame_keypoint_list[::-1]

    # frame_keypoint_list 를 배열로 변환
    X_test = np.array([frame_keypoint_list], dtype="float32")

    # 수어 keypoint 가 저장된 X_test를 tensor객체로 변환
    X_test = torch.tensor(X_test)
    print("X_test.shape=", X_test.shape)

    # model을 GPU에 저장
    model.cuda()

    # X_test.cuda() :X_test을 GPU에 저장
    # float() : 실수 타입으로 변환
    X_test = X_test.cuda().float()

    # model을 실행 (평가) 모드로 변환 => 학습 않함
    model.eval()

    # torch.no_grad() : model의 값을 수정하지 않음
    with torch.no_grad():
        # model.encoder(X_test) : 인코더 실행
        encoder_outputs, hidden = model.encoder(X_test)

    sentence_list = []

    last_word = None
    for index in range(30):
        # Decoder의 첫번째 입력 [STA_INDEX] 를 tensor객체로 변환
        start_tensor = torch.tensor([STA_INDEX], dtype=torch.long).to(device)

        # torch.no_grad() : model의 값을 수정하지 않음
        with torch.no_grad():
            # Decoder 실행
            decoder_output, hidden = model.decoder(start_tensor, hidden, encoder_outputs)

        # decoder_output : Decoder의 예측결과
        # decoder_output.argmax(1) : Decoder의 예측결과에서 가장 확률이 높은 단어 인덱스 리턴 => 2차원 배열
        # .item() : 배열에 저장된 데이터 리턴
        pred_token = decoder_output.argmax(1).item()
        print("pred_token=", pred_token)
        # index_to_word.get(pred_token) : 예측된 단어 인덱스를 단어로 변환
        print("pred_word=", index_to_word.get(pred_token))
        print("=" * 100)

        if pred_token == END_INDEX:
            break

        # index_to_word.get(pred_token) : 예측된 단어 인덱스를 단어로 변환
        pred_word = index_to_word.get(pred_token)

        if last_word!=pred_word:
            # 예측된 단어를 추가
            # index_to_word.get(pred_token) : 예측된 단어 인덱스를 단어로 변환
            sentence_list.append(index_to_word.get(pred_token))

        last_word =  pred_word

    return " ".join(sentence_list)

@app.route("/tts", methods=['POST'])
def texttospeech():
    # JSON 형식의 데이터 받기
    data = request.json
    text = data.get('text')

    # TTS 변환 및 MP3 파일 저장
    tts = gTTS(text=text, lang='ko', slow=False)
    nowTime =  str(datetime.datetime.utcnow().timestamp())
    mp3_file_path = 'C:\Lecture\web-workspace\spring-https\src\main\\resources\static\mp3\ex_ko' + nowTime + '.mp3'


    try:
        tts.save(mp3_file_path)
    except Exception as e:
        return {"error": "Failed to save MP3 file", "details": str(e)}, 500

    # MP3 파일을 클라이언트에게 전송
    return nowTime

if __name__ == '__main__':
    app.run(ssl_context=('./certificateSSL/cert.pem', './certificateSSL/key.pem'), host='0.0.0.0', port=5000)
