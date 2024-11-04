// let remoteStreamElement = document.querySelector('#remoteStream');
 let localStreamElement = document.querySelector('#localStream');
 const myKey = Math.random().toString(36).substring(2, 11);
var flag = 1;
 let pcListMap = new Map();
 let roomId;
 let otherKeyList = [];
 let localStream = undefined;
 let mediaRecordersMap = new Map();
 let videoRecordersMap = new Map();
 let ttsMap = new Map();
  
 const startCam = async () =>{
     if(navigator.mediaDevices !== undefined){
         await navigator.mediaDevices.getUserMedia({ audio: true, video : true })
             .then(async (stream) => {
                 console.log('Stream found');
				 //웹캠, 마이크의 스트림 정보를 글로벌 변수로 저장한다.
                 localStream = stream;
                 // Disable the microphone by default
                 stream.getAudioTracks()[0].enabled = true;
                 localStreamElement.srcObject = localStream;
                 // Connect after making sure that local stream is availble
    
             }).catch(error => {
                 console.error("Error accessing media devices:", error);
             });
     }
    
 }
    
 // 소켓 연결
 const connectSocket = async () =>{
     const socket = new SockJS('/signaling');
     stompClient = Stomp.over(socket);
     stompClient.debug = null;
    
     stompClient.connect({}, function () {
         console.log('Connected to WebRTC server');
            
		 //iceCandidate peer 교환을 위한 subscribe
         stompClient.subscribe(`/topic/peer/iceCandidate/${myKey}/${roomId}`, candidate => {
             const key = JSON.parse(candidate.body).key
             const message = JSON.parse(candidate.body).body;
    
 			 // 해당 key에 해당되는 peer 에 받은 정보를 addIceCandidate 해준다.
             pcListMap.get(key).addIceCandidate(new RTCIceCandidate({candidate:message.candidate,sdpMLineIndex:message.sdpMLineIndex,sdpMid:message.sdpMid}));
    
         });
    				
		 //offer peer 교환을 위한 subscribe
         stompClient.subscribe(`/topic/peer/offer/${myKey}/${roomId}`, offer => {
             const key = JSON.parse(offer.body).key;
             const message = JSON.parse(offer.body).body;
    						
			 // 해당 key에 새로운 peerConnection 를 생성해준후 pcListMap 에 저장해준다.
             pcListMap.set(key,createPeerConnection(key));
			 // 생성한 peer 에 offer정보를 setRemoteDescription 해준다.
             pcListMap.get(key).setRemoteDescription(new RTCSessionDescription({type:message.type,sdp:message.sdp}));
             //sendAnswer 함수를 호출해준다.
 						sendAnswer(pcListMap.get(key), key);
    
         });
    				
		 //answer peer 교환을 위한 subscribe
         stompClient.subscribe(`/topic/peer/answer/${myKey}/${roomId}`, answer =>{
             const key = JSON.parse(answer.body).key;
             const message = JSON.parse(answer.body).body;
    						
			 // 해당 key에 해당되는 Peer 에 받은 정보를 setRemoteDescription 해준다.
             pcListMap.get(key).setRemoteDescription(new RTCSessionDescription(message));
    
         });
    				
	     //key를 보내라는 신호를 받은 subscribe
         stompClient.subscribe(`/topic/call/key`, message =>{
			 //자신의 key를 보내는 send
             stompClient.send(`/app/send/key`, {}, JSON.stringify(myKey));
    
         });
    				
		 //상대방의 key를 받는 subscribe
         stompClient.subscribe(`/topic/send/key`, message => {
             const key = JSON.parse(message.body);
    						
			 //만약 중복되는 키가 ohterKeyList에 있는지 확인하고 없다면 추가해준다.
             if(myKey !== key && otherKeyList.find((mapKey) => mapKey === myKey) === undefined){
                 otherKeyList.push(key);
             }
         });
    
     });
 }
 
function startRecordingg(ox, key) {
	console.log(mediaRecordersMap);
	console.log(videoRecordersMap);
	flag = ox;
	mediaRecordersMap.get(key).start();
    console.log('Recording started');
    document.getElementById('recordButton' + key).disabled = true;
    document.getElementById('stopButton' + key).disabled = true;
    document.getElementById('voicerecordButton' + key).disabled = true; 
    document.getElementById('voicestopButton' + key).disabled = true;
    if (ox == 1) {
	    document.getElementById('stopButton' + key).disabled = false;
	} else {
	    document.getElementById('voicestopButton' + key).disabled = false;
	}
}

function endRecordingg(ox, key){
	mediaRecordersMap.get(key).stop();
    console.log('Recording stopped');
    document.getElementById('recordButton' + key).disabled = false;
    document.getElementById('stopButton' + key).disabled = true;
    document.getElementById('voicerecordButton' + key).disabled = false; 
    document.getElementById('voicestopButton' + key).disabled = true;
}


function forTTS(key){
	var text = document.getElementById("signLanguageToWord" + key).innerText;
	console.log(text);
	axios.post('https://localhost:5000/tts', {
            text: text
        })
        .then(function (response) {
        	ttsMap.set(key, "./mp3/ex_ko" + response.data + ".mp3");
        	try {
        		console.log(1);
        		document.getElementById("audio").src = audioSrc;
        		console.log(2);

        		document.getElementById("audio").load();
        		console.log(3);
        		document.getElementById("audio").play();
        		console.log(4);
            } catch (error) {
                console.error("이클립스가 파일을 감지중.", error);
                alert('이클립스가 파일을 감지중입니다. 잠시 후에 play 버튼을 눌러주세요')
            }
            
        })
        .catch(function (error) {
            console.error("Error converting text to speech:", error);
        });
}

function playTTS(key){
	const audio = document.getElementById("audio");
    try {
    	audio.src = ttsMap.get(key);
    	audio.load();
        audio.play();
    } catch (error) {
        console.error("이클립스가 파일을 감지중.", error);
        alert('아직도 이클립스가 파일을 감지중입니다. 잠시 후에 play 버튼을 다시 눌러주세요')
    }
}


let onTrack = (event, otherKey) => {
    
    if(document.getElementById(`${otherKey}`) === null){
		
		var inner = `<div class="rounded">
				    <video class="rounded" id="video${otherKey}" autoplay controls></video>
		</div>
	    <div class="rounded container-fluid d-flex flex-column" id="controls" style="width:500px; background-color: #f8f9fa;">
	    	<div class="row flex-fill">
    			<div class="col-12 d-flex justify-content-center align-items-center">
    				<p class="fw-semibold fs-4">수어 인식 서비스</p>
    			</div>
    			<div class="col-12 d-flex justify-content-evenly align-items-center">
			    	<button class="btn btn-danger" id="recordButton${otherKey}" onclick="startRecordingg(1, '${otherKey}')">수어 인식</button>
			    	<button class="btn btn-dark" id="stopButton${otherKey}" disabled onclick="endRecordingg(1, '${otherKey}')">중지</button>
		    	</div>
	    	</div>
	    	<div class="row flex-fill" style="background-color: #fff3cd;">
		    	<div class="col-12 d-flex justify-content-center align-items-center">
		    		<p class="fw-semibold fs-4">음성 인식 서비스</p>
		    	</div>
		    	<div class="col-12 d-flex justify-content-evenly align-items-center">
		            <button class="btn btn-danger" id="voicerecordButton${otherKey}" onclick="startRecordingg(2, '${otherKey}')">음성 인식</button>
		            <button class="btn btn-dark" id="voicestopButton${otherKey}" disabled onclick="endRecordingg(2, '${otherKey}')">중지</button>
	            </div>
	    	</div>
	    	<div class="row flex-fill" style="background-color: #e2e3e5;">
			    <div class="col-12 d-flex flex-column justify-content-center align-items-center">
			    	<p class="fw-semibold fs-4">TTS 서비스</p>
		        </div>
		        <div class="col-12 d-flex justify-content-evenly align-items-center">
		            <button class="btn btn-success" onclick="forTTS('${otherKey}')">tts</button>
		            <button class="btn btn-success" onclick="playTTS('${otherKey}')">play</button>
		       	</div>
	    	</div>
		</div>
		<divstyle="width: 500px; height: 480px;">
			<div class="team-item bg-white text-center rounded p-4 pt-0 mainDiv3" style="width: 500px; height: 480px;">
				<p class="fw-semibold fs-2">인식 결과</p>
	            <h1><span id="signLanguageToWord${otherKey}" class="forCheatClass"></span></h1>
	        </div>
		</div>`;
		
		//document.getElementById('remoteStreamDiv').insertAdjacentHTML('beforeend', inner);
		document.getElementById('remoteStreamDiv').innerHTML = inner;
		document.getElementById(`video${otherKey}`).srcObject = event.streams[0];
		
	    // MediaRecorder 초기화
        videoRecordersMap.set(`${otherKey}`, []);
        const mediaRecorder = new MediaRecorder(event.streams[0]);
        // 연결되면 키:미디어레코드로 저장
		mediaRecordersMap.set(`${otherKey}`, mediaRecorder);
		
		console.log(videoRecordersMap);
		console.log(mediaRecordersMap);
		
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                videoRecordersMap.get(`${otherKey}`).push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(videoRecordersMap.get(`${otherKey}`), { type: 'video/x-matroska' });
            var url = URL.createObjectURL(blob);
            document.getElementById(`video${otherKey}`).src = url;

            // 다운로드 링크 생성
            var a = document.createElement('a');
            a.href = url;
            a.download = 'recorded-video.webm'; // 파일 이름 설정
            document.body.appendChild(a); // 링크를 DOM에 추가
            a.click(); // 클릭하여 다운로드 시작
            document.body.removeChild(a); // 링크 제거
            
            
            const formData = new FormData();
            formData.append('file', blob, 'recorded-video.webm');

            // Axios를 사용하여 서버에 POST 요청
            if (flag == 1) {
	            document.getElementById(`signLanguageToWord${otherKey}`).innerHTML = "수어를 서버로 전송, 계산 중 ~~";
				// 메세지 입력
	            axios.post('https://localhost:5000/sign_language', formData, {
	                headers: {
	                    'Content-Type': 'multipart/form-data'
	                }
	            })
	            .then(response => {
	                console.log('File uploaded successfully:', response.data);
	                document.getElementById(`signLanguageToWord${otherKey}`).innerHTML = "수어 - 텍스트 변환 <br>"+ response.data;
	            })
	            .catch(error => {
	                console.error('Error uploading file:', error);
	                document.getElementById(`signLanguageToWord${otherKey}`).innerHTML = "전송오류 !!!!";
	            });
			} else if (flag == 2) {
				document.getElementById(`signLanguageToWord${otherKey}`).innerHTML = "목소리를 서버로 전송, 계산 중 ~~";
				// 메세지 입력
	            axios.post('https://localhost:5000/wave01', formData, {
	                headers: {
	                    'Content-Type': 'multipart/form-data'
	                }
	            })
	            .then(response => {
	                console.log('File uploaded successfully:', response.data);
	                document.getElementById(`signLanguageToWord${otherKey}`).innerHTML = "목소리 - 텍스트 변환 <br>"+ response.data;
	            })
	            .catch(error => {
	                console.error('Error uploading file:', error);
	                document.getElementById(`signLanguageToWord${otherKey}`).innerHTML = "전송오류 !!!!";
	            });
			}
            videoRecordersMap.set(`${otherKey}`, []);
        };



     }
    
 };
    
 const createPeerConnection = (otherKey) =>{
     const pc = new RTCPeerConnection();
     try {
         pc.addEventListener('icecandidate', (event) =>{
             onIceCandidate(event, otherKey);
         });
         pc.addEventListener('track', (event) =>{
             onTrack(event, otherKey);
         });
         if(localStream !== undefined){
             localStream.getTracks().forEach(track => {
                 pc.addTrack(track, localStream);
             });
         }
    
         console.log('PeerConnection created');
     } catch (error) {
         console.error('PeerConnection failed: ', error);
     }
     return pc;
 }
    
 let onIceCandidate = (event, otherKey) => {
     if (event.candidate) {
         console.log('ICE candidate');
         stompClient.send(`/app/peer/iceCandidate/${otherKey}/${roomId}`,{}, JSON.stringify({
             key : myKey,
             body : event.candidate
         }));
     }
 };
    
 let sendOffer = (pc ,otherKey) => {
     pc.createOffer().then(offer =>{
         setLocalAndSendMessage(pc, offer);
         stompClient.send(`/app/peer/offer/${otherKey}/${roomId}`, {}, JSON.stringify({
             key : myKey,
             body : offer
         }));
         console.log('Send offer');
     });
 };
    
 let sendAnswer = (pc,otherKey) => {
     pc.createAnswer().then( answer => {
         setLocalAndSendMessage(pc ,answer);
         stompClient.send(`/app/peer/answer/${otherKey}/${roomId}`, {}, JSON.stringify({
             key : myKey,
             body : answer
         }));
         console.log('Send answer');
     });
 };
    
 const setLocalAndSendMessage = (pc ,sessionDescription) =>{
     pc.setLocalDescription(sessionDescription);
 }
    
 //룸 번호 입력 후 캠 + 웹소켓 실행
 document.querySelector('#enterRoomBtn').addEventListener('click', async () =>{
     await startCam();
    
     if(localStream !== undefined){
         document.querySelector('#localStream').style.display = 'block';
         document.querySelector('#startSteamBtn').style.display = '';
     }
     roomId = document.querySelector('#roomIdInput').value;
     //document.querySelector('#roomIdInput').disabled = true;
     //document.querySelector('#enterRoomBtn').disabled = true;
     document.getElementById('mainDiv1').style.display = "none";
     document.getElementById('mainDiv2').style.display = "none";
     var divs = document.querySelectorAll('.mainDiv3');
     divs.forEach(function(div) {
         div.style.display = "block"; // 보이게 설정
     });
    
     await connectSocket();
 });
    
 // 스트림 버튼 클릭시 , 다른 웹 key들 웹소켓을 가져 온뒤에 offer -> answer -> iceCandidate 통신
 // peer 커넥션은 pcListMap 으로 저장
 document.querySelector('#startSteamBtn').addEventListener('click', async () =>{
     await stompClient.send(`/app/call/key`, {}, {});
    
     setTimeout(() =>{
    
         otherKeyList.map((key) =>{
             if(!pcListMap.has(key)){
                 pcListMap.set(key, createPeerConnection(key));
                 sendOffer(pcListMap.get(key),key);
             }
    
         });
    
     },1000);
 });