function sendText() {
    const textToSend = '안녕하세요, 텍스트 음성 변환입니다!'; // 전송할 텍스트

    axios.post('http://localhost:5000/tts', {
        text: textToSend
    })
    .then(response => {
        // 서버에서 반환한 파일 URL
        const audioUrl = response.data.file_url;

        // 오디오 요소 생성 및 설정
        const audio = new Audio(audioUrl);
        audio.play(); // 오디오 재생
    })
    .catch(error => {
        console.error('Error fetching the audio:', error);
    });
}

// 버튼 클릭 시 sendText 함수 호출
document.getElementById('sendButton').addEventListener('click', sendText);
