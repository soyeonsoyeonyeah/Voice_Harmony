//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream;              //stream from getUserMedia()
var rec;                    //Recorder.js object
var input;                  //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //new audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");

function startRecording() {
    console.log("recordButton clicked");

    // Disable the record button until we get a success or fail from getUserMedia()
    recordButton.disabled = true;
    stopButton.disabled = false;

    navigator.mediaDevices.getUserMedia({audio: true, video: false}).then(function(stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        audioContext = new AudioContext({sampleRate: 16000});

        // assign to gumStream for later use
        gumStream = stream;

        // use the stream
        input = audioContext.createMediaStreamSource(stream);

        // Create the Recorder object and configure to record mono sound (1 channel) Recording 2 channels will double the file size
        rec = new Recorder(input, {numChannels: 1})

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function(err) {
        //enable the record button if getUserMedia() fails
        recordButton.disabled = false;
        stopButton.disabled = true;
    });
}

function createDownloadLink(blob) {
    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extension)
    var filename = new Date().toISOString();

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    //save to disk link
    link.href = url;
    link.download = filename+".wav"; //download forces the browser to download the file using the  filename
    link.innerHTML = "Save to disk";

    //add the new audio element to li
    li.appendChild(au);
    
    //add the filename to the li
    li.appendChild(document.createTextNode(filename+".wav "))

    //add the save to disk link to li
    li.appendChild(link);

    //add the li element to the ol
    recordingsList.appendChild(li);
}


function stopRecording() {
    console.log("stopButton clicked");

    //disable the stop button, enable the record to allow for new recordings
    stopButton.disabled = true;
    recordButton.disabled = false;

    //tell the recorder to stop the recording
    //rec.stop(); //stop microphone access
    //gumStream.getAudioTracks()[0].stop();

    //create the wav blob and pass it on to the Flask server
    //rec.exportWAV(function(blob) {
    //    var formData = new FormData();
    //    formData.append("file", blob, "recording.wav");

        // Send the file to the Flask server
    //    fetch("http://127.0.0.1:5000/wavtest", {
    //        method: "POST",
    //        body: formData,
    //        mode: 'cors'  // CORS 문제 해결을 위해 추가
    //    })
    //   .then(response => {
    //        if (!response.ok) {
    //            throw new Error('Network response was not ok');
    //        }
    //        return response.json();
    //    })
    //    .then(data => {
    //        console.log("File sent successfully:", data);
    //    })
    //    .catch(error => {
    //        console.error("Error sending file:", error);
    //    });
    //});
}



//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);