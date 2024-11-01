# Voice Harmony

## Project Overview
This project focuses on improving communication accessibility for the hearing impaired through the development of an AI-powered application. The system combines sign language recognition, lip reading, and speech recognition to enhance communication for individuals with hearing impairments. The primary goal is to foster a more inclusive society, allowing hearing-impaired individuals to communicate seamlessly in environments where spoken language predominates. The application employs an Attention model for real-time sign language detection, LSTM for processing temporal sequences, and integrates Spring Boot and Flask for web-based service delivery.

### Awards & Recognition
- The paper titled *"Improving Communication Access for the Hearing Impaired through AI"* was awarded the **Excellence Award** at the **2024 ICT Mentoring Korean Institute of Information Processing Conference (ACK 2024)**.
- Additionally, the project was a **finalist in the 2024 ProBono ICT Mentoring Competition**.

## Key Features

1. **Real-Time Sign Language Recognition**:
   - The system captures hand gestures using an Attention model to focus on sign language movements, converting them into text in real-time.
   - LSTM models are utilized to interpret sequential hand gestures, enabling accurate recognition of dynamic signs.

2. **Lip Movement Recognition**:
   - Lip movements are detected and analyzed using Transformer models, providing accurate text output based on visual data.

3. **Speech Recognition (STT)**:
   - The OpenAI Whisper model is employed for converting spoken language into text, facilitating smooth communication.

4. **Text-to-Speech Conversion (TTS)**:
   - Converts the recognized text (or user-inputted text) into speech, offering auditory feedback for the hearing impaired.
   - Google’s Text-to-Speech API is used for TTS, providing high-quality, natural-sounding voice output.

## Technologies & Tools

- **Languages**: Java (Spring Boot), Python (Flask), HTML, CSS, JavaScript
- **Frameworks**: Spring Boot 3.3.2, Flask, Thymeleaf
- **Machine Learning Models**: Attention model for sign language detection, LSTM for temporal sequence learning
- **Computer Vision**: Mediapipe for extracting hand landmarks, Transformer for lip reading
- **Development Tools**: PyCharm, Eclipse, Tomcat (Web server), Jupyter Notebook
- **Speech-to-Text & Text-to-Speech**: OpenAI Whisper for STT and Google Text-to-Speech API for TTS

## System Architecture

1. **Web Interface**:
   - Developed using Spring Boot and Thymeleaf, the web interface captures real-time webcam input and displays text output dynamically.
   - The webcam feed is processed on the Flask server, where machine learning models provide predictions based on sign language, lip movement, and speech data.

2. **Flask Server**:
   - Serves as the backend for processing input data. The Flask server utilizes Attention, LSTM, and Whisper models to predict and output text based on the input data.

3. **Data Preprocessing**:
   - **Sign Language Data**: Hand landmarks are extracted using Mediapipe, then processed with Attention mechanisms for focused gesture recognition.
   - **Lip Movement Data**: The OLKAVS lip reading dataset was used for lip movement recognition. Video frames were processed to extract only the mouth region, which was then used to train the Transformer model. The Transformer model, known for its ability to handle long sequences, learns to interpret the subtle variations in mouth shapes to convert them into text.
   - **Speech Data**: Audio is processed using the Whisper model, converting speech into text that is displayed on the user interface.

## Demo Video

Check out the project in action on YouTube by following this link:  
[Sign Language Alphabet Detection Demo](https://youtube.com/your-demo-link)

## Anticipated Impact & Applications

- **Enhanced Communication for the Hearing Impaired**: This system facilitates real-time recognition of sign language, lip movements, and speech, bridging communication gaps for the hearing impaired.
- **Practical Applications**: This technology can be applied in healthcare, education, public services, and everyday interactions, promoting better accessibility and inclusivity.
