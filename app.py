import streamlit as st
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import tempfile
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

session = ort.InferenceSession('best_emotion_model.onnx')
input_name = session.get_inputs()[0].name

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0).astype(np.float32)
    return face


def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    result = session.run(None, {input_name: processed})
    class_idx = np.argmax(result[0])
    return classes[class_idx]


st.title("Facial Emotion Recognition App")
st.sidebar.title("Select Mode")

app_mode = st.sidebar.selectbox(
    "Choose an option",
    ["Image Upload", "Video Upload", "Webcam"]
)

# ---------------------------- IMAGE Upload ----------------------------
if app_mode == "Image Upload":
    st.subheader("Upload an Image to Detect Emotion")

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image_np = np.array(image.convert('RGB'))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            st.warning("No faces detected in the image.")
        else:
            for (x, y, w, h) in faces:
                face_img = image_np[y:y + h, x:x + w]
                label = predict_emotion(face_img)
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image_np, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)
            st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Result", use_column_width=True)


# ---------------------------- VIDEO Upload ----------------------------
elif app_mode == "Video Upload":
    st.subheader("Upload a Video to Detect Emotions")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = tfile.name + '_out.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        st.info('Processing video... Please wait')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                label = predict_emotion(face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            out.write(frame)

        cap.release()
        out.release()

        st.success('Video processing complete!')
        st.subheader("Processed Video")
        st.video(output_path)

        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

        os.remove(tfile.name)
        os.remove(output_path)


# ---------------------------- WEBCAM ----------------------------
elif app_mode == "Webcam":
    st.subheader("Webcam — Real-Time Emotion Detection")
    st.info("Allow camera access and click START to begin live emotion detection. Detection updates continuously as your expression changes.")

    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class EmotionDetector(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_img = img[y:y + h, x:x + w]
                label = predict_emotion(face_img)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionDetector,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )
