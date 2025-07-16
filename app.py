import streamlit as st
import os
from PIL import Image
from ai_agent.predict_utils import predict_image, predict_video

st.set_page_config(page_title="Deepfake Detection AI", layout="centered")
st.title("🧠 Deepfake Detection AI Agent")
st.write("Upload an image or video to detect whether it's Real or Deepfake.")

option = st.radio("Choose input type:", ("🖼️ Predict Image", "🎥 Predict Video"))

if option == "🖼️ Predict Image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save image temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        # Run prediction
        result = predict_image(temp_path)
        st.success(f"🧪 Prediction: **{result}**")
        os.remove(temp_path)

elif option == "🎥 Predict Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save video temporarily
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(temp_path)

        # Run prediction
        result = predict_video(temp_path)
        st.success(f"🧪 Prediction: **{result}**")
        os.remove(temp_path)


