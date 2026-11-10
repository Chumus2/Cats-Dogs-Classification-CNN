import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="😺", layout="centered")

st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or a dog, and the model will classify it")

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Analyzing the image..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    prediction = data["prediction"]
                    confidence = data["confidence_percent"]

                    st.success("Analysis complete!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Prediction", value=prediction)
                    with col2:
                        st.metric(label="Confidence", value=f"{confidence}%")

                    st.progress(confidence / 100)
                else:
                    st.error(f"Server error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Make sure Uvicorn is running!")