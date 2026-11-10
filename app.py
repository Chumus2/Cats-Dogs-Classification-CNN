import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="😺", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    [data-testid="stMetric"] {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff2222;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or a dog, and the model will classify it")

API_URL = "http://127.0.0.1:8000/predict"

with st.sidebar:
    st.header("Settings")
    st.info("This model uses a custom CNN trained on Cats vs Dogs dataset.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")
        prediction_button = st.button("Predict", width="stretch")

    with col2:
        if prediction_button:

            with st.spinner("Analyzing the image..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        data = response.json()
                        prediction = data["prediction"]
                        confidence = data["confidence_percent"]

                        st.success("Analysis complete!")
                        st.balloons()

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric(label="Prediction", value=prediction)
                        with col_m2:
                            st.metric(label="Confidence", value=f"{confidence}%")

                        st.progress(confidence / 100)
                    else:
                        st.error(f"Server error: {response.status_code}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Make sure Uvicorn is running!")