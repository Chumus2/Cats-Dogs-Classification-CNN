import requests
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Cat vs Dog AI Classifier",
    page_icon="😺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4b4b, #ff80bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff2222, #ff4b4b);
        box-shadow: 0 6px 16px rgba(255, 75, 75, 0.5);
        transform: translateY(-2px);
    }
    [data-testid="stMetric"] {
        background-color: #1f242d;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .image-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .stImage img {
        height: 450px !important;
        object-fit: cover !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🐾 Cat vs Dog Batch Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload multiple images and classify them all at once!</div>', unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"
MAX_IMAGES = 3

with st.sidebar:
    st.header("Control Panel")
    st.markdown("---")
    st.info("**Batch Mode:** Select multiple images (JPG, PNG, JPEG) to classify together.")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) > MAX_IMAGES:
        st.error(f"Limit exceeded! You can upload a maximum of {MAX_IMAGES} images. You selected {len(uploaded_files)}.")
        uploaded_files = uploaded_files[:MAX_IMAGES]

    st.markdown("---")

    if uploaded_files:
        st.write(f"**Selected images:** {len(uploaded_files)}")
        predict_button = st.button("Predict All Images")
    else:
        predict_button = False

    st.markdown("---")
    st.markdown("**Stack:** PyTorch • FastAPI • Streamlit")

if uploaded_files:
    if predict_button:
        st.markdown("### Classification Results")

        cols_per_row = 3
        for i in range(0, len(uploaded_files), cols_per_row):
            batch = uploaded_files[i:i + cols_per_row]
            cols = st.columns(len(batch))

            for idx, file in enumerate(batch):
                with cols[idx]:
                    st.markdown('<div class="image-card">', unsafe_allow_html=True)
                    image = Image.open(file).convert("RGB")
                    st.image(image, caption=file.name, width="stretch")

                    with st.spinner("Analyzing..."):
                        try:
                            files = {"file": (file.name, file.getvalue(), file.type)}
                            response = requests.post(API_URL, files=files)

                            if response.status_code == 200:
                                data = response.json()
                                prediction = data["prediction"]
                                confidence = data["confidence_percent"]

                                emoji = "🐱" if prediction == "Cat" else "🐶"
                                st.markdown(f"#### {emoji} {prediction}")
                                st.metric(label="Confidence", value=f"{confidence}%")
                                st.progress(confidence / 100)
                            else:
                                st.error(f"Error {response.status_code}")

                        except requests.exceptions.ConnectionError:
                            st.error("Backend offline!")

                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"### Uploaded Images ({len(uploaded_files)})")
        cols_per_row = 4

        for i in range(0, len(uploaded_files), cols_per_row):
            batch = uploaded_files[i:i + cols_per_row]
            cols = st.columns(len(batch))

            for idx, file in enumerate(batch):
                with cols[idx]:
                    image = Image.open(file).convert("RGB")
                    st.image(image, caption=file.name, width="stretch")

        st.info("Click **'Predict All Images'** in the sidebar to start batch classification!")
else:
    st.markdown("""
        <div style="text-align: center; padding: 40px; border: 2px dashed #30363d; border-radius: 16px; margin-top: 30px;">
            <h3>Please upload one or multiple images using the sidebar to get started!</h3>
            <p style="color: #8b949e;">Supports batch selection of JPG, JPEG, and PNG files.</p>
        </div>
    """, unsafe_allow_html=True)