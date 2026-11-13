# Cat vs. Dog Image Classification System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

A full-stack computer vision application featuring a custom Convolutional Neural Network (CNN) built in PyTorch, served via a high-performance FastAPI backend, and delivered through a modern, responsive Streamlit interface.

---

## 📌 Overview

This project implements an end-to-end machine learning solution for binary image classification (Cats vs. Dogs). 
It separates model inference from the user interface using a decoupled microservice architecture:
- **Core ML:** Custom PyTorch CNN trained with robust data augmentation and clean modular code.
- **Backend API:** FastAPI service providing fast, validated HTTP endpoints for inference.
- **Frontend UI:** Interactive Streamlit web application supporting single or batch image classification, live progress indicators, and custom dark-theme styling.

---

## 🛠️ Tech Stack & Architecture

- **Machine Learning & Deep Learning:** PyTorch, Torchvision, PIL (Pillow)
- **Backend / API:** FastAPI, Uvicorn, Python-Multipart
- **Frontend UI:** Streamlit, Custom CSS
- **Project Structure:** Modular design separating data processing, model definitions, API routes, and UI components.

---

## 📂 Project Architecture

```text
Cats_Dogs_Classification/
│
├── cv/                 # Core computer vision codebase and model logic
│   ├── src/            # Backend implementation and inference handlers
│   ├── app.py          # Streamlit frontend user interface
│   └── ...             # Model weights, data preprocessing, and utilities
│
├── docker/             # Docker infrastructure and configuration
│   ├── Dockerfile      # Optimized multi-layer container image
│   ├── docker-compose.yml # Container orchestration and networking
│   └── requirements.txt # Python package dependencies
│
├── app.py              # Streamlit frontend script
└── README.md
```

---

## 🚀 Getting Started & Installation

### Option 1: Quick Start with Docker Compose (Recommended)

The easiest way to run the entire application in isolated environments is using Docker Compose.
1. **Clone the repository:**
```bash
git clone https://github.com/Chumus2/Cats-Dogs-Classification-CNN.git
cd Cats_Dogs_Classification
```

2. **Download and Prepare the Dataset:**
Link: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
Downloaded dataset name **data** and put into cv directory.

3. Create *checkpoints* directory in **cv** and run *train.py* to build model stats.

4. **Run with Docker Compose:**
```bash
cd docker
docker-compose up --build
```

5. **Access the Application:** 
Website: http://localhost:8501
FastAPI documentation: http://localhost:8000/docs

### Option 2: Local Setup (Without Docker)

If you prefer to run the application directly on your host machine without containers, follow these steps:
1. **Clone the repository:**
```bash
git clone https://github.com/Chumus2/Cats-Dogs-Classification-CNN.git
cd Cats_Dogs_Classification
```

1. **Download and Prepare the Dataset:**
Link: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
Downloaded dataset name **data** and put into cv directory.

2. **Set Up a Python Virtual Environment:**
```bash
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate
# On macOS / Linux:
source venv/bin/activate
```

3. **Install Dependencies:**
```bash
pip install --upgrade pip
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r docker/requirements.txt
```

4. Create *checkpoints* directory in **cv** and run *train.py* to build model stats.

5. **Run the Backend and Frontend Services:**
Open two separate terminal windows (both with your virtual environment activated) to run the microservices concurrently:
   1. *Start the FastAPI Backend:*
   ```bash 
   uvicorn cv.src.api:app --host 0.0.0.0 --port 8000
   ```
   
   2. *Start the Streamlit Frontend:*
   ```bash
   streamlit run app.py
   ```
   
6. **Access the Application:**
Website: http://localhost:8501
FastAPI documentation: http://localhost:8000/docs
