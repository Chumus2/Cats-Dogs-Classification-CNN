FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt .
RUN pip install --no-cache-dir torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements_backend.txt

COPY cv ./cv

EXPOSE 8000 8501