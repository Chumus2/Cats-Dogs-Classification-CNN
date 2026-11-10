import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import configs
from PIL import Image
from io import BytesIO
from torchvision import transforms
from model import CatDogClassifier
from colorama import Fore, Style, init
from fastapi import FastAPI, File, UploadFile


init(autoreset=True)
app = FastAPI(title="Cat vs Dog Classifier")

model = CatDogClassifier(
    in_channels=configs.IN_CHANNELS,
    out_channels=configs.OUT_CHANNELS,
    kernel_size=configs.KERNEL_SIZE,
    stride=configs.STRIDE,
    padding=configs.PADDING
)

model_path = os.path.join(configs.OUTPUT_DIR, "best_model.pth")
if not os.path.exists(model_path):
    print(f"{Fore.RED}[Error]{Style.RESET_ALL} Checkpoint not found at {model_path}")

checkpoint = torch.load(model_path, map_location=configs.DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(configs.DEVICE)
model.eval()

transformer = transforms.Compose([
    transforms.Resize((configs.IMG_WIDTH, configs.IMG_HEIGHT)),
    transforms.ToTensor(),
])
classes = ["Cat", "Dog"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    input_tensor = transformer(image).unsqueeze(0).to(configs.DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    pred_label = classes[predicted_class.item()]
    conf_score = round(confidence.item() * 100, 2)

    return {
        "filename": file.filename,
        "prediction": pred_label,
        "confidence_percent": conf_score,
    }