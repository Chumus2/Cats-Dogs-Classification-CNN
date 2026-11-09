import os
import glob
import torch
import configs
import warnings
from PIL import Image
from model import CatDogClassifier
from torchvision import transforms
from colorama import Fore, Style, init


warnings.filterwarnings("ignore")
init(autoreset=True)

def test_model():
    print(f"Using device: {configs.DEVICE}")

    model = CatDogClassifier(
        in_channels=configs.IN_CHANNELS,
        out_channels=configs.OUT_CHANNELS,
        kernel_size=configs.KERNEL_SIZE,
        stride=configs.STRIDE,
        padding=configs.PADDING,
    )
    model = model.to(configs.DEVICE)

    model_path = os.path.join(configs.OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"{Fore.RED}[Error]{Style.RESET_ALL} Checkpoint not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=configs.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    test_images = []

    for ext in image_extensions:
        files = glob.glob(os.path.join(configs.TEST_DIR, "**", ext), recursive=True)
        test_images.extend(files)
    test_images = sorted(list(set(test_images)))

    print(f"{Fore.YELLOW}[Info]{Style.RESET_ALL} Test images found: {len(test_images)}")

    transformer = transforms.Compose([
        transforms.Resize((configs.IMG_WIDTH, configs.IMG_HEIGHT)),
        transforms.ToTensor(),
    ])
    classes = ["Cat", "Dog"]

    print("\n" + "=" * 50)
    print("BATCH INFERENCE (PREDICTION TEST)")
    print("=" * 50)

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"{Fore.RED}[Error]{Style.RESET_ALL} Image not found at {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        input_tensor = transformer(img).unsqueeze(0).to(configs.DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        pred_label = classes[predicted_class.item()]
        conf_score = confidence.item() * 100

        print(f"File: {os.path.basename(img_path)} > Prediction: "
              f"{Fore.GREEN}{pred_label}{Style.RESET_ALL} "
              f"(Confidence: {conf_score:.2f}%)"
        )

    print("=" * 50 + "\n")


if __name__ == "__main__":
    test_model()