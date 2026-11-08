import os
import glob
import torch
import configs
import warnings
from tqdm import tqdm
import torch.nn as nn
from dataset import load_dataset
from model import CatDogClassifier


warnings.filterwarnings("ignore")
def train_model():

    print(f"Using device: {configs.DEVICE}")
    train, val = load_dataset()
    model = CatDogClassifier(
        in_channels=configs.IN_CHANNELS,
        out_channels=configs.OUT_CHANNELS,
        kernel_size=configs.KERNEL_SIZE,
        stride=configs.STRIDE,
        padding=configs.PADDING,
    )
    model = model.to(configs.DEVICE)
    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR)

    if not os.path.exists(configs.OUTPUT_DIR):
        os.makedirs(configs.OUTPUT_DIR)

    for old_checkpoint in glob.glob(os.path.join(configs.OUTPUT_DIR, "*.pth")):
        os.remove(old_checkpoint)

    print("\n" + "=" * 50)
    print("STARTING TRAINING PROCESS")
    print("=" * 50 + "\n")
    for epoch in range(configs.EPOCHS):

        model.train()
        train_loss = 0.0
        train_loop = tqdm(train, desc=f"Epoch [{epoch + 1}/{configs.EPOCHS}] (Train)", leave=False)

        for images, labels in train_loop:
            images = images.to(configs.DEVICE)
            labels = labels.to(configs.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        train_avg_loss = train_loss / len(train)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            val_loop = tqdm(val, desc=f"Epoch [{epoch + 1}/{configs.EPOCHS}] (Val)", leave=False)
            for images, labels in val_loop:
                images = images.to(configs.DEVICE)
                labels = labels.to(configs.DEVICE)

                outputs = model(images)
                loss = crit(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_avg_loss = val_loss / len(val)
        val_acc = correct / total * 100

        print(f"Epoch [{epoch + 1}/{configs.EPOCHS}] "
              f"| Train Loss: {train_avg_loss:.4f} "
              f"| Val Loss: {val_avg_loss:.4f} "
              f"| Val Accuracy: {val_acc:.2f}%")

    print("\n" + "=" * 50)
    print("ENDING TRAINING PROCESS")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    train_model()