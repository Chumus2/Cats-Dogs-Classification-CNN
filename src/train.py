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
        loop = tqdm(train, desc=f"Epoch [{epoch + 1}/{configs.EPOCHS}]", leave=False)

        for images, labels in loop:
            images = images.to(configs.DEVICE)
            labels = labels.to(configs.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_avg_loss = train_loss / len(train)


if __name__ == "__main__":
    train_model()