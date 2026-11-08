import os
import glob
from urllib import error

import torch
import configs
import warnings
import torch.nn as nn
from dataset import load_dataset
from model import CatDogClassifier
from colorama import Fore, Style, init
from torch.optim.lr_scheduler import ReduceLROnPlateau


warnings.filterwarnings("ignore")
init(autoreset=True)

def train_model():

    print(f"Using device: {configs.DEVICE}")

    train, val = load_dataset()
    if train is None or val is None:
        print(f"{Fore.RED}Dataset is empty")
        return None

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

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=configs.FACTOR, patience=configs.SCHEDULER_PATIENCE)

    if not os.path.exists(configs.OUTPUT_DIR):
        os.makedirs(configs.OUTPUT_DIR)

    for old_checkpoint in glob.glob(os.path.join(configs.OUTPUT_DIR, "*.pth")):
        os.remove(old_checkpoint)

    best_val_loss = float("inf")
    patience_counter = 0

    print("\n" + "=" * 50)
    print("STARTING TRAINING PROCESS")
    print("=" * 50 + "\n")
    for epoch in range(configs.EPOCHS):

        model.train()
        train_loss = 0.0

        for images, labels in train:
            images = images.to(configs.DEVICE)
            labels = labels.to(configs.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_avg_loss = train_loss / len(train)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val:
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

        scheduler.step(val_avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch + 1}/{configs.EPOCHS}] "
              f"| Train Loss: {train_avg_loss:.4f} "
              f"| {Fore.RED}Val Loss: {val_avg_loss:.4f} {Style.RESET_ALL}"
              f"| {Fore.RED}Val Accuracy: {val_acc:.2f}% {Style.RESET_ALL}"
              f"| LR: {current_lr}")

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_avg_loss,
                "val_loss": val_avg_loss,
                "accuracy": val_acc,
                "learning_rate": current_lr
            }
            model_path = os.path.join(configs.OUTPUT_DIR, f"best_model.pth")
            torch.save(checkpoint, model_path)
            print(f"{Fore.GREEN}[Checkpoint]{Style.RESET_ALL} Improvement found! Saved new best_model.pth {Style.BRIGHT}(Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

            if patience_counter >= configs.PATIENCE:
                print(f"{Fore.RED}[Early Stopping]{Style.RESET_ALL} Validation loss did not improve for {configs.PATIENCE} consecutive epochs. Training stopped early.")
                break

    print("\n" + "=" * 50)
    print("ENDING TRAINING PROCESS")
    print("=" * 50 + "\n")

    if os.path.exists(os.path.join(configs.OUTPUT_DIR, "best_model.pth")):
        best_checkpoint = torch.load(os.path.join(configs.OUTPUT_DIR, "best_model.pth"))
        print("Best Model:")
        print(f"Epoch: {best_checkpoint['epoch']}/{configs.EPOCHS}")
        print(f"Train Loss: {best_checkpoint['train_loss']:.4f}")
        print(f"{Fore.RED}Val Loss: {best_checkpoint['val_loss']:.4f}")
        print(f"{Fore.RED}Accuracy: {best_checkpoint['accuracy']:.2f}%")
        print(f"Final LR: {best_checkpoint['learning_rate']:.6f}")
        print("=" * 50 + "\n")
    else:
        print(f"{Fore.RED}Checkpoint not found")


if __name__ == "__main__":
    train_model()