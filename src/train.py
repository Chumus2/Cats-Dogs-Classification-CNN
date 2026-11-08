import os
import glob
import torch
import configs
import warnings
import torch.nn as nn
from dataset import load_dataset
from model import CatDogClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau


warnings.filterwarnings("ignore")
def train_model():

    print(f"Using device: {configs.DEVICE}")

    train, val = load_dataset()
    if train is None or val is None:
        print("Dataset is empty")
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
              f"| Val Loss: {val_avg_loss:.4f} "
              f"| Val Accuracy: {val_acc:.2f}%"
              f"| LR: {current_lr}")

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_avg_loss,
                "accuracy": val_acc,
                "learning_rate": current_lr
            }
            model_path = os.path.join(configs.OUTPUT_DIR, f"best_model.pth")
            torch.save(checkpoint, model_path)
            print(f"[Checkpoint] Improvement found! Saved new best_model.pth (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

            if patience_counter >= configs.PATIENCE:
                print(f"[Early Stopping] Validation loss did not improve for {configs.PATIENCE} consecutive epochs. Training stopped early.")
                break

    print("\n" + "=" * 50)
    print("ENDING TRAINING PROCESS")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    train_model()