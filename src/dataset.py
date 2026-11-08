import os
import torch
import configs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_dataset():
    if not os.path.exists(configs.DATA_DIR):
        print(f"Dataset not exist ({configs.DATA_DIR})")
        return None, None

    train_transformer = transforms.Compose([
        transforms.Resize((configs.IMG_WIDTH, configs.IMG_HEIGHT)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    val_transformer = transforms.Compose([
        transforms.Resize((configs.IMG_WIDTH, configs.IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    temp_dataset = datasets.ImageFolder(configs.DATA_DIR, transform=None)
    if len(temp_dataset) == 0:
        print(f"Dataset is empty. In folder {configs.DATA_DIR} images not exist")
        return None

    print(f"Classes found: {temp_dataset.classes}")
    print(f"Summary images: {len(temp_dataset)}")

    train_size = int(configs.TRAIN_SIZE * len(temp_dataset))
    val_size = len(temp_dataset) - train_size

    indices = torch.randperm(len(temp_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train = datasets.ImageFolder(configs.DATA_DIR, transform=train_transformer)
    val = datasets.ImageFolder(configs.DATA_DIR, transform=val_transformer)

    train_subset = Subset(train, train_indices)
    val_subset = Subset(val, val_indices)

    train_loader = DataLoader(train_subset, configs.BATCH_SIZE, shuffle=True, num_workers=configs.NUM_WORKERS)
    val_loader = DataLoader(val_subset, configs.BATCH_SIZE, shuffle=False, num_workers=configs.NUM_WORKERS)
    print(f"Train size: {len(train_loader)} | Val size: {len(val_loader)}")

    return train_loader, val_loader