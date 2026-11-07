import os
import configs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_dataset():
    if not os.path.exists(configs.DATA_DIR):
        print(f"Dataset not exist ({configs.DATA_DIR})")
        return None

    transformer = transforms.Compose([
        transforms.Resize((configs.IMG_WIDTH, configs.IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    full_dataset = datasets.ImageFolder(configs.DATA_DIR, transform=transformer)
    if len(full_dataset) == 0:
        print(f"Dataset is empty. In folder {configs.DATA_DIR} images not exist")
        return None

    print(f"Classes found: {full_dataset.classes}")
    print(f"Summary images: {len(full_dataset)}")

    train_size = int(configs.TRAIN_SIZE * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train, val = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train, configs.BATCH_SIZE, shuffle=True, num_workers=configs.NUM_WORKERS)
    val_loader = DataLoader(val, configs.BATCH_SIZE, shuffle=False, num_workers=configs.NUM_WORKERS)
    print("Dataset loaded")
    print(f"Train size: {len(train_loader)} | Val size: {len(val_loader)}")

    return train_loader, val_loader