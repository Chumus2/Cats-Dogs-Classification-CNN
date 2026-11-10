import os
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "checkpoints")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_WIDTH = 224
IMG_HEIGHT = 224
TRAIN_SIZE = 0.8
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 45
IN_CHANNELS = 3
OUT_CHANNELS = 2
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
LR = 0.001
PATIENCE = 6
FACTOR = 0.5
SCHEDULER_PATIENCE = 2