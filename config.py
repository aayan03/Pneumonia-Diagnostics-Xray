"""Shared configuration for the pneumonia detection project."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Dataset (only needed for local training, never uploaded to GitHub)
DATASET_DIR = BASE_DIR / "dataset" / "chest_xray"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Model and output directories
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Trained model is committed to Git so Streamlit Cloud can load it directly.
MODEL_PATH = MODEL_DIR / "pneumonia_efficientnetb0.h5"

# Training hyperparameters
IMAGE_SIZE = 224
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def create_required_directories() -> None:
    """Create folders needed at runtime."""
    for directory in [MODEL_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
