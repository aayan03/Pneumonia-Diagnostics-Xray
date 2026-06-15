"""Predict pneumonia from a single chest X-ray image."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH


def load_image(image_path):
    """Load and resize an image for EfficientNetB0."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_image(image_path):
    """Return predicted class, confidence, and pneumonia probability."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    image_array = load_image(image_path)
    pneumonia_probability = float(model.predict(image_array, verbose=0)[0][0])

    predicted_index = 1 if pneumonia_probability >= 0.5 else 0
    confidence = pneumonia_probability if predicted_index == 1 else 1 - pneumonia_probability
    return CLASS_NAMES[predicted_index], confidence * 100, pneumonia_probability


def main():
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray.")
    parser.add_argument("--image", required=True, help="Path to the X-ray image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label, confidence, pneumonia_probability = predict_image(image_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Pneumonia probability: {pneumonia_probability:.4f}")


if __name__ == "__main__":
    main()

