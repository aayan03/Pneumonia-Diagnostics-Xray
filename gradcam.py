"""Grad-CAM utilities for explaining pneumonia predictions."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import IMAGE_SIZE


def _find_last_conv_layer(base_model) -> str:
    """Return the name of the last Conv2D inside the EfficientNet feature extractor."""
    import tensorflow as tf
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def _make_gradcam_heatmap(image_array, model):
    """Return a normalised Grad-CAM heatmap (H x W float32 in [0, 1])."""
    import tensorflow as tf

    base_model = model.get_layer("efficientnetb0")
    last_conv = _find_last_conv_layer(base_model)

    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv).output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(image_array)
        x = model.get_layer("global_average_pooling")(base_outputs)
        x = model.get_layer("dropout")(x, training=False)
        predictions = model.get_layer("prediction")(x)
        loss = predictions[:, 0]

    gradients = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(gradients, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    return heatmap


def get_gradcam_image(image_path, model, alpha=0.45):
    """Return a PIL Image with the Grad-CAM heatmap blended over the original."""
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

    heatmap = _make_gradcam_heatmap(image_array, model)

    original_np = np.array(original)
    heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_np, 1 - alpha, colored, alpha, 0)
    return Image.fromarray(overlay)
