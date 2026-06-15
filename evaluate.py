"""Evaluate the trained pneumonia detection model."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_PATH,
    OUTPUT_DIR,
    TEST_DIR,
    create_required_directories,
)


def load_test_dataset():
    """Load the test dataset without shuffling."""
    return tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def collect_labels(dataset):
    """Collect true labels from a batched TensorFlow dataset."""
    labels = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy().astype(int).reshape(-1).tolist())
    return np.array(labels)


def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def save_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close()


def main():
    create_required_directories()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train.py first.")

    test_ds = load_test_dataset()
    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = collect_labels(test_ds)
    y_score = model.predict(test_ds).reshape(-1)
    y_pred = (y_score >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    save_confusion_matrix(y_true, y_pred)
    save_roc_curve(y_true, y_score)

    report_text = (
        "Pneumonia Detection Evaluation Report\n"
        "====================================\n\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1 Score : {f1:.4f}\n\n"
        f"{report}\n"
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

