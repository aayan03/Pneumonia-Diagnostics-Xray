"""Train an EfficientNetB0 model for pneumonia detection.

Expected dataset layout:
dataset/chest_xray/train/NORMAL
dataset/chest_xray/train/PNEUMONIA
dataset/chest_xray/val/NORMAL
dataset/chest_xray/val/PNEUMONIA
dataset/chest_xray/test/NORMAL
dataset/chest_xray/test/PNEUMONIA
"""

import tensorflow as tf

# TF 2.16+ moved keras to a standalone package; this keeps both versions working.
try:
    from tensorflow import keras
except ImportError:
    import keras  # type: ignore

from keras import layers, models
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EPOCHS,
    IMAGE_SIZE,
    INPUT_SHAPE,
    LEARNING_RATE,
    MODEL_PATH,
    TRAIN_DIR,
    VAL_DIR,
    create_required_directories,
)


def load_datasets():
    """Load train and validation datasets from folders."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds


def build_model():
    """Create a CPU-friendly transfer learning model using EfficientNetB0."""
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.08),
        ],
        name="data_augmentation",
    )

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=INPUT_SHAPE, name="input_image")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs, outputs, name="pneumonia_efficientnetb0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def main():
    create_required_directories()
    train_ds, val_ds = load_datasets()
    model = build_model()

    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
