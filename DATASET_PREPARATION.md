# Dataset Preparation

## Dataset

Use the Kaggle Chest X-Ray Pneumonia dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Manual Download Steps

1. Open the Kaggle dataset page.
2. Sign in to Kaggle.
3. Click Download.
4. Extract the ZIP file.
5. Copy the extracted `chest_xray` folder into this project under `dataset/`.

The final structure must be:

```text
dataset/chest_xray/train/NORMAL
dataset/chest_xray/train/PNEUMONIA
dataset/chest_xray/val/NORMAL
dataset/chest_xray/val/PNEUMONIA
dataset/chest_xray/test/NORMAL
dataset/chest_xray/test/PNEUMONIA
```

## Kaggle API Download

Install Kaggle:

```bash
pip install kaggle
```

Download from Kaggle:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

Extract the ZIP and place the dataset in the required folder.

## Preprocessing

The project uses these preprocessing steps:

- Load images from folders using `tf.keras.utils.image_dataset_from_directory`.
- Resize every image to `224 x 224`.
- Convert grayscale X-rays to RGB format.
- Use binary labels: `NORMAL = 0`, `PNEUMONIA = 1`.
- Use TensorFlow prefetching for better performance.

## Data Augmentation

Data augmentation is applied during training:

- Horizontal flip
- Small rotation
- Small zoom
- Contrast adjustment

These transformations help reduce overfitting and improve generalization.

## Laptop-Friendly Notes

- `BATCH_SIZE = 16` is used to reduce memory usage.
- EfficientNetB0 is lightweight compared to larger CNN models.
- The pretrained base model is frozen, so training is faster on CPU.
- You can reduce `EPOCHS` in `config.py` if training is slow.

