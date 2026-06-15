# Medical Image Diagnosis Using Deep Learning for Pneumonia Detection from Chest X-Rays

This project detects pneumonia from chest X-ray images using TensorFlow/Keras transfer learning with EfficientNetB0. It includes model training, evaluation, single-image prediction, Grad-CAM explainability, and a Flask web app.

## Features

- EfficientNetB0 transfer learning
- Kaggle Chest X-Ray Pneumonia dataset support
- CPU-friendly training configuration
- Accuracy, precision, recall, F1 score, confusion matrix, ROC curve
- Single-image prediction with confidence score
- Grad-CAM heatmap visualization
- Flask upload and result pages
- Bootstrap frontend

## Folder Structure

```text
app.py
train.py
evaluate.py
predict.py
gradcam.py
config.py
requirements.txt
models/
dataset/chest_xray/
static/css/
static/uploads/
static/results/
templates/
outputs/
report/
viva/
```

## Dataset Download

Download the dataset from Kaggle:

[Chest X-Ray Images Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

After extracting, place the folders like this:

```text
dataset/chest_xray/train/NORMAL
dataset/chest_xray/train/PNEUMONIA
dataset/chest_xray/val/NORMAL
dataset/chest_xray/val/PNEUMONIA
dataset/chest_xray/test/NORMAL
dataset/chest_xray/test/PNEUMONIA
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

The model will be saved at:

```text
models/pneumonia_efficientnetb0.h5
```

## Evaluate the Model

```bash
python evaluate.py
```

Evaluation files are saved in:

```text
outputs/
```

## Predict a Single Image

```bash
python predict.py --image path/to/xray.jpeg
```

## Run the Flask App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Screenshots

Add screenshots of:

- Upload page
- Result page
- Grad-CAM output
- Confusion matrix
- ROC curve

## Future Scope

- Add multi-class pneumonia type classification
- Train with larger datasets
- Add doctor login and patient records
- Export diagnosis reports as PDF
- Improve explainability using SHAP or LIME

## Medical Disclaimer

This project is for academic demonstration only. It must not be used as a replacement for professional medical diagnosis.

