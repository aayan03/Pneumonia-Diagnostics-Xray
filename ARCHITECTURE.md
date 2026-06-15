# Project Architecture

## Project Title

Medical Image Diagnosis Using Deep Learning for Pneumonia Detection from Chest X-Rays

## Folder Structure

```text
Medical image detection system/
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── gradcam.py
├── config.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── DATASET_PREPARATION.md
├── DEPLOYMENT.md
├── render.yaml
├── models/
├── dataset/chest_xray/
├── static/css/style.css
├── static/uploads/
├── static/results/
├── templates/index.html
├── templates/result.html
├── outputs/
├── report/project_report.md
└── viva/viva_questions.md
```

## File Explanation

| File | Purpose |
|---|---|
| `config.py` | Stores paths, image size, batch size, class names, and model path. |
| `train.py` | Loads the dataset, builds EfficientNetB0, trains, and saves the model. |
| `evaluate.py` | Evaluates the saved model using accuracy, precision, recall, F1 score, confusion matrix, and ROC curve. |
| `predict.py` | Runs command-line prediction for one image. |
| `gradcam.py` | Generates Grad-CAM heatmaps and overlays them on X-ray images. |
| `app.py` | Runs the Flask web app for image upload and prediction. |
| `templates/index.html` | Upload page. |
| `templates/result.html` | Prediction result page. |
| `static/css/style.css` | Custom styling. |
| `models/` | Stores the trained model. |
| `outputs/` | Stores evaluation graphs and reports. |

## System Flow

```text
User uploads X-ray
        |
        v
Flask app validates file
        |
        v
Image is resized to 224 x 224 RGB
        |
        v
EfficientNetB0 model predicts probability
        |
        v
Result page displays class and confidence
        |
        v
Grad-CAM highlights important image regions
```

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow/Keras |
| Model | EfficientNetB0 |
| Web Framework | Flask |
| Frontend | HTML, CSS, Bootstrap |
| Evaluation | Scikit-learn, Matplotlib, Seaborn |
| Explainability | Grad-CAM |

