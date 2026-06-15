# Deployment Guide

## Local Deployment

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run the Flask app:

```bash
python app.py
```

Open the app:

```text
http://127.0.0.1:5000
```

## Render Deployment

Render free instances have limited memory and storage. For best results, train the model locally and upload the trained `.h5` file with your project repository.

Steps:

1. Push the project to GitHub.
2. Make sure `models/pneumonia_efficientnetb0.h5` exists in the deployed project.
3. Create a new Web Service on Render.
4. Connect your GitHub repository.
5. Use these settings:

```text
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

This repository includes a `render.yaml` file:

```yaml
services:
  - type: web
    name: pneumonia-detection-flask
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    plan: free
```

## Important Deployment Notes

- Do not train the model on Render free plan.
- Train locally, save the model, then deploy.
- Keep uploaded files temporary.
- Add a medical disclaimer in the final presentation.

