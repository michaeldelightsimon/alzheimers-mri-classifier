# Alzheimer's MRI Stage Classifier

A web app that classifies a brain MRI slice into one of four stages of dementia using a pre-trained **ResNet50** fine-tuned with transfer learning. It shows the predicted stage, the model's confidence, a short plain-language explanation, a Grad-CAM heatmap of the areas that influenced the prediction, and how long the prediction took.

> **Disclaimer:** This is a student project for a university course. It is not a medical device and cannot diagnose anything. Only a doctor can confirm a diagnosis.

## Classes

`NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`

## Results (test set, 960 images)

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| MildDemented | 0.90 | 0.96 | 0.93 | 135 |
| ModerateDemented | 1.00 | 1.00 | 1.00 | 9 |
| NonDemented | 0.91 | 0.88 | 0.89 | 480 |
| VeryMildDemented | 0.83 | 0.86 | 0.85 | 336 |

Overall accuracy: **88%**.

Notes and limitations:
- The ModerateDemented score rests on only 9 test images, so it should not be generalised.
- The model mostly confuses NonDemented and VeryMildDemented, which look very similar on a single slice.
- The data was split randomly, so similar slices from the same patient may appear in both training and test sets, which can make the scores look better than they are.
- The model only works on plain grayscale MRI slices like the training data. Coloured, edited or restored images give unreliable results.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `alz_resnet50.pth` | Trained model weights |
| `train_model.ipynb` | Colab notebook used to train and evaluate the model (add it if you have exported it) |
| `Dockerfile` | Builds the container image |
| `docker-compose.yml` | Runs the container with one command |
| `requirements.txt` | Python packages |

## Run with Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/YOUR-USERNAME/alzheimers-mri-classifier.git
cd alzheimers-mri-classifier
docker build -t alz-classifier:1.0 .
docker run -p 8501:8501 alz-classifier:1.0
```

Open http://localhost:8501 and upload an MRI image.

Or with Compose:

```bash
docker compose up -d --build
```

## Run without Docker

Requires Python 3.10 to 3.12.

```bash
git clone https://github.com/YOUR-USERNAME/alzheimers-mri-classifier.git
cd alzheimers-mri-classifier
python -m venv venv
venv\Scripts\activate        # Windows (on Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Updating the app

1. Edit `app.py`.
2. Build a new version: `docker build -t alz-classifier:1.1 .`
3. Replace the running container:
   ```bash
   docker stop alz
   docker run -d --name alz -p 8501:8501 alz-classifier:1.1
   ```
   With Compose, change the image tag and run `docker compose up -d --build`. Compose only recreates the container that changed.

Plain `docker run` and Compose stop the old container before starting the new one, so there is a short gap. Docker Swarm (`docker service update`) can replace containers one at a time and roll back automatically if the new version fails.

## Security notes

- The container runs as a non-root user.
- The image is based on `python:3.12-slim` to keep the attack surface small.
- No passwords or API keys are stored in the image or repository.

## Training

The model was trained in Google Colab (T4 GPU) using a pre-trained ResNet50 from torchvision, with a class-weighted loss to handle class imbalance (very few ModerateDemented images). The data came from a public four-class Alzheimer's MRI dataset on Kaggle (6,400 images), split 70% train, 15% validation and 15% test.

## Author

Michael Delight Simon, Information Systems Engineering, Near East University.
