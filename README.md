# 🚗 PerceptionOps: End-to-End Traffic Object Detection Platform

## Live Demo
```
Streamlit Application: https://your-streamlit-url.up.railway.app
```
![Video Demo](assets/video_tracking.gif)

PerceptionOps is an end-to-end traffic perception platform built using YOLOv8, FastAPI, Streamlit, Docker, Prometheus, and Grafana. What began as a computer vision model training project evolved into a production-style application featuring real-time inference, monitoring, containerized deployment, cloud hosting on Railway, and automated CI/CD workflows through GitHub Actions.

The project was designed to explore the complete machine learning lifecycle—from dataset validation and model training to API serving, observability, cloud deployment, and user-facing application development. Rather than focusing solely on model performance, the emphasis was placed on building a deployable and maintainable perception system that mirrors real-world machine learning infrastructure.

---

## 📹 Demo

### Image Detection

![Image Detection](assets/image_detection.png)

### Video Detection

![Video Detection](assets/video_detection.png)


### Monitoring Dashboard

![Monitoring Dashboard](assets/monitoring_dashboard.png)

### Video Tracking Demo

![Tracking Demo](assets/video_tracking.gif)

---

##  Features

* Traffic object detection using YOLOv8
* Image upload and inference
* Video upload and processing
* Live webcam inference through Streamlit WebRTC
* FastAPI inference service
* Dockerized deployment
* Prometheus metrics collection
* Grafana monitoring dashboard
* Structured JSON inference logging
* Detection analytics and latency tracking_

---

##  Project Architecture

```text
                        ┌─────────────────┐
                        │ Streamlit UI    │
                        │                 │
                        │ • Images        │
                        │ • Videos        │
                        │ • Live Camera   │
                        └────────┬────────┘
                                 │
                                 ▼
                     ┌────────────────────┐
                     │ FastAPI Backend    │
                     │                    │
                     │ YOLOv8 Inference   │
                     └────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ YOLOv8 Model        │
                    └─────────────────────┘

                              │
                              ▼

          ┌────────────────────────────────────┐
          │ Monitoring & Observability         │
          ├────────────────────────────────────┤
          │ Prometheus Metrics                 │
          │ Grafana Dashboard                  │
          │ JSON Inference Logs                │
          └────────────────────────────────────┘
```

---

##  Dataset

The model was trained on a traffic object detection dataset containing vehicles and road scene objects.

### Challenges Encountered

The dataset was not immediately ready for training.

Some issues included:

* Missing annotation files
* Inconsistent class mappings
* Images without corresponding labels
* Class imbalance across object categories

Before training, the dataset was reviewed and validated to ensure that image-label pairs were correctly aligned and class definitions remained consistent throughout the dataset.

This step was important because poor annotations directly affected model quality.

---

##  Model Training

### Model

```text
YOLOv8n
```

The nano version was selected because it provides a good balance between:

* Training speed
* Inference latency
* Model size
* Deployment simplicity

### Training Pipeline

```text
Dataset
   ↓
Annotation Validation
   ↓
YOLOv8 Training
   ↓
Model Evaluation
   ↓
Export Best Model
   ↓
Inference API
```

### Why YOLOv8?

The project focused on building a deployable perception system rather than experimenting with multiple architectures.

YOLOv8 provided:

* Strong baseline performance
* Fast inference speed
* Simple deployment workflow
* Built-in support for image and video prediction

---

##  Backend API

FastAPI was used to expose the trained model through REST endpoints.

### Available Endpoints

#### Image Predictions

```http
POST /predict
```

Returns:

```json
{
  "model_version": "yolov8n_baseline_v1",
  "latency_ms": 18.35,
  "num_detections": 11,
  "class_distribution": {
    "car": 10,
    "bus": 1
  }
}
```

---

#### Annotated Image

```http
POST /predict-image
```

Returns an image with bounding boxes.

---

#### Video Processing

```http
POST /predict-video
```

Returns a processed video with object detections.

---

#### Metrics

```http
GET /metrics
```

Prometheus-compatible metrics endpoint.

---

##  Monitoring and Logging

One objective of this project was to go beyond model training and implement basic production-style observability.

### Prometheus Metrics

Tracked metrics include:

* Total inference requests
* Detection counts
* Average confidence scores
* Inference latency
* Python runtime metrics

Example:

```text
requests_total
detections_total
average_confidence
inference_latency_ms
```

### Grafana Dashboard

The dashboard visualizes:

* Request volume
* Detection volume
* Average confidence
* Latency trends
* System health

![Grafana Dashboard](assets/monitoring_dashboard.png)

### JSON Logging

Each inference request is logged in structured JSON format.

Example:

```json
{
  "timestamp": "2026-06-20T18:24:13",
  "latency_ms": 19.58,
  "detections": 8,
  "class_distribution": {
    "car": 7,
    "truck": 1
  }
}
```

Structured logs make it easier to analyze prediction behavior and debug production issues.

---

##  Frontend Application

The frontend was built with Streamlit.

### Image Detection

Users can:

* Upload images
* Run inference
* View detections
* Review detection statistics

### Video Detection

Users can:

* Upload MP4 videos
* Process videos using YOLOv8
* View the resulting annotated video

### Live Camera

Users can:

* Enable webcam access
* Run real-time detection
* Observe detections frame by frame

> Note: Live camera access works locally and will work on mobile devices once the application is deployed using HTTPS.

---

##  Local Deployment

All services run using Docker Compose.

### Services

```text
FastAPI
Streamlit
Prometheus
Grafana
```

### Run Locally

```bash
docker compose up --build
```

### Access

```text
Streamlit:  http://localhost:8501
FastAPI:    http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
```

---

## Cloud Deployment

PerceptionOps was deployed as a containerized application using Railway, providing public access to both the inference API and user-facing application. The deployment separates the frontend and backend into independent services, enabling scalable inference and simplified maintenance.

Deployment Architecture

```text 
                    ┌─────────────────────┐
                    │ Streamlit Frontend  │
                    │     (Railway)       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  FastAPI Backend    │
                    │     (Railway)       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   YOLOv8 Model      │
                    └─────────────────────┘
```


--- 


## CI/CD Pipeline

The project uses GitHub-based deployment workflows where code changes are pushed to GitHub and automatically deployed through Railway.
```text
Developer
    │
    ▼
GitHub Repository
    │
    ▼
GitHub Actions
    │
    ▼
Railway Deployment
    │
    ▼
Production Application
```

---

##  Challenges and Lessons Learned

This project involved much more than training a model.

Some of the engineering challenges encountered were:

### Dataset Issues

* Missing annotations
* Label inconsistencies
* Dataset validation

### Dockerization

* OpenCV dependency conflicts
* Missing Linux libraries
* Container networking

### Video Processing

* Output file discovery
* Handling YOLO-generated directories
* Video serving through FastAPI

### Monitoring

* Prometheus metric design
* Grafana dashboard creation
* Dashboard persistence using Docker volumes

### Frontend Integration

* Image rendering issues
* Video streaming workflows
* Webcam limitations due to browser security policies

These challenges were valuable because they exposed deployment and operational concerns that are rarely encountered during model training alone.

---

##  Results

The final system supports:

* Image inference
* Video inference
* Live webcam inference
* REST API serving
* Docker deployment
* Monitoring with Prometheus
* Visualization with Grafana
* Structured JSON logging

---

##  Tech Stack

### Machine Learning

* YOLOv8
* PyTorch
* OpenCV

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit
* Streamlit WebRTC

### Monitoring

* Prometheus
* Grafana

### Deployment

* Docker
* Docker Compose

---

##  Author

**Ragul Narayanan Magesh**

GitHub: https://github.com/ragulnarayanan/perception-ops

---

