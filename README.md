# Network Security Analysis & Threat Detection

A comprehensive, machine learning-driven solution designed to analyze network traffic and identify potential security threats. Leveraging advanced ML techniques, Docker containerization, robust CI/CD pipelines, MLflow model tracking, and MongoDB, this system effectively predicts and mitigates malicious activities in real-time.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Monitoring with MLflow](#monitoring-with-mlflow)
9. [Contributing](#contributing)
10. [License](#license)
11. [Author](#author)

---

## Overview

This project uses supervised machine learning models to detect security threats within network traffic data. Built for scalability and reliability, the solution integrates AWS services (EC2, ECR, S3), Docker, GitHub Actions, MLflow, and MongoDB, ensuring production-grade readiness and ease of deployment.

**Key Highlights:**
- **Data Ingestion & Validation:** Automated collection and validation of data from MongoDB.
- **Data Transformation:** Comprehensive preprocessing, including KNN-based imputation.
- **Model Training:** Employs various classification algorithms (Logistic Regression, Decision Tree, Random Forest, K-Neighbors, AdaBoost, Gradient Boosting) with optimized hyperparameter tuning.
- **Model Tracking:** Uses MLflow for model versioning, performance tracking (F1-score, precision, recall), and artifact management.
- **Deployment:** Containerized using Docker with a fully automated CI/CD pipeline via GitHub Actions, deployed on AWS EC2.

---

## Features

- **End-to-End ML Pipeline** – Automates every stage from ingestion to deployment.
- **Multiple ML Models** – Tests and selects optimal classifiers.
- **CI/CD Automation** – Auto-deploys after successful testing and integration.
- **MLflow Integration** – Logs detailed metrics and artifacts.
- **Docker Containerization** – Ensures consistency across deployments.
- **FastAPI REST API** – User-friendly API endpoints for predictions and model retraining.

---

## Architecture

```plaintext
GitHub Repo
      │
      ▼
GitHub Actions CI/CD
      │
      ▼
Amazon ECR (Docker Images)
      │
      ▼
AWS EC2 (Docker Container)
      │           │
      ▼           ▼
   MongoDB       MLflow Tracking
```

**Components:**
- **GitHub Actions:** CI/CD automation.
- **AWS EC2:** Dockerized deployment environment.
- **MongoDB:** Data storage and retrieval.
- **MLflow:** Experiment tracking and model logging.

---

## Project Structure

```
network-security-system/
├── .github/workflows/main.yml      # CI/CD workflow
├── Dockerfile                      # Docker definition
├── app.py                          # FastAPI app
├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── data_validation.py
│   └── model_trainer.py
├── data/                           # Sample data files
├── networksecurity/
│   ├── entity/                     # Configs and artifacts
│   ├── exception/                  # Custom exceptions
│   ├── logging/                    # Logging setup
│   └── utils/                      # Utility functions
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Getting Started

### Prerequisites

- Python (>= 3.8)
- Docker
- AWS CLI
- MongoDB (local or cloud-hosted)
- MLflow (local or cloud-hosted)

### Installation

**Clone Repository:**
```bash
git clone https://github.com/your-username/network-security-system.git
cd network-security-system
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Setup Environment Variables:** Create `.env` file:
```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
MONGO_DB_URL=your_mongodb_connection_string
DAGSHUB_API_TOKEN=your_dagshub_token
```

---

## Usage

### Local Run (without Docker):
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```
Visit [http://localhost:8080/docs](http://localhost:8080/docs) for API docs.

### Local Run (with Docker):
```bash
docker build -t network-security:latest .
docker run -d -p 8080:8080 --env-file .env network-security:latest
```
Access at [http://localhost:8080](http://localhost:8080).

### Key API Endpoints:
- `GET /train`: Trigger training pipeline.
- `POST /predict`: Upload data CSV for predictions.
- `GET /home`: Dashboard for threat statistics.

---

## CI/CD Pipeline

The workflow defined in `.github/workflows/main.yml` automates:

- **Integration:** Code checks, linting, and unit tests.
- **Delivery:** Docker image build and push to Amazon ECR.
- **Deployment:** Image deployment and container execution on AWS EC2.

---

## Monitoring with MLflow

MLflow tracks:
- Metrics (F1-score, precision, recall).
- Model versions for easy performance comparison.

Ensure `MLflow` URL is set in `.env` or use external services like DagsHub.

---

## Contributing

- Fork repository and use feature branches.
- Submit Pull Requests for review.
- Follow best coding practices and clear commit messaging.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Author

**Hussein Baghdadi**

For collaboration or inquiries, open an issue or reach out via [LinkedIn](https://linkedin.com).

