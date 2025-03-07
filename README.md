# Network Security Analysis & Threat Detection

A comprehensive machine learning-driven project designed to analyze network traffic and detect potential security threats. The solution employs advanced classification models trained on structured CSV datasets to accurately identify and predict network security anomalies.

## Overview

The project integrates machine learning techniques with cloud infrastructure, leveraging AWS services including EC2, ECR, and S3. It incorporates Docker containerization, continuous integration and continuous deployment (CI/CD) via GitHub Actions, MLflow for robust model tracking, and MongoDB for secure and efficient data management.

## Features

- **Machine Learning Models:** Implements multiple classifiers including Logistic Regression, Decision Tree, Random Forest, K-Neighbors, AdaBoost, and Gradient Boosting.
- **Model Evaluation:** Utilizes cross-validation and hyperparameter tuning for optimal performance.
- **Model Tracking:** Integrated MLflow for tracking experiments, metrics (precision, recall, F1-score), and model versions.
- **Containerization:** Dockerized application for consistent deployment across environments.
- **CI/CD Pipeline:** Automated build, push, and deployment process via GitHub Actions.
- **Cloud Deployment:** AWS EC2 for deployment, ECR for Docker image management, and S3 for model storage.
- **Database Integration:** MongoDB used with data ingestion.
- **API Integration:** FastAPI is used to serve predictions via RESTful APIs and also train.

## Dataset Structure

The dataset is a structured CSV file (`phishingData.csv`) with multiple feature columns representing network traffic attributes, with the last column used as the target variable indicating the presence of a security threat.

```
feature_1, feature_2, ..., feature_n, target
```

## Project Architecture

```
NetworkSecurity
│
├── entity
│   ├── artifact_entity.py
│   └── config_entity.py
│
├── exception
│   └── exception.py
│
├── logging
│   └── logging.py
│
├── utils
│   ├── main_utils
│   │   └── utils.py
│   └── ml_utils
│       ├── model
│       │   └── estimator.py
│       └── metric
│           └── classification_metric.py
│
├── components
│   ├── model_trainer.py
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   └── data_validation.py
│
├── data
│   └── phishingData.csv
│
├── Dockerfile
├── docker-compose.yml
└── .github
    └── workflows
        └── main.yml
```

## Getting Started

### Prerequisites

- Docker
- AWS CLI
- Python >= 3.8
- MongoDB

### Installation

Clone the repository:

```sh
git clone https://github.com/your-username/network-security-system.git
cd network-security-system
```

### Environment Variables

Create a `.env` file with:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
DAGSHUB_API_TOKEN=your_dagshub_token
MONGO_DB_URL=your_mongodb_url
```

### Docker Deployment

Build and run locally:

```sh
docker build -t network-security:latest .
docker run -d -p 8080:8080 --env-file .env network-security:latest
```

### CI/CD

- The repository includes a GitHub Actions workflow (`main.yml`) to automate testing, building Docker images, pushing to ECR, and deploying on AWS EC2.

### Usage

The FastAPI service will be accessible at:

```
http://localhost:8080/docs
```

## API Endpoints

- `/predict`: POST request for classifying network traffic.

## Monitoring

- **MLflow:** Track experiments and model performance at [MLflow Dashboard](https://dagshub.com/hussein.baghdadi01/network-security-system/.mlflow).

## Contributing

Contributions are welcome. Please submit pull requests to the `main` branch.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Developed by Hussein Baghdadi**