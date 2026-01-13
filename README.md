# 📌 mlops-loan-default

**End-to-End Machine Learning Project: Loan Default Prediction**

This project demonstrates a complete ML workflow for predicting loan defaults, including pipeline orchestration, Dockerization, image registry usage, and CI/CD automation. It is designed for internship-level demonstration purposes, with cloud deployment and advanced monitoring intentionally out of scope.

---

## 🧰 Tech Stack

* **Python 3.10+**
* **ML Library:** scikit-learn (`RandomForestClassifier`)
* **Pipeline Orchestration:** ZenML
* **Experiment Tracking:** MLflow (used but commented out in Docker due to compatibility issues)
* **Containerization:** Docker
* **CI/CD:** GitHub Actions
* **Container Registry:** Docker Hub
* **Dataset:** Kaggle Loan Default Dataset

---

## 🎯 Problem Statement

Predict whether a loan applicant will default based on financial and demographic features. This helps financial institutions reduce risk and make informed lending decisions.

---

## 📂 Project Structure

```
mlops-loan-default/
├── .zen/                 # ZenML pipeline configurations
├── data/raw/             # Original dataset
├── notebooks/            # EDA & experimentation
├── src/                  # Modular pipeline steps and utilities
├── requirements.txt      # Python dependencies
├── run.py                # Script to trigger pipeline locally
├── Dockerfile            # Docker container for training & inference
├── .github/
│   └── workflows/        # CI/CD workflow for Docker build & push
└── README.md             # Project documentation
```

---

## ⚡ Key Features

* **ZenML Pipeline:** Modular steps for preprocessing, training, and evaluation.
* **Dockerized Workflow:** Fully containerized ML project for portability.
* **Docker Hub Registry:** Images pushed to Docker Hub for reproducibility.
* **CI/CD:** Automated Docker build and push using GitHub Actions.
* **Experiment Tracking:** MLflow integration (commented in Docker due to compatibility issues).
* **Reproducibility:** Any user can pull the Docker image and run the pipeline locally.

---

## 🏗 Architecture Diagram

```
Raw Data → ZenML Pipeline → Model Training → Docker Image → Docker Hub → CI/CD Automation
```

---

##  Getting Started

### Clone the repo

```bash
git clone https://github.com/mahendiranaiml/mlops-loan-default.git
cd mlops-loan-default
```

### Build Docker Image (locally)

```bash
docker build -t mahendiranaiml/mlops-training:smote13 .
```

### Run Docker Image

```bash
docker run -it mahendiranaiml/mlops-training:smote13
```

### CI/CD

The project uses GitHub Actions to automate:

1. Docker image build
2. Push to Docker Hub

Pipeline triggers automatically on **push to `main` branch**.

---

## 📊 Model

* **Algorithm:** Random Forest Classifier
* **Target:** Loan Default (binary classification)
* **Features:** Financial & demographic fields (age, income, credit score, etc.)
* **Evaluation Metrics:** Accuracy, Precision, Recall (as shown in notebooks)

---

## 💡 Notes

* MLflow is integrated but **commented out in Docker** due to version compatibility issues.
* Cloud deployment and real-time monitoring are intentionally left out — the focus is on **end-to-end workflow reproducibility**.
* Designed for **internship portfolios**: demonstrates coding, pipeline orchestration, Dockerization, and CI/CD.

---

## 🔗 Docker Hub

Pull the image:

```bash
docker pull mahendiranaiml/mlops-training:smote13
```

---

## 📈 Future Improvements

* Add FastAPI inference service
* Add automated retraining pipeline
* Cloud deployment and monitoring
* MLflow integration fully functional in Docker

---

## 📝 References

* Kaggle Loan Default Dataset: [https://www.kaggle.com/datasets/nikhil1e9/loan-default](https://www.kaggle.com/)
* ZenML Documentation: [https://docs.zenml.io/](https://docs.zenml.io/)
* Docker Documentation: [https://docs.docker.com/](https://docs.docker.com/)
* GitHub Actions: [https://docs.github.com/en/actions](https://docs.github.com/en/actions)

---
