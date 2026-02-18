# Colorectal Cancer Survival Prediction MLOps Pipeline

This project implements an end-to-end MLOps pipeline for predicting colorectal cancer survival outcomes. It focuses on integrating modern MLOps tools like **Minikube** for orchestration, **MLflow** for experiment tracking, and **DagsHub** for centralized model management.

---

## 🚀 Key Features
- **Experiment Tracking**: Integrated with MLflow and DagsHub for remote logging.
- **Orchestration**: Kubeflow pipelines running on Minikube for automated workflows.
- **Containerization**: Dockerized components using `uv` for fast dependency management.
- **Model Training**: Gradient Boosting Classifier with automated feature selection.
- **Real-time Prediction**: Flask-based web interface for inference.

---

## 🛠️ Infrastructure and Tools

### 1. MLflow with DagsHub Integration
The project uses **DagsHub** as the remote tracking server for MLflow experiments. This allows multiple team members to visualize metrics, compare runs, and manage models without local infrastructure.

**Setup Instructions:**
To sync your local MLflow runs with DagsHub, set the following environment variables:
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<username>/Colorectal-Cancer-Prediction.mlflow
export MLFLOW_TRACKING_USERNAME=<your_username>
export MLFLOW_TRACKING_PASSWORD=<your_token>
```

**Benefits:**
- **Centralized Dashboard**: View accuracy, precision, and F1-score across different iterations.
- **Model Registry**: Track model versions and transitions from staging to production.
- **Reproducibility**: DagsHub tracks code, data, and models in a single ecosystem.

### 2. Minikube and Kubeflow Pipelines
The MLOps pipeline is orchestrated using **Kubeflow Pipelines (KFP)** on a local **Minikube** cluster, demonstrating the scalability of the training process.

**Pipeline Workflow:**
1.  **Data Processing**: Cleans data, handles encoding, and selects top features using Chi-square (e.g., tumor size, healthcare costs).
2.  **Model Training**: Trains a Gradient Boosting model and logs metrics to the remote MLflow server.

**Deployment Steps:**
1.  **Start Minikube**:
    ```bash
    minikube start --memory 8192 --cpus 4
    ```
2.  **Deploy Kubeflow**:
    Follow the [Kubeflow manifests installation](https://github.com/kubeflow/manifests) or use a lightweight setup like `kfp-server`.
3.  **Build Docker Image in Minikube**:
    ```bash
    eval $(minikube docker-env)
    docker build -t maskedwolf4/colorectal-cancer-prediction:latest .
    ```
4.  **Run Pipeline**:
    ```bash
    python kubeflow_pipeline/mlops_pipeline.py
    ```
    Upload the generated `mlops_pipeline.yaml` to the Kubeflow dashboard.

---

## 🏥 Real-life Applications
The Colorectal Cancer Survival Prediction model has significant implications in modern healthcare:

1.  **Clinical Decision Support**: Assists oncologists in identifying high-risk patients who may require more aggressive monitoring or alternative treatment plans.
2.  **Personalized Medicine**: By analyzing factors like tumor size and pre-existing conditions (e.g., diabetes), treatments can be tailored to individual patient profiles.
3.  **Public Health Planning**: Understanding mortality rates and healthcare costs allows regional health departments to allocate resources more effectively.

---

## 📂 Use Cases
- **Hospital Administration**: Predicting survival rates helps in managing ICU bed availability and specialized nursing staff allocation.
- **Insurance Companies**: Assessing risks to provide more accurate health insurance policies based on demographic and clinical data.
- **Pharmaceutical Research**: Evaluating how different treatment types (e.g., surgery, chemotherapy) impact survival rates across a diverse patient population.

---

## 🧠 Learnings from the Project
- **MLOps Automation**: Transitioning from manual notebook-based training to automated Kubeflow pipelines ensures consistency and scalability.
- **Container Strategy**: Using multi-stage Docker builds with `uv` significantly reduced image size and build times for Kubernetes pods.
- **Kubernetes Management**: Gained hands-on experience in managing resource-intensive tasks on Minikube and handling containerized environments.
- **Remote Experimentation**: Learned the importance of a centralized tracking server (DagsHub) to avoid losing experiment history and facilitate team collaboration.

---

## 📁 Project Structure
```text
.
├── artifacts/            # Data and saved models
├── kubeflow_pipeline/    # KFP pipeline definitions
├── src/                  # Source code for training and processing
├── templates/            # Web interface files
├── main.py               # Flask application
├── Dockerfile            # Container definition
└── mlops_pipeline.yaml   # Compiled pipeline artifact
```

---

## 🚦 Getting Started (Local Development)
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Run data processing:
   ```bash
   uv run python -m src.data_processing
   ```
3. Run model training:
   ```bash
   uv run python -m src.model_training
   ```
4. Start the web app:
   ```bash
   uv run python main.py
   ```
