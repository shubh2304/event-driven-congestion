# 🐳 ASTRAM Docker Deployment Guide

This guide describes how to containerize, build, run, and manage the **ASTRAM Event-Driven Congestion Forecasting System** using Docker and Docker Compose.

---

## 🛠️ Prerequisites
Before starting, ensure you have the following installed on your system:
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine (Linux).
* [Docker Compose](https://docs.docker.com/compose/install/) (included in Docker Desktop by default).

---

## 🏗️ Docker Configurations Included

1. **`Dockerfile.backend`** (Root):
   * Sets up a Python 3.11 environment.
   * Installs standard OS requirements, including OpenMP compiler libraries (`libgomp1`), which are required by the LightGBM models on Linux.
   * Copies Python source code, model pickle files, and reference CSVs.
   * Exposes FastAPI on port `8000` running under `uvicorn` with `4` workers.
2. **`frontend/Dockerfile`** (Frontend):
   * Multi-stage Node 18 build for caching compilation layers.
   * Compiles the Next.js production files (`npm run build`).
   * Copies only required static/runtime resources, keeping image size minimal.
   * Exposes Next.js client on port `3000`.
3. **`docker-compose.yml`** (Root):
   * Configures and links both `backend` and `frontend` containers.
   * Mounts the host's local `./logs` folder to `/app/logs` inside the backend container to ensure that prediction logs are persistent.

---

## 🚀 Step-by-Step Deployment

### Step 1: Pre-build Model Check
Ensure your model directories contain the pre-trained pickling artifacts. At startup, the API server imports these classifiers.
* Check that files like `closure_classifier.pkl`, `priority_classifier.pkl`, and `duration_regressor.pkl` exist in the `models/` directory.

### Step 2: Build & Start Containers
From the root directory of the project, run:
```bash
docker compose up --build -d
```
* **`--build`**: Tells compose to build the backend and frontend Dockerfiles using the local contexts.
* **`-d`**: Runs the containers in detached mode (background).

### Step 3: Verify Running Services
Run the following command to check if both containers are healthy:
```bash
docker compose ps
```
You should see:
* `astram-backend` running on `0.0.0.0:8000->8000/tcp`
* `astram-frontend` running on `0.0.0.0:3000->3000/tcp`

---

## 🌐 Interacting with the Interfaces

| Service Interface | Local URL | Description |
|---|---|---|
| **Next.js Web Client** | [http://localhost:3000](http://localhost:3000) | Main dashboard UI for simulating predictions, checking analytics, and conversing with the chatbot |
| **FastAPI REST Gateway** | [http://localhost:8000](http://localhost:8000) | Rest API endpoints |
| **Interactive Swagger Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Backend REST endpoints list (FastAPI interactive sandbox) |

---

## 🌍 Production Configuration (Custom Domains/Hosts)

### 1. Modifying Frontend API Target
Next.js bakes environment variables into the built bundle during the Docker build stage. To point the frontend to a custom production backend URL, pass the `NEXT_PUBLIC_API_URL` build argument:
```bash
docker compose build --build-arg NEXT_PUBLIC_API_URL=https://api.yourdomain.com frontend
```
Or define it in `docker-compose.yml` under the `build` configuration args:
```yaml
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        - NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

### 2. Log Persistence
The container mounts the `./logs` host directory. Prediction logs containing inputs, probabilities, and recommendations are continuously appended to:
* `./logs/prediction_log.csv` (on your host machine).
This file can be safely backed up or ingested by other pipeline monitors without stopping the containers.

---

## 🧹 Maintenance and Troubleshooting

### View Container Logs
To inspect logs or debug startup failures (e.g., model loading issues):
```bash
# View all logs
docker compose logs -f

# View backend logs only
docker compose logs -f backend

# View frontend logs only
docker compose logs -f frontend
```

### Stopping the Services
To stop and remove containers while preserving your persistent logs:
```bash
docker compose down
```

### Clean Rebuilds
If you modify source code, package configurations, or python libraries, perform a clean rebuild using:
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Memory Requirements
* **RAM Profile**: The backend container requires at least **1.5 GB to 2 GB** of RAM because it loads several large serialization matrices (`.pkl`) and ML models into memory. Ensure Docker Desktop has sufficient memory allocated (default is usually 2GB+ which is fine).
