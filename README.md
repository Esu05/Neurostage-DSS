# 🧠 NeuroStage DSS

**Alzheimer's Disease Staging — Clinical Decision Support System**  
Based on: *Zolfaghari et al., Scientific Reports (2025) 15:25342*  
Model: Hybrid Dual-CNN (2×2 + 4×4 kernels) + Soft-Voting Ensemble (SVM · RF · KNN)

---

## Architecture

```
Browser (React + Vite)
    │  POST /predict  (multipart/form-data)
    ▼
FastAPI Backend (app.py)
    │
    ├── cv2.imdecode → GRAYSCALE
    ├── cv2.resize(176, 176)
    ├── img / 255.0   → [0, 1]
    ├── shape (1, 176, 176, 1)
    │
    ├── Keras: fusion_cnn_model.keras
    │   └── feature_extractor → dense2 (128-d)
    │
    └── scikit-learn: ensemble_classifier.joblib
        └── VotingClassifier (SVM + RF + KNN)
            └── .predict_proba() → [ND, VMD, MID, MOD]
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Trained model files from Colab (see below)

---

## Step 1 — Export Model Files from Google Colab

After running the Colab notebook successfully, run **Cell 13** to save outputs.  
Then download these two files to your local machine:

| File | Description |
|------|-------------|
| `fusion_cnn_model.keras` | Full Keras dual-CNN model (dual-stream + dense head) |
| `ensemble_classifier.joblib` | Scikit-learn VotingClassifier (SVM + RF + KNN) |

Place both files inside the `backend/` directory:
```
neurostage/
└── backend/
    ├── app.py
    ├── requirements.txt
    ├── fusion_cnn_model.keras        ← place here
    └── ensemble_classifier.joblib   ← place here
```

> **Note:** The Colab Cell 13 saves `fusion_cnn_model.keras` but the ensemble  
> must be saved separately. Add this line after the ensemble is trained (Cell 10):
> ```python
> import joblib
> joblib.dump(ensemble_clf, '/content/drive/MyDrive/alzheimer_results/ensemble_classifier.joblib')
> ```
> Where `ensemble_clf` is your fitted `VotingClassifier` instance.

---

## Step 2 — Start the Backend

```bash
cd neurostage/backend

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch FastAPI
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Verify it's running:
```
http://localhost:8000/health
http://localhost:8000/docs     ← Swagger UI
```

---

## Step 3 — Start the Frontend

```bash
cd neurostage/frontend

npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Usage

1. Drop or click to upload a brain MRI slice (JPG or PNG).
2. Click **▶ RUN DIAGNOSTIC SCAN**.
3. The system returns:
   - **Predicted Stage** with a colour-coded verdict badge
   - **Confidence score**
   - **Full probability distribution** across all 4 classes with animated bars

| Class | Badge | Colour |
|-------|-------|--------|
| Non-Demented | NEGATIVE | 🟢 Green |
| Very Mild Demented | STAGE I | 🟡 Amber |
| Mild Demented | STAGE II | 🟠 Orange |
| Moderate Demented | STAGE III | 🔴 Red |

---

## API Reference

### `POST /predict`

**Request:** `multipart/form-data` with field `file` (image/jpeg or image/png)

**Response:**
```json
{
  "predicted_class": "ND",
  "predicted_label": "Non-Demented",
  "confidence": 97.42,
  "probabilities": {
    "Non-Demented": 97.42,
    "Very Mild Demented": 1.83,
    "Mild Demented": 0.52,
    "Moderate Demented": 0.23
  }
}
```

### `GET /health`
```json
{
  "status": "ok",
  "models_loaded": {
    "cnn": true,
    "ensemble": true
  }
}
```

---

## Project Structure

```
neurostage/
├── backend/
│   ├── app.py                     # FastAPI application
│   ├── requirements.txt
│   ├── fusion_cnn_model.keras     # (you provide)
│   └── ensemble_classifier.joblib # (you provide)
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx                # Full dashboard UI
        └── index.css
```

---

## Paper Reference

> Zolfaghari, S., Joudaki, A. & Sarbaz, Y.  
> *A hybrid learning approach for MRI-based detection of Alzheimer's disease stages using dual CNNs and ensemble classifier.*  
> Scientific Reports **15**, 25342 (2025).
