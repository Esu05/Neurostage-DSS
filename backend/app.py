"""
NeuroStage DSS — FastAPI Backend
Based on: Zolfaghari et al., Scientific Reports (2025)
Pipeline: Dual-CNN Feature Extraction → 128-dim Dense → Soft-Voting Ensemble (SVM+RF+KNN)
"""

import io
import numpy as np
import cv2
import joblib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurostage")

# ── Global model registry ─────────────────────────────────────────────────────
registry: Dict = {}

# ── Class config (matches Colab CLASS_NAMES order exactly) ───────────────────
CLASS_NAMES    = ["ND", "VMD", "MID", "MOD"]
DISPLAY_NAMES  = {
    "ND":  "Non-Demented",
    "VMD": "Very Mild Demented",
    "MID": "Mild Demented",
    "MOD": "Moderate Demented",
}

# ── Image preprocessing constants (Cell 2 of Colab) ──────────────────────────
IMG_SIZE = 176   # Resize target: 176×176 px


# ── Lifespan: load models once on startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load fusion_cnn_model.keras and ensemble_classifier.joblib,
    then slice a feature-extractor sub-model up to the 'dense2' layer
    (128-dimensional output), matching Colab Cell 10 export logic.
    """
    logger.info("🔄  Loading model weights …")

    try:
        # 1. Load the full compiled Keras fusion model (dual-CNN + dense head)
        full_model: tf.keras.Model = load_model("fusion_cnn_model.h5")
        logger.info("✅  fusion_cnn_model.keras loaded  —  "
                    f"params: {full_model.count_params():,}")

        # 2. Slice feature extractor: input → dense2 (128-d)
        #    Mirrors: feature_extractor = Model(inputs=full_model.input,
        #                                       outputs=full_model.get_layer('dense2').output)
        feature_extractor: tf.keras.Model = Model(
            inputs=full_model.input,
            outputs=full_model.get_layer("dense_1").output,
        )
        registry["full_model"]         = full_model
        registry["feature_extractor"]  = feature_extractor
        logger.info("✅  feature_extractor slice ready  —  output shape: "
                    f"{feature_extractor.output_shape}")

    except Exception as exc:
        logger.error(f"❌  Failed to load Keras model: {exc}")
        registry["full_model"]        = None
        registry["feature_extractor"] = None

    try:
        # 3. Load the scikit-learn VotingClassifier (SVM + RF + KNN)
        ensemble = joblib.load("ensemble_classifier.joblib")
        registry["ensemble"] = ensemble
        logger.info("✅  ensemble_classifier.joblib loaded")
    except Exception as exc:
        logger.error(f"❌  Failed to load ensemble: {exc}")
        registry["ensemble"] = None

    yield   # ── application runs here ─────────────────────────────────────

    registry.clear()
    logger.info("🛑  Models unloaded.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroStage DSS API",
    description="Alzheimer's disease staging via Hybrid Dual-CNN + Ensemble Classifier",
    version="2026.1",
    lifespan=lifespan,
)

# Allow React dev server at localhost:5173 (Vite default) and :3000 (CRA)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://neurostage-dss.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schema ───────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    predicted_class:    str           # Short label e.g. "ND"
    predicted_label:    str           # Full label e.g. "Non-Demented"
    confidence:         float         # 0–100 %
    probabilities:      Dict[str, float]  # full_label → probability %


# ── /predict endpoint ─────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts a brain MRI image upload and returns Alzheimer's stage probabilities.

    Steps (mirror Colab preprocessing in Cell 2 + inference in Cell 10):
      1. Decode image bytes → grayscale NumPy array (cv2.IMREAD_GRAYSCALE)
      2. Resize to 176×176 with cv2.resize
      3. Normalise [0,1] by dividing by 255.0
      4. Expand dims → shape (1, 176, 176, 1)
      5. Pass through feature_extractor → 128-d vector
      6. VotingClassifier.predict_proba() → soft-voted probabilities
      7. Return JSON with full probability distribution + top prediction
    """
    # ── Guard: models must be loaded ─────────────────────────────────────────
    if registry.get("feature_extractor") is None or registry.get("ensemble") is None:
        raise HTTPException(
            status_code=503,
            detail="Model weights are not loaded. "
                   "Place fusion_cnn_model.keras and ensemble_classifier.joblib "
                   "next to app.py and restart the server.",
        )

    # ── 1. Read bytes & decode to grayscale ───────────────────────────────────
    contents = await file.read()
    np_arr   = np.frombuffer(contents, dtype=np.uint8)
    img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise HTTPException(status_code=422, detail="Cannot decode image. "
                            "Send a valid JPEG/PNG brain MRI slice.")

    # ── 2. Resize to 176×176 (Cell 2: IMG_SIZE = 176) ────────────────────────
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))

    # ── 3. Normalise to [0, 1] ────────────────────────────────────────────────
    img_norm = img_resized.astype(np.float32) / 255.0

    # ── 4. Shape → (1, 176, 176, 1) for the CNN input ────────────────────────
    img_input = np.expand_dims(img_norm, axis=(0, -1))   # batch + channel dims

    # ── 5. Feature extraction through 'dense2' slice (128-d) ─────────────────
    features = registry["feature_extractor"].predict(img_input, verbose=0)
    # features.shape == (1, 128)

    # ── 6. Ensemble soft-vote probabilities ───────────────────────────────────
    proba_matrix = registry["ensemble"].predict_proba(features)
    # proba_matrix.shape == (1, 4)  →  one row, four class probabilities

    proba_arr = proba_matrix[0]   # shape (4,)

    # ── 7. Build response ─────────────────────────────────────────────────────
    top_idx         = int(np.argmax(proba_arr))
    top_class       = CLASS_NAMES[top_idx]
    top_label       = DISPLAY_NAMES[top_class]
    top_confidence  = float(proba_arr[top_idx] * 100)

    # Map full display names → probability percentages
    probabilities = {
        DISPLAY_NAMES[cls]: round(float(proba_arr[i] * 100), 2)
        for i, cls in enumerate(CLASS_NAMES)
    }

    logger.info(f"📊  Prediction: {top_label} ({top_confidence:.1f}%)")

    return PredictResponse(
        predicted_class=top_class,
        predicted_label=top_label,
        confidence=round(top_confidence, 2),
        probabilities=probabilities,
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    ensemble = registry.get("ensemble")
    full_model = registry.get("full_model")
    return {
        "status": "ok",
        "ensemble_classes": ensemble.classes_.tolist() if ensemble else None,
        "layer_names": [l.name for l in full_model.layers] if full_model else None,
        "models_loaded": {
            "cnn": registry.get("feature_extractor") is not None,
            "ensemble": registry.get("ensemble") is not None,
        },
    }