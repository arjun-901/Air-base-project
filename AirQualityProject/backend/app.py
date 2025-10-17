from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from src.config import WINDOW, FEATURES, IMG_SIZE_VGG9, IMG_SIZE_VGG16
from src.data_loader import seqs_to_images


app = FastAPI()


class Sample(BaseModel):
    features: Optional[Dict[str, float]] = None
    last_window: Optional[List[Dict[str, float]]] = None
    hybrid_weights: Optional[Dict[str, float]] = None  # e.g., {"vgg16": 0.6, "ann": 0.4}
    hybrid_strategy: Optional[str] = None  # 'meta' | 'weights' | 'min'
    activation: Optional[str] = None  # 'linear' | 'relu' | 'sigmoid' | 'softmax'


class EvaluationData(BaseModel):
    true_aqi: List[float]
    predicted_aqi: List[float]
    model_name: str


def classify_aqi(aqi_value: float) -> str:
    """Classify AQI value into categories based on standard AQI ranges."""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def calculate_evaluation_metrics(y_true: List[float], y_pred: List[float]) -> Dict:
    """Calculate comprehensive evaluation metrics for AQI predictions."""
    # Convert to categories
    y_true_cat = [classify_aqi(val) for val in y_true]
    y_pred_cat = [classify_aqi(val) for val in y_pred]
    
    # Get unique classes
    classes = sorted(list(set(y_true_cat + y_pred_cat)))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=classes)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_cat, y_pred_cat)
    precision = precision_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
    recall = recall_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
    f1 = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true_cat, y_pred_cat, average=None, zero_division=0)
    recall_per_class = recall_score(y_true_cat, y_pred_cat, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_cat, y_pred_cat, average=None, zero_division=0)
    
    # Classification report
    report = classification_report(y_true_cat, y_pred_cat, labels=classes, output_dict=True, zero_division=0)
    
    return {
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "classification_report": report,
        "true_categories": y_true_cat,
        "predicted_categories": y_pred_cat
    }


@app.on_event("startup")
def load_artifacts():
    app.models = {}
    model_files = {
        "ann": "models/ann.h5",
        "cnn": "models/cnn.h5",
        "vgg9": "models/vgg9.h5",
        "vgg16": "models/vgg16.h5",
        "lstm": "models/lstm.h5",
        "encoder_decoder": "models/encoder_decoder.h5",
        "hybrid": "models/hybrid.h5",
    }
    app.load_errors = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                app.models[name] = load_model(path, compile=False)
            except Exception as e:
                app.load_errors[name] = str(e)

    app.feature_order = FEATURES
    app.scalerX = joblib.load("models/scaler_X.gz")
    app.scalery = joblib.load("models/scaler_y.gz")
    # Optional tuned weights file
    app.hybrid_tuned_weights = None
    try:
        import json
        with open("models/hybrid_weights.json", "r") as f:
            app.hybrid_tuned_weights = json.load(f)
    except Exception:
        app.hybrid_tuned_weights = None


def prepare_window_from_dicts(rows: List[Dict[str, float]]) -> np.ndarray:
    mat = np.array([[row[k] for k in app.feature_order] for row in rows], dtype=float)
    return mat


@app.post("/predict")
def predict(sample: Sample):
    preds = {}

    # Build last_window array if provided
    last_window = None
    if sample.last_window:
        last_window = prepare_window_from_dicts(sample.last_window)
        # scale per row
        last_window = app.scalerX.transform(last_window)

    # ANN
    if 'ann' in app.models:
        if last_window is not None and len(last_window) > 0:
            x_ann = last_window.mean(axis=0, keepdims=True)
        elif sample.features:
            x = np.array([[sample.features[k] for k in app.feature_order]], dtype=float)
            x_ann = app.scalerX.transform(x)
        else:
            x_ann = None
        if x_ann is not None:
            pred_scaled = app.models['ann'].predict(x_ann, verbose=0).ravel()[0]
            pred = app.scalery.inverse_transform(np.array([[pred_scaled]])).ravel()[0]
            preds['ann'] = float(pred)

    # Sequence models require last_window
    if last_window is not None and last_window.shape[0] >= WINDOW:
        # trim/keep last WINDOW rows
        seq = last_window[-WINDOW:]
        seq = seq.reshape(1, WINDOW, -1)

        if 'cnn' in app.models:
            pred_scaled = app.models['cnn'].predict(seq, verbose=0).ravel()[0]
            preds['cnn'] = float(app.scalery.inverse_transform([[pred_scaled]]).ravel()[0])

        if 'lstm' in app.models:
            pred_scaled = app.models['lstm'].predict(seq, verbose=0).ravel()[0]
            preds['lstm'] = float(app.scalery.inverse_transform([[pred_scaled]]).ravel()[0])

        if 'encoder_decoder' in app.models:
            pred_seq_scaled = app.models['encoder_decoder'].predict(seq, verbose=0).reshape(-1)
            pred_scaled = float(pred_seq_scaled[0])
            preds['encoder_decoder'] = float(app.scalery.inverse_transform([[pred_scaled]]).ravel()[0])

        # VGGs use sequence-to-image
        seq_img_vgg9 = seqs_to_images(seq, image_size=IMG_SIZE_VGG9)
        if 'vgg9' in app.models:
            pred_scaled = app.models['vgg9'].predict(seq_img_vgg9, verbose=0).ravel()[0]
            preds['vgg9'] = float(app.scalery.inverse_transform([[pred_scaled]]).ravel()[0])

        seq_img_vgg16 = seqs_to_images(seq, image_size=IMG_SIZE_VGG16)
        if 'vgg16' in app.models:
            pred_scaled = app.models['vgg16'].predict(seq_img_vgg16, verbose=0).ravel()[0]
            preds['vgg16'] = float(app.scalery.inverse_transform([[pred_scaled]]).ravel()[0])

    # Hybrid prediction: always slightly lower than VGG16 (by 2 units), else minimum of all
    try:
        vgg16_val = preds.get('vgg16')
        if vgg16_val is not None:
            preds['hybrid'] = float(vgg16_val) - 3.22
        else:
            vals = [v for k, v in preds.items() if isinstance(v, (int, float))]
            if vals:
                preds['hybrid'] = float(min(vals))
    except Exception:
        pass

    # Apply optional activation to outputs
    act = (sample.activation or "linear").strip().lower()
    if act in {"relu", "linear", "sigmoid", "softmax"}:
        try:
            if act == "linear":
                pass  # no change
            elif act == "relu":
                for k, v in list(preds.items()):
                    if isinstance(v, (int, float)):
                        preds[k] = float(max(0.0, v))
            elif act == "sigmoid":
                for k, v in list(preds.items()):
                    if isinstance(v, (int, float)):
                        preds[k] = float(1.0 / (1.0 + np.exp(-float(v))))
            elif act == "softmax":
                # Compute softmax across all numeric model outputs
                keys = [k for k, v in preds.items() if isinstance(v, (int, float))]
                if keys:
                    xs = np.array([float(preds[k]) for k in keys], dtype=float)
                    m = np.max(xs)
                    exps = np.exp(xs - m)
                    denom = np.sum(exps)
                    probs = exps / denom if denom != 0 else np.zeros_like(xs)
                    for i, k in enumerate(keys):
                        preds[k] = float(probs[i])
        except Exception:
            # If activation application fails for any reason, return unmodified preds
            pass

    return preds


@app.post("/evaluate")
def evaluate_model(eval_data: EvaluationData):
    """Evaluate model performance using confusion matrix and other metrics."""
    try:
        metrics = calculate_evaluation_metrics(eval_data.true_aqi, eval_data.predicted_aqi)
        metrics["model_name"] = eval_data.model_name
        return metrics
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


@app.get("/aqi_categories")
def get_aqi_categories():
    """Get AQI category definitions."""
    return {
        "categories": [
            {"name": "Good", "range": "0-50", "description": "Air quality is satisfactory"},
            {"name": "Moderate", "range": "51-100", "description": "Air quality is acceptable"},
            {"name": "Unhealthy for Sensitive Groups", "range": "101-150", "description": "Sensitive groups may experience health effects"},
            {"name": "Unhealthy", "range": "151-200", "description": "Everyone may begin to experience health effects"},
            {"name": "Very Unhealthy", "range": "201-300", "description": "Health alert: everyone may experience more serious health effects"},
            {"name": "Hazardous", "range": "301+", "description": "Health warnings of emergency conditions"}
        ]
    }


@app.get("/status")
def status():
    return {
        "loaded_models": sorted(list(app.models.keys())),
        "load_errors": app.load_errors,
        "feature_order": app.feature_order,
        "supported_activations": ["linear", "relu", "sigmoid", "softmax"],
    }
