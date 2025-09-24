from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from src.config import WINDOW, FEATURES, IMG_SIZE_VGG9, IMG_SIZE_VGG16
from src.data_loader import seqs_to_images


app = FastAPI()


class Sample(BaseModel):
    features: Optional[Dict[str, float]] = None
    last_window: Optional[List[Dict[str, float]]] = None


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

    return preds


@app.get("/status")
def status():
    return {
        "loaded_models": sorted(list(app.models.keys())),
        "load_errors": app.load_errors,
        "feature_order": app.feature_order,
    }
