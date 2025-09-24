import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import load_model
import joblib

from src.config import DATA_PATH, FEATURES, TARGET, WINDOW, PRED_HORIZON, MODEL_DIR, IMG_SIZE_VGG9, IMG_SIZE_VGG16
from src.data_loader import load_df, prepare_scalers_and_tabular, create_sequences, seqs_to_images


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def evaluate_all():
    df = load_df(DATA_PATH)
    X_raw, y_raw, scalerX, scalery = prepare_scalers_and_tabular(
        df, FEATURES, target_col=TARGET, scaler_prefix=os.path.join(MODEL_DIR, 'scaler')
    )

    X_seq, y_seq = create_sequences(X_raw, y_raw, window_size=WINDOW, pred_horizon=PRED_HORIZON)
    y_true = y_seq[:, 0]

    results = {}

    # ANN
    ann_path = os.path.join(MODEL_DIR, 'ann.h5')
    if os.path.exists(ann_path):
        X_ann = X_seq.mean(axis=1)
        ann = load_model(ann_path)
        pred_scaled = ann.predict(X_ann, verbose=0).ravel()
        y_pred = scalery.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        y_true_inv = scalery.inverse_transform(y_true.reshape(-1, 1)).ravel()
        results['ann'] = {
            'MAE': float(mean_absolute_error(y_true_inv, y_pred)),
            'RMSE': float(rmse(y_true_inv, y_pred)),
            'R2': float(r2_score(y_true_inv, y_pred)),
        }

    # CNN
    cnn_path = os.path.join(MODEL_DIR, 'cnn.h5')
    if os.path.exists(cnn_path):
        cnn = load_model(cnn_path)
        pred_scaled = cnn.predict(X_seq, verbose=0).ravel()
        y_pred = scalery.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        y_true_inv = scalery.inverse_transform(y_true.reshape(-1, 1)).ravel()
        results['cnn'] = {
            'MAE': float(mean_absolute_error(y_true_inv, y_pred)),
            'RMSE': float(rmse(y_true_inv, y_pred)),
            'R2': float(r2_score(y_true_inv, y_pred)),
        }

    # LSTM
    lstm_path = os.path.join(MODEL_DIR, 'lstm.h5')
    if os.path.exists(lstm_path):
        lstm = load_model(lstm_path)
        pred_scaled = lstm.predict(X_seq, verbose=0).ravel()
        y_pred = scalery.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        y_true_inv = scalery.inverse_transform(y_true.reshape(-1, 1)).ravel()
        results['lstm'] = {
            'MAE': float(mean_absolute_error(y_true_inv, y_pred)),
            'RMSE': float(rmse(y_true_inv, y_pred)),
            'R2': float(r2_score(y_true_inv, y_pred)),
        }

    # VGG9
    vgg9_path = os.path.join(MODEL_DIR, 'vgg9.h5')
    if os.path.exists(vgg9_path):
        vgg9 = load_model(vgg9_path)
        X_img = seqs_to_images(X_seq, image_size=IMG_SIZE_VGG9)
        pred_scaled = vgg9.predict(X_img, verbose=0).ravel()
        y_pred = scalery.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        y_true_inv = scalery.inverse_transform(y_true.reshape(-1, 1)).ravel()
        results['vgg9'] = {
            'MAE': float(mean_absolute_error(y_true_inv, y_pred)),
            'RMSE': float(rmse(y_true_inv, y_pred)),
            'R2': float(r2_score(y_true_inv, y_pred)),
        }

    # VGG16
    vgg16_path = os.path.join(MODEL_DIR, 'vgg16.h5')
    if os.path.exists(vgg16_path):
        vgg16 = load_model(vgg16_path)
        X_img = seqs_to_images(X_seq, image_size=IMG_SIZE_VGG16)
        pred_scaled = vgg16.predict(X_img, verbose=0).ravel()
        y_pred = scalery.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        y_true_inv = scalery.inverse_transform(y_true.reshape(-1, 1)).ravel()
        results['vgg16'] = {
            'MAE': float(mean_absolute_error(y_true_inv, y_pred)),
            'RMSE': float(rmse(y_true_inv, y_pred)),
            'R2': float(r2_score(y_true_inv, y_pred)),
        }

    # Encoder-Decoder
    encdec_path = os.path.join(MODEL_DIR, 'encoder_decoder.h5')
    if os.path.exists(encdec_path):
        encdec = load_model(encdec_path)
        y_seq_ed = y_seq.reshape(len(y_seq), PRED_HORIZON, 1)
        pred_scaled_seq = encdec.predict(X_seq, verbose=0).reshape(-1, PRED_HORIZON)
        pred_scaled = pred_scaled_seq[:, 0]
        y_pred = scalery.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        y_true_inv = scalery.inverse_transform(y_true.reshape(-1, 1)).ravel()
        results['encoder_decoder'] = {
            'MAE': float(mean_absolute_error(y_true_inv, y_pred)),
            'RMSE': float(rmse(y_true_inv, y_pred)),
            'R2': float(r2_score(y_true_inv, y_pred)),
        }

    return results


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate_all(), indent=2))
