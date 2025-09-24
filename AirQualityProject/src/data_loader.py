# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from skimage.transform import resize

def load_df(path, datetime_col='datetime'):
    df = pd.read_csv(path, parse_dates=[datetime_col])
    df.sort_values(datetime_col, inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def prepare_scalers_and_tabular(df, feature_cols, target_col='AQI', scaler_prefix='models/scaler'):
    X = df[feature_cols].values.astype(float)
    y = df[[target_col]].values.astype(float)

    scalerX = MinMaxScaler()
    scalery = MinMaxScaler()
    Xs = scalerX.fit_transform(X)
    ys = scalery.fit_transform(y)

    joblib.dump(scalerX, scaler_prefix + '_X.gz')
    joblib.dump(scalery, scaler_prefix + '_y.gz')
    return Xs, ys.ravel(), scalerX, scalery

def create_sequences(X, y, window_size=24, pred_horizon=1):
    Xs, ys = [], []
    for i in range(len(X) - window_size - pred_horizon + 1):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size : i+window_size+pred_horizon])
    Xs = np.array(Xs)            # (n_windows, window_size, n_features)
    ys = np.array(ys)            # (n_windows, pred_horizon)  (regression)
    return Xs, ys

def seqs_to_images(X_seq, image_size=(64,64)):
    # X_seq: (n, window_size, n_features) -> transpose to (n_features, window_size)
    n, w, f = X_seq.shape
    imgs = np.zeros((n, image_size[0], image_size[1], 3), dtype=np.float32)
    for i in range(n):
        mat = X_seq[i].T  # shape (features, window)
        img = resize(mat, image_size, anti_aliasing=True)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        imgs[i] = img
    return imgs
