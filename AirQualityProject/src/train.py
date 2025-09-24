# src/train.py (simplified skeleton)
from src.data_loader import load_df, prepare_scalers_and_tabular, create_sequences, seqs_to_images
from src.models.ann import build_ann
from src.models.cnn1d import build_cnn1d
# ... import other model builders ...
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# config
DATA_PATH = "data/air_quality.csv"
FEATURES = ['PM2.5','PM10','NO2','SO2','CO','O3','temp','humidity','wind']
TARGET = 'AQI'
WINDOW = 24
PRED_H = 1

df = load_df(DATA_PATH)
X_raw, y_raw, scalerX, scalery = prepare_scalers_and_tabular(df, FEATURES, target_col=TARGET)

# build sequences
X_seq, y_seq = create_sequences(X_raw, y_raw, window_size=WINDOW, pred_horizon=PRED_H)
# for encoder-decoder: y_seq shape should be (n, pred_horizon, 1)
y_seq_ed = y_seq.reshape(len(y_seq), PRED_H, 1)

# ANN uses aggregated features per window (mean)
X_ann = X_seq.mean(axis=1)  # shape (n, n_features)

# CNN1D/LSTM use X_seq
# VGG: convert sequences -> images (resize)
X_img = seqs_to_images(X_seq, image_size=(64,64))  # or (224,224) for VGG16

# split train/test (simple)
split = int(0.8*len(X_ann))
# ANN
model_ann = build_ann(X_ann.shape[1])
callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
model_ann.fit(X_ann[:split], y_seq[:split,0], validation_data=(X_ann[split:], y_seq[split:,0]), epochs=100, batch_size=32, callbacks=callbacks)
model_ann.save("models/ann.h5")

# similarly train cnn1d, lstm, vgg9, vgg16, encoder_decoder...
