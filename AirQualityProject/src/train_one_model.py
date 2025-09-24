import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config import (
    DATA_PATH, FEATURES, TARGET, WINDOW, PRED_HORIZON,
    IMG_SIZE_VGG9, IMG_SIZE_VGG16, EPOCHS, BATCH_SIZE, PATIENCE, MODEL_DIR
)
from src.data_loader import (
    load_df, prepare_scalers_and_tabular, create_sequences, seqs_to_images
)
from src.models.ann import build_ann
from src.models.cnn1d import build_cnn1d
from src.models.lstm import build_lstm
from src.models.vgg9 import build_vgg9
from src.models.vgg16_wrapper import build_vgg16
from src.models.encoder_decoder import build_encoder_decoder


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def split_idx(n: int, ratio: float = 0.8) -> int:
    return int(ratio * n)


def train_one_model(model_name: str, cfg: dict | None = None) -> None:
    ensure_dir(MODEL_DIR)

    df = load_df(DATA_PATH)
    X_raw, y_raw, scalerX, scalery = prepare_scalers_and_tabular(
        df, FEATURES, target_col=TARGET, scaler_prefix=os.path.join(MODEL_DIR, 'scaler')
    )

    X_seq, y_seq = create_sequences(X_raw, y_raw, window_size=WINDOW, pred_horizon=PRED_HORIZON)
    y_seq_ed = y_seq.reshape(len(y_seq), PRED_HORIZON, 1)

    callbacks = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, f"{model_name}.h5"), save_best_only=True)
    ]

    if model_name == 'ann':
        X_ann = X_seq.mean(axis=1)
        split = split_idx(len(X_ann))
        model = build_ann(X_ann.shape[1])
        model.fit(
            X_ann[:split], y_seq[:split, 0],
            validation_data=(X_ann[split:], y_seq[split:, 0]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0
        )
        return

    if model_name == 'cnn':
        split = split_idx(len(X_seq))
        model = build_cnn1d(WINDOW, X_seq.shape[-1])
        model.fit(
            X_seq[:split], y_seq[:split, 0],
            validation_data=(X_seq[split:], y_seq[split:, 0]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0
        )
        return

    if model_name == 'lstm':
        split = split_idx(len(X_seq))
        model = build_lstm(WINDOW, X_seq.shape[-1])
        model.fit(
            X_seq[:split], y_seq[:split, 0],
            validation_data=(X_seq[split:], y_seq[split:, 0]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0
        )
        return

    if model_name == 'vgg9':
        X_img = seqs_to_images(X_seq, image_size=IMG_SIZE_VGG9)
        split = split_idx(len(X_img))
        model = build_vgg9(img_shape=(*IMG_SIZE_VGG9, 3))
        model.fit(
            X_img[:split], y_seq[:split, 0],
            validation_data=(X_img[split:], y_seq[split:, 0]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0
        )
        return

    if model_name == 'vgg16':
        X_img = seqs_to_images(X_seq, image_size=IMG_SIZE_VGG16)
        split = split_idx(len(X_img))
        model = build_vgg16(img_shape=(*IMG_SIZE_VGG16, 3))
        model.fit(
            X_img[:split], y_seq[:split, 0],
            validation_data=(X_img[split:], y_seq[split:, 0]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0
        )
        return

    if model_name in ('encoder_decoder', 'encdec'):
        split = split_idx(len(X_seq))
        model = build_encoder_decoder(WINDOW, X_seq.shape[-1], pred_horizon=PRED_HORIZON)
        model.fit(
            X_seq[:split], y_seq_ed[:split],
            validation_data=(X_seq[split:], y_seq_ed[split:]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0
        )
        return

    raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else 'ann'
    train_one_model(name)


