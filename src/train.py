"""
Trains the CNN on the ELA dataset built by prepare_dataset.py and saves
the trained model + a confusion matrix / classification report.

Set SMOKE_TEST = True (must match prepare_dataset.py) to load the small
smoke dataset (X_smoke.npy / y_smoke.npy) and use lighter training params.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from model import build_cnn

# ---------------------------------------------------------------------------
# Smoke-test controls  <-- keep in sync with prepare_dataset.py
# ---------------------------------------------------------------------------
SMOKE_TEST = True   # True = sample run, False = full dataset
# ---------------------------------------------------------------------------

# Limit TensorFlow to half the logical CPU cores so the machine stays usable.
# Increase or remove this cap for a full training run.
_cpu_cores = os.cpu_count() or 2
# Use all cores for the full run — change to _cpu_cores // 2 if machine feels sluggish
_tf_threads = max(1, _cpu_cores)
tf.config.threading.set_intra_op_parallelism_threads(_tf_threads)
tf.config.threading.set_inter_op_parallelism_threads(_tf_threads)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Full-run params
EPOCHS     = 30
BATCH_SIZE = 32

# Smoke-run overrides
SMOKE_EPOCHS     = 15
SMOKE_BATCH_SIZE = 32
SMOKE_SAMPLES    = 500   # images per class — enough to learn, fast enough on CPU


def load_data():
    # Always load from the full X.npy / y.npy
    x_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"{x_path} not found.\n"
            "Run prepare_dataset.py first to build the dataset."
        )
    X = np.load(x_path)
    y = np.load(y_path)

    if SMOKE_TEST:
        # Slice SMOKE_SAMPLES per class in-memory — no reprocessing needed
        rng = np.random.default_rng(42)
        idx_auth = np.where(y == 0)[0]
        idx_forg = np.where(y == 1)[0]
        idx_auth = rng.choice(idx_auth, size=min(SMOKE_SAMPLES, len(idx_auth)), replace=False)
        idx_forg = rng.choice(idx_forg, size=min(SMOKE_SAMPLES, len(idx_forg)), replace=False)
        idx = np.concatenate([idx_auth, idx_forg])
        rng.shuffle(idx)
        X, y = X[idx], y[idx]
        print(f"Smoke slice: {len(y)} samples "
              f"(authentic={( y==0).sum()}, forged={(y==1).sum()})")

    return X, y


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    active_epochs     = SMOKE_EPOCHS     if SMOKE_TEST else EPOCHS
    active_batch_size = SMOKE_BATCH_SIZE if SMOKE_TEST else BATCH_SIZE
    mode_label        = f"SMOKE-TEST (epochs={active_epochs}, batch={active_batch_size}, threads={_tf_threads})" \
                        if SMOKE_TEST else f"FULL RUN (epochs={active_epochs}, batch={active_batch_size})"
    print(f"\n=== train  [{mode_label}] ===\n")

    print("Loading dataset...")
    X, y = load_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    model = build_cnn(input_shape=X.shape[1:])
    model.summary()

    suffix = "_smoke" if SMOKE_TEST else ""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f"best_model{suffix}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=active_epochs,
        batch_size=active_batch_size,
        callbacks=callbacks,
    )

    # Final evaluation on held-out test set
    print("\nEvaluating on test set...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print(classification_report(y_test, y_pred, target_names=["authentic", "forged"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save final model
    model.save(os.path.join(MODEL_DIR, f"final_model{suffix}.keras"))
    print(f"\nModel saved to {MODEL_DIR}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"training_curves{suffix}.png"))
    print("Training curves saved.")


if __name__ == "__main__":
    main()
