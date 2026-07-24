"""
Lightweight CNN for ELA-based forgery detection.
~50K parameters — trains in minutes on CPU.
"""

from tensorflow.keras import layers, models


def build_cnn(input_shape=(128, 128, 3)) -> models.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Augmentation (only active during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),   # replaces Flatten+Dense(256) — far fewer params

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
