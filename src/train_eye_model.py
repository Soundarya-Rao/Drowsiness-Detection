"""
Eye State Classifier Training Script (Open vs Closed Eyes)
Drowsiness Detection System for Indian Truck Drivers

Uses Transfer Learning with MobileNetV2 pretrained on ImageNet
and saves the trained model to models/eye_model.keras.
"""

import os
import sys
import argparse
from pathlib import Path

# Suppress verbose TensorFlow C++ logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "eyes"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "eye_model.keras"
PLOT_SAVE_PATH = MODELS_DIR / "eye_model_training_curve.png"


def verify_dataset_structure(data_dir: Path):
    """Verify that open and closed directories exist and contain images."""
    open_dir = data_dir / "open"
    closed_dir = data_dir / "closed"

    if not open_dir.exists() or not closed_dir.exists():
        print(f"[!] ERROR: Expected subdirectories '{open_dir}' and '{closed_dir}' do not exist.")
        print("[!] Please run 'python src/prepare_data.py' first.")
        sys.exit(1)

    open_count = sum(1 for _ in open_dir.iterdir() if _.is_file())
    closed_count = sum(1 for _ in closed_dir.iterdir() if _.is_file())

    if open_count == 0 or closed_count == 0:
        print(f"[!] ERROR: Dataset folders are empty (open: {open_count}, closed: {closed_count}).")
        print("[!] Populate the dataset before training.")
        sys.exit(1)

    print(f"[*] Found {open_count:,} open eye images and {closed_count:,} closed eye images.")
    return open_count, closed_count


def build_eye_model(img_size: int = 96) -> tf.keras.Model:
    """
    Construct transfer learning model using MobileNetV2 as frozen base
    with custom classification head and embedded data augmentation.
    """
    # 1. Input layer
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # 2. Data Augmentation layers
    # Mimics variable cabin illumination (night, high beams, shadows) and driver head angles
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.08, name="aug_rotation")(x)  # +/- ~15 degrees
    x = layers.RandomBrightness(0.2, name="aug_brightness")(x)
    x = layers.RandomContrast(0.2, name="aug_contrast")(x)

    # 3. MobileNetV2 Preprocessing: scales [0, 255] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # 4. Pretrained MobileNetV2 Base (frozen)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    # Keep base in inference mode so BatchNormalization layers retain ImageNet statistics
    x = base_model(x, training=False)

    # 5. Lightweight Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="eye_state_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Eye_Classifier")
    return model


def plot_and_save_curves(history, save_path: Path):
    """Plot training and validation accuracy/loss curves and save to disk."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#FFFFFF")

    # Accuracy Plot
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2563EB", marker="o", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#16A34A", marker="s", linewidth=2)
    ax1.set_title("Eye Model - Accuracy Curve", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0.80, 1.01])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)

    # Loss Plot
    ax2.plot(epochs_range, loss, label="Training Loss", color="#DC2626", marker="o", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#EA580C", marker="s", linewidth=2)
    ax2.set_title("Eye Model - Loss Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Binary Crossentropy Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Eye State Classifier.")
    parser.add_argument("--img-size", type=int, default=96, choices=[96, 128],
                        help="Input image resolution (96 or 128, default: 96 for laptop CPU speed).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training (default: 64).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Maximum training epochs (default: 10).")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Initial learning rate for Adam optimizer (default: 0.001).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total training samples to speed up CPU training (e.g. 20000).")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test on 8,000 samples (runs in ~3-4 minutes on CPU).")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and evaluate existing model at models/eye_model.keras.")

    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    if args.eval_only:
        print("      EVALUATING EYE STATE CLASSIFIER (models/eye_model.keras)")
    else:
        print("      TRAINING EYE STATE CLASSIFIER (OPEN vs CLOSED)")
    print("=" * 65)
    if not args.eval_only:
        print(f"[*] Base Architecture : MobileNetV2 (ImageNet weights, frozen)")
        print(f"[*] Input Resolution  : {args.img_size}x{args.img_size}x3")
        print(f"[*] Batch Size        : {args.batch_size}")
        print(f"[*] Max Epochs        : {args.epochs}")
        print(f"[*] Learning Rate     : {args.learning_rate}")

    # 1. Dataset Verification
    verify_dataset_structure(DATA_DIR)

    # 2. Load Train and Validation Splits (80% / 20%)
    print("\n[*] Loading dataset using image_dataset_from_directory (80/20 split)...")
    # Setting class_names ensures 0 = closed, 1 = open
    train_ds = None
    if not args.eval_only:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            labels="inferred",
            label_mode="binary",
            class_names=["closed", "open"],
            batch_size=args.batch_size,
            image_size=(args.img_size, args.img_size),
            shuffle=True,
            seed=42,
            validation_split=0.2,
            subset="training"
        )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["closed", "open"],
        batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="validation"
    )

    # Apply sample capping if requested
    if args.quick:
        max_samples = 8000
    else:
        max_samples = args.max_samples

    if max_samples:
        val_batches = max(1, int((max_samples * 0.2) // args.batch_size))
        val_ds = val_ds.take(val_batches)
        if train_ds is not None:
            train_batches = max(1, int((max_samples * 0.8) // args.batch_size))
            print(f"[*] Subsampling dataset to ~{max_samples:,} images ({train_batches} train batches, {val_batches} val batches)...")
            train_ds = train_ds.take(train_batches)

    # 3. Optimize dataset pipeline with prefetching (prevents disk IO bottleneck)
    if train_ds is not None:
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Evaluation-Only Mode
    if args.eval_only:
        if not MODEL_SAVE_PATH.exists():
            print(f"[!] ERROR: Saved model not found at {MODEL_SAVE_PATH}.")
            print("[!] Train the model first before running --eval-only.")
            sys.exit(1)

        print(f"\n[*] Loading existing trained model from: {MODEL_SAVE_PATH}")
        model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

        print("\n" + "=" * 65)
        print("                 VALIDATION EVALUATION RESULTS")
        print("=" * 65)
        eval_results = model.evaluate(val_ds, return_dict=True, verbose=1)
        for name, value in eval_results.items():
            print(f"  - {name.capitalize():<16}: {value:.4f}")

        val_acc = eval_results.get("accuracy", eval_results.get("acc", 0.0))
        val_prec = eval_results.get("precision", 0.0)
        val_rec = eval_results.get("recall", 0.0)
        val_loss = eval_results.get("loss", 0.0)

        print("-" * 65)
        print(f"  Validation Loss     : {val_loss:.4f}")
        print(f"  Validation Accuracy : {val_acc * 100:.2f}%")
        print(f"  Validation Precision: {val_prec * 100:.2f}%")
        print(f"  Validation Recall   : {val_rec * 100:.2f}%")
        print("=" * 65)
        return

    # 4. Build Model
    print("\n[*] Constructing transfer learning model...")
    model = build_eye_model(img_size=args.img_size)
    model.summary()

    # 5. Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    # 6. Callbacks: Early Stopping, LR Scheduler, Model Checkpoint
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    # 7. Model Training
    print("\n[*] Starting training. Note: early stopping will halt if validation accuracy plateaus.")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=training_callbacks,
        verbose=1
    )

    # 8. Ensure best model is saved
    model.save(str(MODEL_SAVE_PATH))
    print(f"\n[+] Trained model saved successfully to: {MODEL_SAVE_PATH}")

    # 9. Final Evaluation on Validation Set
    print("\n" + "=" * 65)
    print("                 FINAL EVALUATION RESULTS")
    print("=" * 65)
    eval_results = model.evaluate(val_ds, return_dict=True, verbose=0)
    for name, value in eval_results.items():
        print(f"  - {name.capitalize():<16}: {value:.4f}")

    val_acc = eval_results.get("accuracy", eval_results.get("acc", 0.0))
    val_prec = eval_results.get("precision", 0.0)
    val_rec = eval_results.get("recall", 0.0)
    val_loss = eval_results.get("loss", 0.0)

    print("-" * 65)
    print(f"  Validation Loss     : {val_loss:.4f}")
    print(f"  Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"  Validation Precision: {val_prec * 100:.2f}%")
    print(f"  Validation Recall   : {val_rec * 100:.2f}%")
    print("=" * 65)

    # 10. Generate and save training curve plots
    if history is not None:
        plot_and_save_curves(history, PLOT_SAVE_PATH)


if __name__ == "__main__":
    main()
