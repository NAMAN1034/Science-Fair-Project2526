from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#setup
def get_settings():
    #define params for training
    parser = argparse.ArgumentParser(description="train the tremor analysis model")
    #data stuff
    parser.add_argument("--csv",required=True,help="/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/imuandemgdata.csv",)
    parser.add_argument("--model-out",default="/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/models/tremor_model.keras",)
    parser.add_argument("--norm-out",default="/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/models/tremor_model_norm.npz",)
    parser.add_argument("--meta-out",default="/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/models/tremormodelmeta.json",)
    #hyperparams
    parser.add_argument("--window", type=int, default=128, help="# of samples per snapshot")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1, help="forecast only")
    parser.add_argument("--label-col", default=None, help="required for classify")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--task", default="forecast", choices=["forecast", "classify"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()
#prepping data
def prepare_sensor_data(path, task, label_col=None):
    #load the csv and clean noise and/or missing values
    if not os.path.exists(path):
        raise FileNotFoundError(f"no file found at given path")
    df=pd.read_csv(path)
    features=["emg", "ax", "ay", "az", "gx", "gy", "gz"]
    #time sort
    if "time" in df.columns:
        df=df.sort_values("time")
    #remove NaNs
    selected_cols=features+([label_col] if label_col else[])
    df=df[selected_cols].dropna().reset_index(drop=True)
    return df, features

def create_sliding_windows(data, window_size, stride=4):
    #chop stream into windows
    x_list, y_list = [],[]
    for i in range(0, len(data)-window_size, stride):
        window=data[i:i+window_size]
        target=data[i+window_size]
        x_list.append(window)
        y_list.append(target)
    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
#model architecture
def build_model(window, channels, task, num_classes=0, lr=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, channels)),
        tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
    ])
    if task == "forecast":
        model.add(tf.keras.layers.Dense(channels))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"],
        )
    else:
        if num_classes < 2:
            raise ValueError("classification requires at least 2 classes")
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model
def plot_results(history, task):
    plt.style.use('ggplot')#make graph look cool
    plt.figure(figsize=(12,5))
    #graph error loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='training loss', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_loss'], label='validation loss', color='#ff7f0e', linestyle='--')
    plt.title('Model Error')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend()
    #graph accuracy/mae(mean absolute error)
    plt.subplot(1,2,2)
    if task =='classify':
        plt.plot(history.history['accuracy'], label='training accuracy', color='#2ca02c')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#d62728')
    else:
        plt.plot(history.history['mae'], label='train mae', color='#9467bd')
        plt.plot(history.history['val_mae'], label='val mae', color='#8c564b')
        plt.title('model accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/models/training_performance.png")
    print("done graphing")
def validate_args(args):
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0,1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
def prepare_sensor_data(path, task, label_col=None):
    if not os.path.exists(path):
        raise FileNotFoundError("no file found at given path")
    df = pd.read_csv(path)
    features = ["emg", "ax", "ay", "az", "gx", "gy", "gz"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if "time" in df.columns:
        df = df.sort_values("time")
    class_mapping = {}
    y_all = None
    if task == "classify":
        if not label_col:
            raise ValueError("--label-col is required when --task classify")
        if label_col not in df.columns:
            raise ValueError(f"label column not found: {label_col}")
        df = df[features + [label_col]].dropna().reset_index(drop=True)

        labels = df[label_col].astype("category")
        class_mapping = {str(cat): int(i) for i, cat in enumerate(labels.cat.categories)}
        y_all = labels.cat.codes.to_numpy(dtype=np.int32)
    else:
        df = df[features].dropna().reset_index(drop=True)
    return df, features, y_all, class_mapping
def create_windows_forecast(data, window_size, stride=1, horizon=1):
    x_list, y_list = [], []
    max_start = len(data) - window_size - horizon + 1
    for i in range(0, max_start, stride):
        end = i + window_size
        x_list.append(data[i:end])
        y_list.append(data[end + horizon - 1])
    if not x_list:
        raise ValueError("no forecast windows created")
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)
def create_windows_classify(data, labels, window_size, stride=1):
    x_list, y_list = [], []
    max_start = len(data) - window_size + 1
    for i in range(0, max_start, stride):
        end = i + window_size
        x_list.append(data[i:end])
        y_list.append(int(labels[end - 1]))
    if not x_list:
        raise ValueError("no classification windows created")
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.int32)

#main stuff
def main():
    args = get_settings()
    validate_args(args)
    print(f"starting training for {args.task}")
    df, feature_names, y_all, class_mapping = prepare_sensor_data(args.csv, args.task, args.label_col)
    sensor_values = df[feature_names].to_numpy(dtype=np.float32)
    min_needed = args.window + (args.horizon if args.task == "forecast" else 1)
    if len(sensor_values) < min_needed:
        raise ValueError(f"not enough rows ({len(sensor_values)}) for window={args.window}")
    #stop data leakage
    train_end = int(len(sensor_values) * args.train_ratio)
    val_end = int(len(sensor_values) * (args.train_ratio + args.val_ratio))
    train_raw = sensor_values[:train_end]
    val_raw = sensor_values[train_end:val_end]
    test_raw = sensor_values[val_end:]
    for name, arr in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        if len(arr) < min_needed:
            raise ValueError(f"{name} split too small for window/horizon. rows={len(arr)}")
    # normalize from train only
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0) + 1e-7
    train_scaled = (train_raw - mean) / std
    val_scaled = (val_raw - mean) / std
    test_scaled = (test_raw - mean) / std
    if args.task == "forecast":
        X_train, y_train = create_windows_forecast(train_scaled, args.window, args.stride, args.horizon)
        X_val, y_val = create_windows_forecast(val_scaled, args.window, args.stride, args.horizon)
        X_test, y_test = create_windows_forecast(test_scaled, args.window, args.stride, args.horizon)
        num_classes = 0
    else:
        y_train_raw = y_all[:train_end]
        y_val_raw = y_all[train_end:val_end]
        y_test_raw = y_all[val_end:]
        X_train, y_train = create_windows_classify(train_scaled, y_train_raw, args.window, args.stride)
        X_val, y_val = create_windows_classify(val_scaled, y_val_raw, args.window, args.stride)
        X_test, y_test = create_windows_classify(test_scaled, y_test_raw, args.window, args.stride)
        num_classes = int(np.max(y_all)) + 1
    model = build_model(args.window, len(feature_names), args.task, num_classes=num_classes)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    plot_results(history, args.task)
    model_dir = os.path.dirname(args.model_out)
    norm_dir = os.path.dirname(args.norm_out)
    meta_dir = os.path.dirname(args.meta_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if norm_dir:
        os.makedirs(norm_dir, exist_ok=True)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)

    model.save(args.model_out)
    np.savez(args.norm_out, mean=mean, std=std, feature_names=np.array(feature_names, dtype=object))

    metadata = {
        "task": args.task,
        "window_size": args.window,
        "stride": args.stride,
        "horizon": args.horizon,
        "features": feature_names,
        "class_mapping": class_mapping,
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("training done. model + norm + metadata saved.")
    plot_results(history, args.task)

    #save results
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"training done. model saved")
    #save a summary
    metadata={
        "task":args.task,
        "final_loss":float(history.history['loss'][-1]),
        "final_val_loss":float(history.history["val_loss"][-1]),
        "window_size":args.window
    }
    with open("/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/tremormodelmeta.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("metadata saved")

if __name__ == "__main__":
    main()
