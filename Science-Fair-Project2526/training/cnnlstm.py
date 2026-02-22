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
    parser.add_argument("--csv", required=True, help="/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/imuandemgdata.csv")
    parser.add_argument("--model-out", default="/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/tremor_model.keras")
    #hyperparams
    parser.add_argument("--window", type=int, default=128, help="# of samples per snapshot")
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
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, channels)),
        #1st layer (basic edges in tremor)
        tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        #2nd layer (complex rhythmic patterns)
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.Dropout(0.2),
        #memory
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu")
    ])
    if task=="forecast":
        model.add(tf.keras.layers.Dense(channels))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=['mae'])
    else:
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='d62728')
    else:
        plt.plot(history.history['mae'], label='train mae', color='9467bd')
        plt.plot(history.history['val_mae'], label='val mae', color='#8c564b')
        plt.title('model accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/training_performance.png")
    print("done graphing")

#main stuff
def main():
    args=get_settings()
    print(f"starting training for{args.task}")
    #load and process
    df, feature_names = prepare_sensor_data(args.csv, args.task)
    sensor_values=df[feature_names].values
    #scale data
    mean=sensor_values.mean(axis=0)
    std=sensor_values.std(axis=0)
    scaled_data=(sensor_values-mean)/(std+1e-7)
    np.savez("/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/tremor_model_norm.npz", mean=mean, std=std)
    #make windows
    X,y=create_sliding_windows(scaled_data, args.window)
    #split into 70% train, 15% values, and 15% test sets
    train_idx=int(len(X)*0.7)
    val_idx=int(len(X)*0.85)

    X_train, X_val, X_test=X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test=y[:train_idx], y[train_idx:val_idx], y[val_idx:]
    #build+train
    model = build_model(window=args.window, channels=len(feature_names), task=args.task)   
    #no 'memorizing' noise
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)]
    history=model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose = 1
    )
    
    #save results
    os.makedirs("models", exist_ok=True)
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
