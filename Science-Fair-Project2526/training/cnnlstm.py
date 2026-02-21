from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf

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
def build_network(input_shape, task, num_classes=0):
    #cnn extracts vibrations and lstm learns rhythm
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        #cnn feature extraction
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        #lstm memory+sequence recognition
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu')
    ])
    if task =="forecast":
        #output layer for sensor values
        model.add(tf.keras.layers.Dense(input_shape[1]))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])    
        return model

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
    #make windows
    X,y=create_sliding_windows(scaled_data, args.window)
    #split into 80% train and 20% test sets
    split=int(len(X)*0.8)
    X_train, X_test=X[:split], X[split:]
    y_train, y_test=y[:split], y[split:]
    #build+train
    model=build_network(input_shape=(args.window, len(feature_names)), task=args.task)
    #no 'memorizing' noise
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    history=model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    #save results
    os.makedirs("models", exist_ok=True)
    model.save(args.model_out)
    print(f"training done. model saved")

if __name__ == "__main__":
    main()
