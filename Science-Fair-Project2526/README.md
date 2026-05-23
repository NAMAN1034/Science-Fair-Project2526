# Science-Fair-Project2526

**A Neuroadaptive Closed-Loop Wearable Device Integrating Bio-Inspired Reinforcement Learning and Predictive Control for Real-Time Non-Invasive Tremor Modulation and Synthetic Dopaminergic Feedback in Parkinson’s Disease** 

_by Naman Pradhan_

This repository contains a neuroadaptive wearable pipeline for real‑time tremor tracking and modulation using EMG + IMU signals. The system fuses signal processing, CNN+LSTM prediction, hybrid MPC+RL control, and a synthetic dopaminergic feedback model. It is designed to run on a Raspberry Pi 4B with a Teensy microcontroller for sensor acquisition and control messaging.

**__What’s Inside:__**
- Real‑time pipeline for EMG+IMU fusion, prediction, and hybrid control.
- Offline analysis scripts for data preprocessing, model training, and control simulation.
- Logging and evaluation hooks for latency, energy, and tremor suppression metrics.
- Repository Structure

**Real‑Time Pipeline Overview**
- **Signal Processing**
  
  EMG bandpass + envelope, IMU tremor band + analytic phase, PSD and coupling metrics.
- **Prediction (CNN+LSTM)**
  
  Uses a trained model (TFLite or Keras) to predict tremor probability, phase, and amplitude.
- **Hybrid Control**
  
  Phase‑lock + MPC + RL with safety gating and rate limits.
- **Synthetic Dopamine**
  
  Reward prediction error modulates online learning for neuroadaptive behavior.
