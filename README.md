# RNNs with PyTorch

This repository contains a hands-on lab notebook that demonstrates **Recurrent Neural Networks (RNNs)** and their variants using **PyTorch**. The notebook is designed for educational purposes and time series forecasting tasks.

---

## Table of Contents

- [Overview](#overview)  
- [Objectives](#objectives)  
- [Notebook Structure](#notebook-structure)  
- [Models Implemented](#models-implemented)  
- [Time Series Forecasting](#time-series-forecasting)  
- [Stock Price Prediction](#stock-price-prediction)  
- [Requirements](#requirements)  
- [Usage](#usage)  
- [Author](#author)

---

## Overview

This lab demonstrates the implementation of different RNN architectures for sequential data and time series forecasting using PyTorch. The exercises include predicting the next time step of synthetic time series and forecasting real-world stock prices (AAPL).

The notebook covers:
- Creating synthetic time series data
- Building RNN models from scratch
- Multi-step forecasting
- Time series prediction using LSTM and GRU

---

## Objectives

- Understand RNN architecture and its components  
- Learn how to implement RNNs, Deep RNNs, LSTM, and GRU in PyTorch  
- Apply RNNs to synthetic and real-world time series data  
- Explore multi-step forecasting  

---

## Notebook Structure

1. **Introduction and Imports**  
   Import necessary libraries and set up the device for GPU acceleration.  

2. **Synthetic Time Series Forecasting**  
   - Generate synthetic sine wave data with noise  
   - Train models to predict the next time step  

3. **Model Implementation**  
   - Fully Connected Network (FCN)  
   - Simple RNN  
   - Deep RNN  
   - LSTM  
   - GRU  

4. **Multi-Step Forecasting**  
   - Forecast multiple future time steps  
   - Compare predicted values with actual values  

5. **Stock Price Prediction**  
   - Download Apple (AAPL) historical data using `yfinance`  
   - Preprocess data and create sequences  
   - Train RNN models for stock price forecasting  

---

## Models Implemented

- **Fully Connected Network (FCN)**  
- **Simple RNN**  
- **Deep RNN** (stacked RNN layers)  
- **LSTM** (Long Short-Term Memory)  
- **GRU** (Gated Recurrent Unit)  
- **Multi-Step RNN** for predicting multiple future steps  

---

## Time Series Forecasting

- Generated 10,000 synthetic time series of length 51  
- 70% used for training, 30% for testing  
- Models trained to predict the last time step given previous 50 steps  

**Visualization**: Predicted vs actual time series points.

---

## Stock Price Prediction

- Historical Apple (AAPL) stock prices (2010-2021)  
- Objective: Predict future closing prices using RNN, LSTM, and GRU models  
- Challenges addressed:
  - Temporal dependency  
  - Trends and seasonality  
  - Noise and volatility  

---

## Requirements

- Python 3.x  
- Libraries:
  ```bash
  numpy
  torch
  matplotlib
  pandas
  scikit-learn
  yfinance
  torchsummary
