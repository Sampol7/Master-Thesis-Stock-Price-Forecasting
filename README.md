# Stock Price Forecasting: A Hybrid Approach Using System Identification and Machine Learning

This repository contains the code used for the experiments conducted in my master's thesis:  
**S. Vergauwen. *Stock Price Forecasting: A Hybrid Approach Using System Identification and Machine Learning*. Master’s thesis, KU Leuven, 2025.**

## Overview

The project explores stock price forecasting using a hybrid approach that combines **System Identification (SYSID)** techniques with **Machine Learning (ML)** models. 

## Repository Structure

- **`Functions/`**  
  Helper functions for SYSID model selection and performance evaluation.

- **Experiment Folders**  
  Each folder contains experiments applied to a different dataset. These include MATLAB, Python, and R scripts, as well as `.csv` files for preprocessed data, residuals, and forecasts:
  - `Experiments Synthetic Data/`
  - `Experiments - HP - 1000 datapoints/`
  - `Experiments - HP - 2000 datapoints/`
  - `Experiments - HP - 4000 datapoints/`
  - `Experiments - HP - 8000 datapoints/`

## Structure of `Experiments Synthetic Data/`

### Data Preparation and Analysis
- `data_analysis.m`: Generate and analyze synthetic data.

### Baseline Forecasting Models
- `Naive_forecasts.m`
- `random_walk_forecasts.m`

### System Identification (SYSID) Models

**Individual Models:**
- `ARIMA.m`, `ARIMA_t.m`, `ARIMA_GARCH.m`, `ARIMA_t_GARCH_t.m`: Model  selection, residual analysis (no nonlinearity tests) and forecasting.

**Model Grid Evaluation:**
- `ARIMA_all_models.m`, `ARIMA_t_all_models.m`
- `ARIMA_GARCH_all_models.m`, `ARIMA_t_GARCH_t_all_models.m`

**Confidence Intervals (CIs):**
- `ARIMA_CI.m`, `ARIMA_GARCH_CI.m`

**Trading Strategy Applied to SYSID Forecasts:**
- `SYSID_trading_strategy.ipynb`

### Feedforward Neural Networks (FNN)
- `FNN_example_with_early_stopping.ipynb`, `FNN_example_without_early_stopping.ipynb`: Early stopping experiments.
- `FNN_hyperopt_BO-TPE_standardscaler.ipynb`, `FNN_hyperopt_rand_standardscaler.ipynb`: Hyperparameter optimization experiments: Randomized Search vs. Bayesian Optimization with Tree-structured Parzen Estimator (BO-TPE).
- `FNN_hyperopt_BO-TPE_robustscaler.ipynb`, `FNN_hyperopt_BO-TPE_standardscaler.ipynb`, `ARIMA_t_FNN_hyperopt_BO-TPE_robustscaler.ipynb`, `ARIMA_t_FNN_hyperopt_BO-TPE_standardscaler.ipynb`: Hyperparameter optimization and forecasting.

### Support Vector Regression (SVR)
- `SVR_hyperopt_BO-TPE_robustscaler.ipynb`, `SVR_hyperopt_BO-TPE_standardscaler.ipynb`, `ARIMA_t_SVR_hyperopt_BO-TPE_robustscaler.ipynb`, `ARIMA_t_SVR_hyperopt_BO-TPE_standardscaler.ipynb`: Hyperparameter optimization and forecasting.

### Long Short-Term Memory Networks (LSTM)
- `LSTM_hyperopt_BO-TPE_robustscaler.ipynb`, `LSTM_hyperopt_BO-TPE_standardscaler.ipynb`, `ARIMA_t_LSTM_hyperopt_BO-TPE_robustscaler.ipynb`, `ARIMA_t_LSTM_hyperopt_BO-TPE_standardscaler.ipynb`: Hyperparameter optimization and forecasting.

### Residual Analysis
- `ML_residual_analysis.m`: Analyze ML model residuals (no nonlinearity testing).

### Nonlinearity Testing
- `NonlinearityTests.R`: Statistical tests for nonlinearity in residuals of SYSID, ML and SYSID-ML.

## Citation

If you use this repository in your research or work, please cite:

> **S. Vergauwen**  
> *Stock Price Forecasting: A Hybrid Approach Using System Identification and Machine Learning*  
> Master’s Thesis, KU Leuven, 2025.
