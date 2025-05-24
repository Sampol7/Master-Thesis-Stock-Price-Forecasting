# Functions to compute several forecasting metrics

import numpy as np

def compute_error_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    errors = y_true - y_pred

    n = len(y_true)
    
    NMSE = np.sum(errors**2) / np.sum((y_true - np.mean(y_true))**2)
    MAE = np.mean(np.abs(errors))
    RMSE = np.sqrt(np.mean(errors**2))
    MSE = np.mean(errors**2)
    MAPE = np.mean(np.abs(errors / y_true)) * 100

    U_numerator = np.sum(((y_pred[1:n] - y_true[1:n]) / y_true[:n-1])**2)
    U_denominator = np.sum(((y_true[1:n] - y_true[:n-1]) / y_true[:n-1])**2)
    theilsU = np.sqrt(U_numerator / U_denominator) if U_denominator != 0 else np.nan

    return {'NMSE': NMSE, 'MAE': MAE, 'RMSE': RMSE, 'MSE': MSE, 'MAPE': MAPE, 'TheilsU': theilsU}


def compute_directional_metrics(y_true, y_pred):

    actual_diff = np.diff(y_true)
    predicted_diff = np.diff(y_pred)
    predicted_direction = y_pred[1:] - y_true[:-1]

    N = len(actual_diff)
    
    d_t_DA = (actual_diff * predicted_direction) > 0
    DA = np.sum(d_t_DA) / N

    d_t_DS = (actual_diff * predicted_diff) > 0
    DS = np.sum(d_t_DS) / N

    d_t_CU = (predicted_diff > 0) & (actual_diff * predicted_diff >= 0)
    k_t_CU = actual_diff > 0
    CU = np.sum(d_t_CU) / np.sum(k_t_CU) if np.sum(k_t_CU) > 0 else np.nan

    d_t_CD = (predicted_diff < 0) & (actual_diff * predicted_diff >= 0)
    k_t_CD = actual_diff < 0
    CD = np.sum(d_t_CD) / np.sum(k_t_CD) if np.sum(k_t_CD) > 0 else np.nan

    return {'DA': DA, 'DS': DS, 'CU': CU, 'CD': CD}


def compute_forecast_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    errors = y_true - y_pred

    n = len(y_true)
    
    NMSE = np.sum(errors**2) / np.sum((y_true - np.mean(y_true))**2)
    MAE = np.mean(np.abs(errors))
    RMSE = np.sqrt(np.mean(errors**2))
    MSE = np.mean(errors**2)
    MAPE = np.mean(np.abs(errors / y_true)) * 100

    U_numerator = np.sum(((y_pred[1:n] - y_true[1:n]) / y_true[:n-1])**2)
    U_denominator = np.sum(((y_true[1:n] - y_true[:n-1]) / y_true[:n-1])**2)
    theilsU = np.sqrt(U_numerator / U_denominator) if U_denominator != 0 else np.nan


    actual_diff = np.diff(y_true)
    predicted_diff = np.diff(y_pred)
    predicted_direction = y_pred[1:] - y_true[:-1]

    N = len(actual_diff)
    
    d_t_DA = (actual_diff * predicted_direction) > 0
    DA = np.sum(d_t_DA) / N

    d_t_DS = (actual_diff * predicted_diff) > 0
    DS = np.sum(d_t_DS) / N

    d_t_CU = (predicted_diff > 0) & (actual_diff * predicted_diff >= 0)
    k_t_CU = actual_diff > 0
    CU = np.sum(d_t_CU) / np.sum(k_t_CU) if np.sum(k_t_CU) > 0 else np.nan

    d_t_CD = (predicted_diff < 0) & (actual_diff * predicted_diff >= 0)
    k_t_CD = actual_diff < 0
    CD = np.sum(d_t_CD) / np.sum(k_t_CD) if np.sum(k_t_CD) > 0 else np.nan

    return {'NMSE': NMSE, 'MAE': MAE, 'RMSE': RMSE, 'MSE': MSE, 'MAPE': MAPE, 'TheilsU': theilsU, 'DA': DA, 'DS': DS, 'CU': CU, 'CD': CD}


