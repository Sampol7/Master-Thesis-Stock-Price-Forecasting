%% Add path
scriptDir = fileparts(mfilename('fullpath'));
functionsDir = fullfile(scriptDir, '..', 'Functions');
addpath(genpath(functionsDir));


%% FNN SS
FNN_SS = csvread("rawResiduals_FNN_BO-TPE_standardscaler.csv");
analyzeResiduals(FNN_SS)

%% FNN RS
FNN_RS = csvread("rawResiduals_FNN_BO-TPE_robustscaler.csv");
analyzeResiduals(FNN_RS)

%% ARIMA-t-FNN SS
ARIMA_t_FNN_SS = csvread("rawResiduals_ARIMA_t_FNN_BO-TPE_standardscaler.csv");
analyzeResiduals(ARIMA_t_FNN_SS)

%% ARIMA-t-FNN RS
ARIMA_t_FNN_RS = csvread("rawResiduals_ARIMA_t_FNN_BO-TPE_robustscaler.csv");
analyzeResiduals(ARIMA_t_FNN_RS)


%% LSTM SS
LSTM_SS = csvread("rawResiduals_LSTM_BO-TPE_standardscaler.csv");
analyzeResiduals(LSTM_SS)

%% LSTM RS
LSTM_RS = csvread("rawResiduals_LSTM_BO-TPE_robustscaler.csv");
analyzeResiduals(LSTM_RS)

%% ARIMA-t-LSTM SS
ARIMA_t_LSTM_SS = csvread("rawResiduals_ARIMA_t_LSTM_BO-TPE_standardscaler.csv");
analyzeResiduals(ARIMA_t_LSTM_SS)

%% ARIMA-t-LSTM RS
ARIMA_t_LSTM_RS = csvread("rawResiduals_ARIMA_t_LSTM_BO-TPE_robustscaler.csv");
analyzeResiduals(ARIMA_t_LSTM_RS)


%% SVR SS
SVR_SS = csvread("rawResiduals_SVR_BO-TPE_standardscaler.csv");
analyzeResiduals(SVR_SS)

%% SVR RS
SVR_RS = csvread("rawResiduals_SVR_BO-TPE_robustscaler.csv");
analyzeResiduals(SVR_RS)

%% ARIMA-t-SVR SS
ARIMA_t_SVR_SS = csvread("rawResiduals_ARIMA_t_SVR_BO-TPE_standardscaler.csv");
analyzeResiduals(ARIMA_t_SVR_SS)

%% ARIMA-t-SVR RS
ARIMA_t_SVR_RS = csvread("rawResiduals_ARIMA_t_SVR_BO-TPE_robustscaler.csv");
analyzeResiduals(ARIMA_t_SVR_RS)









%% Function

function analyzeResiduals(residuals)
    %null hypothesis of no residual autocorrelation
    num_lags = ceil(log(length(residuals))); 
    [h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(residuals, Lags=num_lags);
    disp(['LBQ p-value raw residuals: ', num2str(pValue_lbq)]);
    
    [pxx, f] = pspectrum(residuals);
    geometric_mean = exp(mean(log(pxx)));
    arithmetic_mean = mean(pxx);
    flatness = geometric_mean / arithmetic_mean;
    disp(['Flatness raw residuals:', num2str(flatness)]);
    
    % null hypothesis of no conditional heteroscedasticity
    [h_arch, pValue_arch] = archtest(residuals, Lags=5);
    disp('ARCH test p-value raw Residuals: ');
    disp(pValue_arch);
end