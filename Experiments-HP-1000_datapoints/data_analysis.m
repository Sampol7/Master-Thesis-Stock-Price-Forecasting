clear;

%% Add path
scriptDir = fileparts(mfilename('fullpath'));
functionsDir = fullfile(scriptDir, '..', 'Functions');
addpath(genpath(functionsDir));

%% Data

data = readtable('hpq_data.csv');
data = data(end-999:end, :);


split_idx = height(data) - 100;
train_data = data(1:split_idx, :);
test_data = data(split_idx+1:end, :);

dates = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd');
train_dates = dates(1:split_idx, :);
test_dates = dates(split_idx+1:end, :);

close = data.Close;
train_close = close(1:split_idx, :);
test_close = close(split_idx+1:end, :);

logclose = log(close);
train_logclose = logclose(1:split_idx, :);
test_logclose = logclose(split_idx+1:end, :);

logreturns = diff(logclose);
train_logreturns = logreturns(1:split_idx-1, :);
test_logreturns = logreturns(split_idx:end, :);


csvwrite('close.csv', close);
csvwrite('train_close.csv', train_close);
csvwrite('test_close.csv', test_close);
csvwrite('train_logclose.csv', train_logclose);
csvwrite('logclose.csv', logclose);
csvwrite('train_logreturns.csv', train_logreturns);



%% ACF plots closing

num_lags = ceil(log(length(train_close))); 


figure;
subplot(2, 1, 1);
autocorr(train_close, 'NumLags', num_lags);
ylim([-1,1]);
title('');

subplot(2, 1, 2);
parcorr(train_close, 'NumLags', num_lags);
ylim([-1,1]);
title('');

figure;
subplot(2, 1, 1);
autocorr(train_logclose, 'NumLags', num_lags);
ylim([-1,1]);
title('');

subplot(2, 1, 2);
parcorr(train_logclose, 'NumLags', num_lags);
ylim([-1,1]);
title('');

figure;
subplot(2, 1, 1);
autocorr(train_logreturns, 'NumLags', num_lags);
ylim([-1,1]);
title('');

subplot(2, 1, 2);
parcorr(train_logreturns, 'NumLags', num_lags);
ylim([-1,1]);
title('');


%% LBQ

%null hypothesis of no residual autocorrelation
[h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(train_close, Lags=num_lags);
disp(['LBQ p-value Train Close: ', num2str(pValue_lbq, '%.4g')]);

%null hypothesis of no residual autocorrelation
[h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(train_logclose, Lags=num_lags);
disp(['LBQ p-value Train Log Close: ', num2str(pValue_lbq, '%.4g')]);

%null hypothesis of no residual autocorrelation
[h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(train_logreturns, Lags=num_lags);
disp(['LBQ p-value Train Log Returns: ', num2str(pValue_lbq, '%.4g')]);


%% Pspectrum residuals

[pxx1, f1] = pspectrum(train_close);
geometric_mean1 = exp(mean(log(pxx1)));
arithmetic_mean1 = mean(pxx1);
flatness1 = geometric_mean1 / arithmetic_mean1;
disp(['Flatness training close:', num2str(flatness1)]);

[pxx2, f2] = pspectrum(train_logclose);
geometric_mean2 = exp(mean(log(pxx2)));
arithmetic_mean2 = mean(pxx2);
flatness2 = geometric_mean2 / arithmetic_mean2;
disp(['Flatness training logclose:', num2str(flatness2)]);

[pxx3, f3] = pspectrum(train_logreturns);
geometric_mean3 = exp(mean(log(pxx3)));
arithmetic_mean3 = mean(pxx3);
flatness3 = geometric_mean3 / arithmetic_mean3;
disp(['Flatness training logreturns:', num2str(flatness3)]);


%% ARCH

% null hypothesis of no conditional heteroscedasticity
[h_arch, pValue_arch] = archtest(train_close, Lags=5);
disp('ARCH test p-value Train Close: ');
disp(pValue_arch);

% null hypothesis of no conditional heteroscedasticity
[h_arch, pValue_arch] = archtest(train_logclose, Lags=5);
disp('ARCH test p-value Train Log Close: ');
disp(pValue_arch);

% null hypothesis of no conditional heteroscedasticity
[h_arch, pValue_arch] = archtest(train_logreturns, Lags=5);
disp('ARCH test p-value Train Log Returns: ');
disp(pValue_arch);

%% Stationarity

% null hypothesis of a unit root against the autoregressive alternative
disp('--- ADF Tests ---');

[h_close, pValue_close, stat_close, cValue_close] = adftest(train_close);
disp(['ADF p-value train_close: ', num2str(pValue_close, '%.4g')]);

[h_logclose, pValue_logclose, stat_logclose, cValue_logclose] = adftest(train_logclose);
disp(['ADF p-value train_logclose: ', num2str(pValue_logclose, '%.4g')]);

[h_logreturns, pValue_logreturns, stat_logreturns, cValue_logreturns] = adftest(train_logreturns);
disp(['ADF p-value train_logreturns: ', num2str(pValue_logreturns, '%.4g')]);


% null hypothesis of a unit root against the AR(1) alternative
disp('--- PP Tests ---');

[h_close, pValue_close, stat_close, cValue_close] = pptest(train_close);
disp(['PP p-value train_close: ', num2str(pValue_close, '%.4g')]);

[h_logclose, pValue_logclose, stat_logclose, cValue_logclose] = pptest(train_logclose);
disp(['PP p-value train_logclose: ', num2str(pValue_logclose, '%.4g')]);

[h_logreturns, pValue_logreturns, stat_logreturns, cValue_logreturns] = pptest(train_logreturns);
disp(['PP p-value train_logreturns: ', num2str(pValue_logreturns, '%.4g')]);


% null hypothesis of a unit root against the AR(1) alternative
disp('--- KPSS Tests ---');

[h_close, pValue_close, stat_close, cValue_close] = kpsstest(train_close);
disp(['kpsstest p-value train_close: ', num2str(pValue_close, '%.4f')]);

[h_logclose, pValue_logclose, stat_logclose, cValue_logclose] = kpsstest(train_logclose);
disp(['kpsstest p-value train_logclose: ', num2str(pValue_logclose, '%.4f')]);

[h_logreturns, pValue_logreturns, stat_logreturns, cValue_logreturns] = kpsstest(train_logreturns);
disp(['kpsstest p-value train_logreturns: ', num2str(pValue_logreturns, '%.4f')]);




