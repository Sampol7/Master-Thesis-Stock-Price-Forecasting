clear;

%% Add path
scriptDir = fileparts(mfilename('fullpath'));
functionsDir = fullfile(scriptDir, '..', 'Functions');
addpath(genpath(functionsDir));


%% Data
rng(2); 

arimaModel = arima(1, 1, 1);
arimaModel.Constant = 0;
arimaModel.AR = {-0.5};  
arimaModel.MA = {-0.1};    
garchModel = garch(1, 1);
garchModel.Constant = 0.1;
garchModel.ARCH = {0.3};   
garchModel.GARCH = {0.5};  
arimaModel.Variance = garchModel;  

arimaModel1 = arima(1, 0, 0);
arimaModel1.Constant = 0;   
arimaModel1.AR = {0.58};      
arimaModel1.Variance = 1;  


numObservations = 1000;
split_idx = numObservations - 100;

simulatedData = simulate(arimaModel, numObservations);
simulatedData1 = simulate(arimaModel1, numObservations);


lin = -simulatedData;
figure;
plot(lin);
xlabel('Time');
ylabel('Value');

nonlin = 0.3*(simulatedData1-2).*(circshift(simulatedData1,1)+2); 
figure;
plot(nonlin);
xlabel('Time');
ylabel('Value');

figure;
plot(simulatedData1);
xlabel('Time');
ylabel('Value');

figure;
plot(simulatedData1.*circshift(simulatedData1,1));
xlabel('Time');
ylabel('Value');

%noise = 0.05*randn(numObservations, 1);
noise = 0.1*std(nonlin)*randn(numObservations, 1);
figure;
plot(noise);
xlabel('Time');
ylabel('Value');


sum_terms = lin + nonlin + noise;
figure;
plot(sum_terms);
xlabel('Time');
ylabel('Value');


timeFactor = linspace(0.3, 1, numObservations)'; 
close = sum_terms.*timeFactor + 15;
figure;
plot(close);
xlabel('Time');
ylabel('Value');

train_close = close(1:split_idx, :);
test_close = close(split_idx+1:end, :);

figure;
plot(train_close);
xlabel('Time');
ylabel('Close');

logclose = log(close);
figure;
plot(logclose);
title('Log Close');
xlabel('Time');
ylabel('Value');

train_logclose = logclose(1:split_idx, :);
test_logclose = logclose(split_idx+1:end, :);

figure;
plot(train_logclose);
title('Log Close of Training Data');
xlabel('Time');
ylabel('Value');

logreturns = diff(logclose);
figure;
plot(logreturns);
title('Log Returns');
xlabel('Time');
ylabel('Value');

train_logreturns = logreturns(1:split_idx-1, :);
test_logreturns = logreturns(split_idx:end, :);

figure;
plot(train_logreturns);
xlabel('Time');
ylabel('Log Returns');

csvwrite('close.csv', close);
csvwrite('train_close.csv', train_close);
csvwrite('test_close.csv', test_close);
csvwrite('train_logclose.csv', train_logclose);
csvwrite('logclose.csv', logclose);
csvwrite('train_logreturns.csv', train_logreturns);




%% ACF plots 

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
autocorr(train_logreturns, 'NumLags', num_lags);
ylim([-1,1]);
title('');

subplot(2, 1, 2);
parcorr(train_logreturns, 'NumLags', num_lags);
ylim([-1,1]);
title('');


figure;
autocorr(train_close, 'NumLags', num_lags);
ylim([-1,1]);
title('');

figure;
autocorr(train_logreturns, 'NumLags', num_lags);
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




%% t-dist train_close (https://nl.mathworks.com/matlabcentral/answers/1599589-manual-t-test-gives-different-result-for-same-t-value)

fitted_t = fitdist(train_close, 'tlocationscale');
mu_t = fitted_t.mu;        
sigma_t = fitted_t.sigma;     
nu_t = fitted_t.nu;

figure;
qqplot(train_close, makedist('tLocationScale', 'mu', mu_t, 'sigma', sigma_t, 'nu', nu_t));
xlabel('Theoretical Quantiles (Student''s t)');
ylabel('Sample Quantiles');
title('');


x = linspace(min(train_close), max(train_close), 1000);
t_pdf = tpdf((x - mu_t) / sigma_t, nu_t) / sigma_t;

figure;
histogram(train_close, 'Normalization', 'pdf');
hold on;
plot(x, t_pdf, 'r-', 'LineWidth', 2);
xlabel('Close');
ylabel('Probability Density');
legend('Log Return Histogram', 'Student''s t-Distribution');
hold off;


%% t-dist train_logreturns

fitted_t = fitdist(train_logreturns, 'tlocationscale');
mu_t = fitted_t.mu;        
sigma_t = fitted_t.sigma;     
nu_t = fitted_t.nu;
disp('Degrees of Freedom t-dist train_logreturns:')
disp(nu_t);
disp('Kurtosis train_logreturns:')
disp(kurtosis(train_logreturns));

figure;
qqplot(train_logreturns, makedist('tLocationScale', 'mu', mu_t, 'sigma', sigma_t, 'nu', nu_t));
xlabel('Theoretical Quantiles (Student''s t)');
ylabel('Sample Quantiles');
title('');


x = linspace(min(train_logreturns), max(train_logreturns), 1000);
t_pdf = tpdf((x - mu_t) / sigma_t, nu_t) / sigma_t;

figure;
histogram(train_logreturns, 'Normalization', 'pdf');
hold on;
plot(x, t_pdf, 'r-', 'LineWidth', 2);
xlabel('Log Return');
ylabel('Probability Density');
legend('Log Return Histogram', 'Student''s t-Distribution');
hold off;



%% Variance 


train_close_variance = movvar(train_close, 100);
train_close_variance_standardized = train_close_variance / mean(train_close_variance);

figure;
plot(train_close_variance_standardized);
xlabel('Time');
ylabel('Standardized Variance');
ylim([0,7]);


train_logreturns_variance = movvar(train_logreturns, 100);
train_logreturns_variance_standardized = train_logreturns_variance / mean(train_logreturns_variance);

figure;
plot(train_logreturns_variance_standardized);
xlabel('Time');
ylabel('Standardized Variance');
ylim([0,7]);
