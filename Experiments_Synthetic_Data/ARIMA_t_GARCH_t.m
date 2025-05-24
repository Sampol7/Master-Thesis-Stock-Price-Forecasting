clear;

%% Add path
scriptDir = fileparts(mfilename('fullpath'));
functionsDir = fullfile(scriptDir, '..', 'Functions');
addpath(genpath(functionsDir));


%% Data
rng(2); 

close = csvread("close.csv");
split_idx = size(close,1)-100;

train_close = close(1:split_idx, :);
test_close = close(split_idx+1:end, :);

logclose = log(close);
train_logclose = logclose(1:split_idx, :);
test_logclose = logclose(split_idx+1:end, :);

logreturns = diff(logclose);
train_logreturns = logreturns(1:split_idx-1, :);
test_logreturns = logreturns(split_idx:end, :);


%% Fit ARIMA-t-GARCH-t for various pA,qA,pG,qG
max_pqA = 5;
max_pqG = 2;
results_ARIMAt_GARCHt_p1q = ARIMAt_GARCHt_p1q(train_logclose, max_pqA, max_pqG);

%% Find best ARIMA-t-GARCH-t from all fits, based on aic, p-value, and correlation
best_pqAG = find_best_ARIMAt_GARCHt(train_logclose,results_ARIMAt_GARCHt_p1q);


%% Fit best ARIMA-t
pA_best = best_pqAG(1);
qA_best = best_pqAG(2);
pG_best = best_pqAG(3);
qG_best = best_pqAG(4);
pvalue_best = best_pqAG(5);
aic_best = best_pqAG(6);
disp(['Best ARIMA-t-GARCH-t model: ARIMA-t(', num2str(pA_best), ',1,', num2str(qA_best), ')-GARCH-t(', num2str(pG_best), ',', num2str(qG_best), ')']);
disp("pvalue_best:")
disp(num2str(pvalue_best, '%.6f'));
disp("aic_best:")
disp(aic_best)

modelARIMAGARCH = arima(pA_best, 1, qA_best);
modelARIMAGARCH.Distribution = 't';
modelGARCH = garch(pG_best, qG_best);
modelGARCH.Distribution = 't';
modelARIMAGARCH.Variance = modelGARCH;
fitARIMAGARCH = estimate(modelARIMAGARCH, train_logclose);

[res_arimagarch, var_arimagarch] = infer(fitARIMAGARCH, train_logclose);
stan_res_arimagarch = res_arimagarch ./ sqrt(var_arimagarch);

train_MSE = sum(res_arimagarch.^2)/size(res_arimagarch,1);
disp(['Training MSE: ' num2str(train_MSE)]);

csvwrite('rawResiduals_ARIMA_t_GARCH_t.csv', res_arimagarch);
csvwrite('stanResiduals_ARIMA_t_GARCH_t.csv', stan_res_arimagarch);

%% Pspectrum residuals

[pxx2, f2] = pspectrum(res_arimagarch);
geometric_mean2 = exp(mean(log(pxx2)));
arithmetic_mean2 = mean(pxx2);
flatness2 = geometric_mean2 / arithmetic_mean2;
disp(['Flatness raw residuals:', num2str(flatness2)]);


[pxx3, f3] = pspectrum(stan_res_arimagarch);
geometric_mean3 = exp(mean(log(pxx3)));
arithmetic_mean3 = mean(pxx3);
flatness3 = geometric_mean3 / arithmetic_mean3;
disp(['Flatness standardized residuals:', num2str(flatness3)]);

%% Verify correlation residuals
num_lags = ceil(log(length(stan_res_arimagarch)));

figure;
subplot(2, 1, 1);
autocorr(stan_res_arimagarch, 'NumLags', num_lags);
ylim([-0.1,0.1]);
title('');

subplot(2, 1, 2);
parcorr(stan_res_arimagarch, 'NumLags', num_lags);
ylim([-0.1,0.1]);
title('');

%null hypothesis of no residual autocorrelation
[h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(res_arimagarch, Lags=num_lags);
disp(['LBQ p-value raw residuals: ', num2str(pValue_lbq)]);

%null hypothesis of no residual autocorrelation
[h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(stan_res_arimagarch, Lags=num_lags);
disp(['LBQ p-value standardized residuals: ', num2str(pValue_lbq)]);


%% Verify dist residuals

mu = mean(res_arimagarch);
sigma = std(res_arimagarch);
x = linspace(min(res_arimagarch), max(res_arimagarch), 1000);
normal_pdf = normpdf(x, mu, sigma);

figure;
qqplot(res_arimagarch, makedist('Normal', 'mu', mu, 'sigma', sigma));
xlabel('Theoretical Quantiles (Normal)');
ylabel('Sample Quantiles');
title('');


fitted_t = fitdist(res_arimagarch, 'tlocationscale');
mu_t = fitted_t.mu;        
sigma_t = fitted_t.sigma;     
nu_t = fitted_t.nu;

figure;
qqplot(res_arimagarch, makedist('tLocationScale', 'mu', mu_t, 'sigma', sigma_t, 'nu', nu_t));
xlabel('Theoretical Quantiles (Student''s t)');
ylabel('Sample Quantiles');
title('');


%% Verify variance residuals

stan_res_arimagarch_variance = movvar(stan_res_arimagarch, 100);
figure;
plot(stan_res_arimagarch_variance);
ylim([0.2,2.2]);
xlabel('Date');
ylabel('Variance');

% null hypothesis of no conditional heteroscedasticity
[h_arch, pValue_arch] = archtest(res_arimagarch, Lags=5);
disp('ARCH test p-value raw Residuals: ');
disp(pValue_arch);

% null hypothesis of no conditional heteroscedasticity
[h_arch, pValue_arch] = archtest(stan_res_arimagarch, Lags=5);
disp('ARCH test p-value standardized Residuals: ');
disp(pValue_arch);


%% Forecast logclose best arima-garch

num_steps = 1; 
forecasted_logclose_ARIMAGARCH = zeros(length(test_logclose),1);
for i=1:length(test_logclose)
    forecasted_logclose_ARIMAGARCH(i) = forecast(fitARIMAGARCH, num_steps, logclose(i:length(train_logclose)+i-1));
end

csvwrite('forecasted_logclose_ARIMA_t_GARCH_t.csv', forecasted_logclose_ARIMAGARCH);


figure;
plot(test_logclose, 'b', 'DisplayName', 'Real Log Close');
hold on;
plot(forecasted_logclose_ARIMAGARCH, 'r', 'DisplayName', 'Forecasted Log Close');
xlabel('Date');
ylabel('Log Close');
legend;
hold off;


error_metrics_arima_garch_logclose = computeErrorMetrics(test_logclose, forecasted_logclose_ARIMAGARCH);
disp('Best ARIMA-t-GARCH-t Forecast Log Close Error Metrics:');
disp(error_metrics_arima_garch_logclose);

disp('Best ARIMA-t-GARCH-t Forecast Log Close Directional Metrics:');
directional_metrics_arima_garch_logclose = computeDirectionalMetrics(test_logclose, forecasted_logclose_ARIMAGARCH);
disp(directional_metrics_arima_garch_logclose);


%% From logclose forecasts to close forecasts
forecasted_close_ARIMAGARCH = exp(forecasted_logclose_ARIMAGARCH);

csvwrite('forecasted_close_ARIMA_t_GARCH_t.csv', forecasted_close_ARIMAGARCH);

%% Evaluate forecast close best arima

figure;
plot(test_close, 'b', 'DisplayName', 'Real Close');
hold on;
plot(forecasted_close_ARIMAGARCH, 'r', 'DisplayName', 'Forecasted Close');
xlabel('Date');
ylabel('Close');
legend;
hold off;


error_metrics_arima_garch_close = computeErrorMetrics(test_close, forecasted_close_ARIMAGARCH);
disp('Best ARIMA-t-GARCH-t Forecast Close Error Metrics:');
disp(error_metrics_arima_garch_close);

disp('Best ARIMA-t-GARCH-t Forecast Close Directional Metrics:');
directional_metrics_arima_garch_close = computeDirectionalMetrics(test_close, forecasted_close_ARIMAGARCH);
disp(directional_metrics_arima_garch_close);


