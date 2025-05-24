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


%% Fit ARIMA for various p,q
max_pq = 5;
results_ARIMA_p1q = ARIMA_p1q(train_logclose, max_pq);

%% Find best ARIMA from all fits, based on aic, p-value, and correlation
best_pqA = find_best_ARIMA(train_logclose,results_ARIMA_p1q);


%% Fit best ARIMA
p_best = best_pqA(1);
q_best = best_pqA(2);
pvalue_best = best_pqA(3);
aic_best = best_pqA(4);
disp(['Best ARIMA model: ARIMA(', num2str(p_best), ',1,', num2str(q_best), ')']);
disp("Best Pvalue:")
disp(pvalue_best)
disp("Best AIC:")
disp(aic_best)

modelARIMA = arima(p_best, 1, q_best);
fitARIMA = estimate(modelARIMA, train_logclose);

[res_arima, var_arima] = infer(fitARIMA, train_logclose);
stan_res_arima = res_arima ./ sqrt(var_arima);

train_MSE = sum(res_arima.^2)/size(res_arima,1);
disp(['Training MSE: ' num2str(train_MSE)]);

csvwrite('rawResiduals_ARIMA.csv', res_arima);
csvwrite('stanResiduals_ARIMA.csv', stan_res_arima);


%% Predictions Training Set
train_logclose_pred = train_logclose - res_arima;

mse = mean((train_logclose - train_logclose_pred).^2);

figure;
plot(train_logclose, 'b', 'DisplayName', 'Real Log Close');
hold on;
plot(train_logclose_pred, 'r', 'DisplayName', 'Log Close Predictions');
title('ARIMA Predictions On Training Data');
xlabel('Time');
ylabel('Value');
legend('Location', 'best');

text(100, 3.2, sprintf('MSE: %.6f', mse), 'FontSize', 10, 'Color', 'k');
hold off;


%% Pspectrum residuals

[pxx2, f2] = pspectrum(res_arima);
geometric_mean2 = exp(mean(log(pxx2)));
arithmetic_mean2 = mean(pxx2);
flatness2 = geometric_mean2 / arithmetic_mean2;
disp(['Flatness raw residuals:', num2str(flatness2)]);


[pxx3, f3] = pspectrum(stan_res_arima);
geometric_mean3 = exp(mean(log(pxx3)));
arithmetic_mean3 = mean(pxx3);
flatness3 = geometric_mean3 / arithmetic_mean3;
disp(['Flatness standardized residuals:', num2str(flatness3)]);


%% Verify correlation residuals
num_lags = ceil(log(length(stan_res_arima))); 

figure;
subplot(2, 1, 1);
autocorr(stan_res_arima, 'NumLags', num_lags);
ylim([-1,1]);
title('');

subplot(2, 1, 2);
parcorr(stan_res_arima, 'NumLags', num_lags);
ylim([-1,1]);
title('');

%null hypothesis of no residual autocorrelation
[h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(stan_res_arima, Lags=num_lags);
disp(['LBQ p-value standardized residuals: ', num2str(pValue_lbq)]);


%% Verify distribution residuals
mu = mean(res_arima);
sigma = std(res_arima);
x = linspace(min(res_arima), max(res_arima), 1000);
normal_pdf = normpdf(x, mu, sigma);

figure;
qqplot(res_arima, makedist('Normal', 'mu', mu, 'sigma', sigma));
xlabel('Theoretical Quantiles (Normal)');
ylabel('Sample Quantiles');
title('');

fitted_t = fitdist(res_arima, 'tlocationscale');
mu_t = fitted_t.mu;        
sigma_t = fitted_t.sigma;     
nu_t = fitted_t.nu;

figure;
qqplot(res_arima, makedist('tLocationScale', 'mu', mu_t, 'sigma', sigma_t, 'nu', nu_t));
xlabel('Theoretical Quantiles (Student''s t)');
ylabel('Sample Quantiles');
title('');


%% Verify variance residuals

stan_res_arima_variance = movvar(stan_res_arima, 100);
figure;
plot(stan_res_arima_variance);
xlabel('Time');
ylabel('Variance');


% null hypothesis of no conditional heteroscedasticity
[h_arch, pValue_arch] = archtest(stan_res_arima, Lags=5);
disp('ARCH test p-value standardizedResiduals: ');
disp(pValue_arch);


%% Forecast logclose best arima
forecasted_logclose_ARIMA = zeros(length(test_logclose),1);
for i=1:length(test_logclose)
    forecasted_logclose_ARIMA(i) = forecast(fitARIMA, 1, logclose(i:length(train_logclose)+i-1));
end

csvwrite('forecasted_logclose_ARIMA.csv', forecasted_logclose_ARIMA);


mse = mean((test_logclose - forecasted_logclose_ARIMA).^2);

figure;
plot(test_logclose, 'b', 'DisplayName', 'Real Log Close');
hold on;
plot(forecasted_logclose_ARIMA, 'r', 'DisplayName', 'Log Close Predictions');
xlabel('Time');
ylabel('Value');
title('ARIMA Forecasts On Test Data');
legend('Location', 'best');
text(65, 3.32, sprintf('MSE: %.6f', mse), 'FontSize', 10, 'Color', 'k');
hold off;

error_metrics_arima_logclose = computeErrorMetrics(test_logclose, forecasted_logclose_ARIMA);
disp('Best ARIMA Forecast Log Close Error Metrics:');
disp(error_metrics_arima_logclose);

disp('Best ARIMA Forecast Log Close Directional Metrics:');
directional_metrics_arima_logclose = computeDirectionalMetrics(test_logclose, forecasted_logclose_ARIMA);
disp(directional_metrics_arima_logclose);



%% From logclose forecasts to close forecasts
forecasted_close_ARIMA = exp(forecasted_logclose_ARIMA);
csvwrite('forecasted_close_ARIMA.csv', forecasted_close_ARIMA);


%% Evaluate forecast close best arima

figure;
plot(test_close, 'b', 'DisplayName', 'Real  Close');
hold on;
plot(forecasted_close_ARIMA, 'r', 'DisplayName', 'Forecasted  Close');
xlabel('Time');
ylabel('Close');
legend;
hold off;

error_metrics_arima_close = computeErrorMetrics(test_close, forecasted_close_ARIMA);
disp('Best ARIMA Forecast Close Error Metrics:');
disp(error_metrics_arima_close);

disp('Best ARIMA Forecast Close Directional Metrics:');
directional_metrics_arima_close = computeDirectionalMetrics(test_close, forecasted_close_ARIMA);
disp(directional_metrics_arima_close);









