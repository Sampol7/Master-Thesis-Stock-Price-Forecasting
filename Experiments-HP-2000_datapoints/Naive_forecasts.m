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


%% Naive Forecasting logclose
forecasted_logclose_naive = zeros(length(test_logclose),1);
for i = 1:length(test_logclose)
    if i == 1
        forecasted_logclose_naive(i) = train_logclose(end);
    else
        forecasted_logclose_naive(i) = test_logclose(i-1); 
    end
end


figure;
plot(test_logclose, 'b', 'DisplayName', 'Real Log Close');
hold on;
plot(forecasted_logclose_naive, 'r', 'DisplayName', 'Naïve Forecasted Log Close');
xlabel('Date');
ylabel('Log Close');
legend;
hold off;


error_metrics_naive_logclose = computeErrorMetrics(test_logclose, forecasted_logclose_naive);
disp('Naïve Forecast Log Close Error Metrics:');
disp(error_metrics_naive_logclose);

disp('Naïve Forecast Log Close Directional Metrics:');
directional_metrics_naive_logclose = computeDirectionalMetrics(test_logclose, forecasted_logclose_naive);
disp(directional_metrics_naive_logclose);


%% Naive Forecasting close
forecasted_close_naive = zeros(length(test_close),1);
for i = 1:length(test_close)
    if i == 1
        forecasted_close_naive(i) = train_close(end);
    else
        forecasted_close_naive(i) = test_close(i-1); 
    end
end


figure;
plot(test_close, 'b', 'DisplayName', 'Real  Close');
hold on;
plot(forecasted_close_naive, 'r', 'DisplayName', 'Naïve Forecasted  Close');
xlabel('Date');
ylabel('Close');
legend;
hold off;


error_metrics_naive_close = computeErrorMetrics(test_close, forecasted_close_naive);
disp('Naïve Forecast Close Error Metrics:');
disp(error_metrics_naive_close);

disp('Naïve Forecast Close Directional Metrics:');
directional_metrics_naive_close = computeDirectionalMetrics(test_close, forecasted_close_naive);
disp(directional_metrics_naive_close);

