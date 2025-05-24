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



%% Fit ARIMA(p,1,q) models, forecast, and compute error metrics
max_pq = 5;
results = []; 

for p = 0:max_pq
    for q = 0:max_pq
        try
            model = arima(p, 1, q);
            model.Distribution = 't';
            fitted_model = estimate(model, train_logclose);

            num_steps = 1;
            forecasted_logclose_ARIMA = zeros(length(test_logclose),1);

            for i = 1:length(test_logclose)
                forecasted_logclose_ARIMA(i) = forecast(fitted_model, num_steps, logclose(i:length(train_logclose) + i - 1));
            end

            forecasted_close_ARIMA = exp(forecasted_logclose_ARIMA);


            error_metrics_arima_logclose = computeErrorMetrics(test_close, forecasted_close_ARIMA);
            directional_metrics_arima_logclose = computeDirectionalMetrics(test_close, forecasted_close_ARIMA);

            results = [results; p, q,   error_metrics_arima_logclose.NMSE,      ...
                                        error_metrics_arima_logclose.MAE,       ...
                                        error_metrics_arima_logclose.RMSE,      ...
                                        error_metrics_arima_logclose.MSE,       ...
                                        error_metrics_arima_logclose.MAPE,      ...
                                        error_metrics_arima_logclose.theilsU,   ... 
                                        directional_metrics_arima_logclose.DA,  ...
                                        directional_metrics_arima_logclose.DS,  ...
                                        directional_metrics_arima_logclose.CU,  ...
                                        directional_metrics_arima_logclose.CD];

        catch ME
            disp(['Error for ARIMA(' num2str(p) ',' num2str(q) '): ' ME.message]);
        end
    end
end

format shortG

disp('Results (p, q, NMSE, MAE, RMSE, MSE, MAPE, theilsU, DA, DS, CU, CD):');
disp(results);


%% Mean and standard deviation

metrics = results(:, 3:12);

min_values = min(metrics);
disp('Minimum (p, q, NMSE, MAE, RMSE, MSE, MAPE, theilsU, DA, DS, CU, CD):');
disp(min_values);

mean_values = mean(metrics);
disp('Mean (p, q, NMSE, MAE, RMSE, MSE, MAPE, theilsU, DA, DS, CU, CD):');
disp(mean_values);

CI_values = 1.96 * std(metrics) / sqrt(size(metrics, 1));
disp('CI (p, q, NMSE, MAE, RMSE, MSE, MAPE, theilsU, DA, DS, CU, CD):');
disp(CI_values);

max_values = max(metrics);
disp('Maximum (p, q, NMSE, MAE, RMSE, MSE, MAPE, theilsU, DA, DS, CU, CD):');
disp(max_values);


%% Histograms
figure;

titles = {'NMSE', 'MAE', 'RMSE', 'MSE', 'MAPE', 'theilsU', 'DA', 'DS', 'CU', 'CD'};

for i = 1:9
    subplot(3,3,i); 
    histogram(metrics(:,i), 10);
    title(titles{i}); 
end


%% Count worse models

countWorseModels_ARIMA(results, 0, 1)

function worseRelative = countWorseModels_ARIMA(results, p, q)
    idx = find(results(:,1) == p & results(:,2) == q, 1);

    target = results(idx, 3:end);

    worseCounts = zeros(1, length(target));

    for i = 1:size(results, 1)
        row = results(i, 3:end);
        for j = 1:length(target)
            if j <= 6  % Lower is better: NMSE, MAE, RMSE, MSE, MAPE, TheilsU
                if row(j) > target(j)
                    worseCounts(j) = worseCounts(j) + 1;
                end
            else  % Higher is better: DA, DS, CU, CD
                if row(j) < target(j)
                    worseCounts(j) = worseCounts(j) + 1;
                end
            end
        end
    end

    worseRelative = worseCounts / size(results,1);
end


