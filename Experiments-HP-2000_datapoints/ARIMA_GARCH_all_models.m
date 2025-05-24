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



%% Fit ARIMA-GARCH models, forecast, and compute error metrics
max_pqA = 5;
max_pqG = 2;
results = []; 

for pA = 0:max_pqA
    for qA = 0:max_pqA
        for pG = 0:max_pqG
            for qG = 0:max_pqG
                try
                    modelARIMAGARCH = arima(pA, 1, qA);
                    modelGARCH = garch(pG, qG);
                    modelARIMAGARCH.Variance = modelGARCH;
                    fitted_model = estimate(modelARIMAGARCH, train_logclose);
        
                    num_steps = 1;
                    forecasted_logclose = zeros(length(test_logclose),1);
        
                    for i = 1:length(test_logclose)
                        forecasted_logclose(i) = forecast(fitted_model, num_steps, logclose(i:length(train_logclose) + i - 1));
                    end
        
                    forecasted_close = exp(forecasted_logclose);
        
                    error_metrics = computeErrorMetrics(test_close, forecasted_close);
                    directional_metrics = computeDirectionalMetrics(test_close, forecasted_close);
        
                    results = [results;
                        pA, qA, pG, qG, ...
                        error_metrics.NMSE, ...
                        error_metrics.MAE, ...
                        error_metrics.RMSE, ...
                        error_metrics.MSE, ...
                        error_metrics.MAPE, ...
                        error_metrics.theilsU, ...
                        directional_metrics.DA, ...
                        directional_metrics.DS, ...
                        directional_metrics.CU, ...
                        directional_metrics.CD];
                catch ME
                    disp(['Error for ARIMA-GARCH(' num2str(pA) ',' num2str(qA) ...
                          ')-(' num2str(pG) ',' num2str(qG) '): ' ME.message]);
                end
            end
        end
    end
end

format shortG

format shortG
disp('Results (pA, qA, pG, qG, NMSE, MAE, RMSE, MSE, MAPE, theilsU, DA, DS, CU, CD):');
disp(results);



%% Mean and standard deviation

metrics = results(:, 5:14);

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



%% Count worse models

countWorseModels_ARIMAGARCH(results, 0, 0, 1, 1)

function worseRelative = countWorseModels_ARIMAGARCH(results, pA, qA, pG, qG)
    idx = find(results(:,1)==pA & results(:,2)==qA & results(:,3)==pG & results(:,4)==qG, 1);
    
    if isempty(idx)
        error('Input (pA,qA,pG,qG) pair not found.');
    end
    
    target = results(idx, 5:end);
    
    worseCounts = zeros(1, length(target));
    
    for i = 1:size(results, 1)
        row = results(i, 5:end);
        for j = 1:length(target)
            if j <= 6  % Lower is better for NMSE, MAE, RMSE, MSE, MAPE, TheilsU
                if row(j) > target(j)
                    worseCounts(j) = worseCounts(j) + 1;
                end
            else  % Higher is better for DA, DS, CU, CD 
                if row(j) < target(j)
                    worseCounts(j) = worseCounts(j) + 1;
                end
            end
        end
    end

    worseRelative = worseCounts / size(results,1);
end

