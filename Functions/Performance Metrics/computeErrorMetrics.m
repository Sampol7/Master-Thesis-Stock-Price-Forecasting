% Function to calculate several forecasting metrics

function error_metrics = computeErrorMetrics(y_true, y_pred)
    n = length(y_true);

    y_true = y_true(:);
    y_pred = y_pred(:);
    errors = y_true - y_pred;

    NMSE = sum((errors).^2) / sum((y_true - mean(y_true)).^2);
    MAE = mean(abs(errors));
    RMSE = sqrt(mean(errors .^ 2));
    MSE = mean(errors .^ 2);
    MAPE = mean(abs((errors) ./ y_true)) * 100;

    U_numerator = sum(((y_pred(2:n) - y_true(2:n)) ./ y_true(1:n-1)).^2);
    U_denominator = sum(((y_true(2:n) - y_true(1:n-1)) ./ y_true(1:n-1)).^2);
    theilsU = sqrt(U_numerator / U_denominator);

    error_metrics = struct('NMSE', NMSE, 'MAE', MAE, 'RMSE', RMSE, 'MSE', MSE, 'MAPE', MAPE, 'theilsU', theilsU);
end
