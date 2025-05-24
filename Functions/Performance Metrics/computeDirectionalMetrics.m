% Function to calculate the several directional metrics

function directional_metrics = computeDirectionalMetrics(y_true, y_pred)

    actual_diff = diff(y_true);
    predicted_diff = diff(y_pred);
    predicted_direction = y_pred(2:end) - y_true(1:end-1);
    

    N = length(actual_diff);

    d_t_DA = (actual_diff .* predicted_direction) > 0;
    DA = (1/N) * sum(d_t_DA);

    d_t_DS = (actual_diff .* predicted_diff) > 0;
    DS = (1/N) * sum(d_t_DS);
    
    d_t_CU = (predicted_diff > 0) & (actual_diff .* predicted_diff > 0);
    k_t_CU = (actual_diff) > 0;
    if sum(k_t_CU) == 0
        CU = NaN;
    else
        CU = sum(d_t_CU) / sum(k_t_CU);
    end
    
    d_t_CD = (predicted_diff < 0) & (actual_diff .* predicted_diff > 0);
    k_t_CD = (actual_diff) < 0;
    if sum(k_t_CD) == 0
        CD = NaN;
    else
        CD = sum(d_t_CD) / sum(k_t_CD);
    end

    directional_metrics = struct('DA', DA, 'DS', DS, 'CU', CU, 'CD', CD);

end

