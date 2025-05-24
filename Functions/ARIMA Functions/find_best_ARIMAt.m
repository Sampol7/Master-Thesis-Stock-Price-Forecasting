% Function that selects most promising ARIMA-t model based on maximal p-value, AIC,
% and correlation in residuals
 
function best_pqA = find_best_ARIMAt(train_logclose, results_ARIMAt_p1q)
tolerances = [0.05, 0.1, 0.2, 0.5, 1.0, -1.0];

for tolerance = tolerances
    if tolerance == -1.0
        disp("No ARIMA model found for which residuals are uncorrelated.");
        filtered_results = filter_by_pvalueA(results_ARIMAt_p1q, 0.05);
        best_pqA = select_nth_best_aicA(filtered_results,1);
    end

    filtered_results = filter_by_pvalueA(results_ARIMAt_p1q, tolerance);

    
    for i = 1:size(filtered_results, 1)
        best_pqA = select_nth_best_aicA(filtered_results,i);
        p_best = best_pqA(1);
        q_best = best_pqA(2);
        
        modelARIMA = arima(p_best, 1, q_best);
        modelARIMA.Distribution = 't';
        fitARIMA = estimate(modelARIMA, train_logclose, "Display","off");
        
        [res_arima, var_arima] = infer(fitARIMA, train_logclose);
        stan_res_arima = res_arima ./ sqrt(var_arima);
        
        
        num_lags = ceil(log(length(stan_res_arima))); 
        
        %null hypothesis of no residual autocorrelation
        [h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(stan_res_arima, Lags=num_lags);
        
        if h_lbq == 0
            disp(['First ARIMA model without correlation in residuals was found at position ', num2str(i), ...
            ' in the list of tolerance ', num2str(tolerance), '.']);
            break; 
        end
    end

    if h_lbq == 0
        break; 
    end
    
end
