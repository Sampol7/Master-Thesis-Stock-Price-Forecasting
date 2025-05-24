% Function that selects most promising ARIMA-GARCH model based on maximal p-value, AIC,
% and correlation in residuals

function best_pqAG = find_best_ARIMA_GARCH(train_logclose, results_ARIMA_GARCH_p1q)
tolerances = [0.05, 0.1, 0.2, 0.5, 1.0, -1.0];

for tolerance = tolerances
    if tolerance == -1.0
        error("No ARIMA-GARCH model found for which residuals are uncorrelated.");
    end

    filtered_results = filter_by_pvalueAG(results_ARIMA_GARCH_p1q, tolerance);

    for i = 1:size(filtered_results, 1)
        best_pqAG = select_nth_best_aicAG(filtered_results,i);
        pA_best = best_pqAG(1);
        qA_best = best_pqAG(2);
        pG_best = best_pqAG(3);
        qG_best = best_pqAG(4);
        
        modelARIMAGARCH = arima(pA_best, 1, qA_best);
        modelGARCH = garch(pG_best, qG_best);
        modelARIMAGARCH.Variance = modelGARCH;
        fitARIMAGARCH = estimate(modelARIMAGARCH, train_logclose, "Display","off");
        
        [res_arimagarch, var_arimagarch] = infer(fitARIMAGARCH, train_logclose);
        stan_res_arimagarch = res_arimagarch ./ sqrt(var_arimagarch);
        
        
        num_lags = ceil(log(length(stan_res_arimagarch)));
        
        %null hypothesis of no residual autocorrelation
        [h_lbq,pValue_lbq,stat_lbq,cValue_lbq] = lbqtest(stan_res_arimagarch, Lags=num_lags)
        
        if h_lbq == 0
            disp(['First ARIMA-GARCH model without correlation in residuals was found at position ', num2str(i), ...
            ' in the list of tolerance ', num2str(tolerance), '.']);
            break; 
        end
    end

    if h_lbq == 0
        break; 
    end
    
end
