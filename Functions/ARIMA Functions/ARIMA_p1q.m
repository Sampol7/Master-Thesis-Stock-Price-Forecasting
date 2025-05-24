%Function to calculate maximal p-value and AIC for all ARIMA(p_A,1,q_A) models in (p_A,q_A) \in [0,max_pq] x [0,max_pq] grid.

function results = ARIMA_p1q(time_series, max_pq)
    results = []; % Store p, q, maxPValue, AIC
    
    for p = 0:max_pq
        for q = 0:max_pq
            try
                model = arima(p, 1, q);
                fitted_model = estimate(model, time_series);
                summary = summarize(fitted_model)
                Ptable = summary.Table;
                Pvalues = Ptable.PValue(2:end-1);
                maxPValue = max(Pvalues); 
                
                if p == 0 && q == 0 
                    maxPValue = 0;
                end
                logLikelihood = summary.LogLikelihood;
                aic = summary.AIC;
                bic = summary.BIC;
                
                results = [results; p, q, maxPValue, aic];
    
            catch ME
                disp(['Error for ARMA(' num2str(p) ',' num2str(q) '): ' ME.message]);
            end
        end
    end
end




