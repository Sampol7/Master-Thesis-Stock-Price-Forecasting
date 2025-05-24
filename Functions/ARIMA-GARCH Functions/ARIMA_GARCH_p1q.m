%Function to calculate maximal p-value and AIC for all ARIMA(p_A,1,q_A)-GARCH(p_G,q_G) models in (p_A,q_A,p_G,q_G) \in [0,max_pqA] x [0,max_pqA] x [0,max_pqG] x [0,max_pqG] grid.

function results = ARIMA_GARCH_p1q(time_series, max_pqA, max_pqG)
    results = []; % Store pA, qA, pG, qG, maxPValue, aic
    
    for pA = 0:max_pqA
        for qA = 0:max_pqA
            for pG = 0:max_pqG
                for qG = 0:max_pqG
                    try
                        modelARIMAGARCH = arima(pA, 1, qA);
                        modelGARCH = garch(pG, qG);
                        modelARIMAGARCH.Variance = modelGARCH;
    
                        fitted_model = estimate(modelARIMAGARCH, time_series);
                        summary = summarize(fitted_model);
                        table = summary.Table;
                        Vtable = summary.VarianceTable;
                        
                        Pvalues = table.PValue(2:end);
                        VPvalues = Vtable.PValue(2:end);
                        maxPValue = max([Pvalues;VPvalues]); 
                        if pA == 0 && qA == 0 && pG == 0 && qG == 0
                            maxPValue = 0;
                        end
                        logLikelihood = summary.LogLikelihood;
                        aic = summary.AIC;
                        bic = summary.BIC;
                        
                        results = [results; pA, qA, pG, qG, maxPValue, aic];
            
                    catch ME
                        disp(['Error for ARMA(' num2str(pA) ',' num2str(qA) ')-GARCH(' num2str(pG) ',' num2str(qG) '): ' ME.message]);
                    end
                end
            end
        end
    end
end


