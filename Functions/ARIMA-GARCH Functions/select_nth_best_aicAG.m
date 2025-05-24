% Function that selects fit with n-th lowest AIC from a list of fits for which the largest p-value is smaller than a certain threshold

function best_pqAG = select_nth_best_aicAG(filtered_results, n)
    if isempty(filtered_results)
        best_pqAG = [];
    elseif n > size(filtered_results, 1)
        error('n is larger than the number of available (p,q) pairs.');
    else
        sorted_results = sortrows(filtered_results, 6); 
        
        best_pqAG = sorted_results(n, :);
    end
end


