%Function to remove all fits for which maximal p-value is larger than tolerance

function filtered_results = filter_by_pvalueAG(results, tolerance)
    filtered_results = results(results(:, 5) < tolerance, :);
end

