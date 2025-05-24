% Confidence intervals of spectral flatness baselines for white noise

rng(42);

num_trials = 1000;       
lengths = [10, 100, 900, 1900, 3900, 7900];

for i = 1:length(lengths)
    length_i = lengths(i);
    flatness_values = zeros(num_trials, 1);

    for k = 1:num_trials
        x = randn(length_i, 1);             
        [pxx, ~] = pspectrum(x);              
        g_mean = exp(mean(log(pxx)));           
        a_mean = mean(pxx);                    
        flatness_values(k) = g_mean / a_mean;   
    end

    disp(['length: ', num2str(length_i), ', Mean : ', num2str(mean(flatness_values))]);

    std_flatness = std(flatness_values);
    ci_flatness = 1.96 * std_flatness / sqrt(num_trials);
    disp(['length: ', num2str(length_i), ', CI: ', num2str(ci_flatness)]);

end

