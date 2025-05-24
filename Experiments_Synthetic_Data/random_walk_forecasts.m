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


%% TheilsU and DA

n_runs = 1000;
T = length(test_close);
sigmas_RW = logspace(-2, 0, 6);

all_errors = zeros(length(sigmas_RW), n_runs, 6);
all_dirs = zeros(length(sigmas_RW), n_runs, 4);

for s = 1:length(sigmas_RW)
    sigma_RW = sigmas_RW(s);
    
    for run = 1:n_runs
        forecasted_close_rw = zeros(T, 1);
        
        for t = 1:T
            if t == 1
                prev_close = train_close(end);
            else
                prev_close = test_close(t - 1);
            end
            forecasted_close_rw(t) = prev_close + sigma_RW * randn();
        end
        
        all_errors(s, run, :) = struct2array(computeErrorMetrics(test_close, forecasted_close_rw));
        all_dirs(s, run, :) = struct2array(computeDirectionalMetrics(test_close, forecasted_close_rw));
    end
    
    mean_errors = mean(all_errors(s,:,:), 2);
    ci_errors = 1.96 * std(all_errors(s,:,:), 0, 2) / sqrt(n_runs);
    
    mean_dirs = mean(all_dirs(s,:,:), 2);
    ci_dirs = 1.96 * std(all_dirs(s,:,:), 0, 2) / sqrt(n_runs);
    

    disp(['Sigma = ', num2str(sigma_RW, '%.4g')]);
    
    disp('TheilsU');
    disp(['mean = ', num2str(mean_errors(6), '%.4g')]);
    disp(['CI = ', num2str(ci_errors(6), '%.4g')]);
    disp(' ');
    
    disp('DA');
    disp(['mean = ', num2str(mean_dirs(1), '%.4g')]);
    disp(['CI = ', num2str(ci_dirs(1), '%.4g')]);
    disp(' ');
end



%% NOR 

n_group_members = 100;
n_runs = 1000;
T = length(test_close);
sigmas_RW = logspace(-2, 0, 6);

all_NOR = zeros(length(sigmas_RW), n_runs);

for s = 1:length(sigmas_RW)
    sigma_RW = sigmas_RW(s);
    
    for run = 1:n_runs
        theilsU_vals = zeros(n_group_members, 1);
        
        for member = 1:n_group_members
            forecasted_close_rw = zeros(T, 1);
            for t = 1:T
                if t == 1
                    prev_close = train_close(end);
                else
                    prev_close = test_close(t-1);
                end
                forecasted_close_rw(t) = prev_close + sigma_RW * randn();
            end
            
            err_metrics = computeErrorMetrics(test_close, forecasted_close_rw);
            theilsU_vals(member) = err_metrics.theilsU;
        end
        
        all_NOR(s, run) = sum(theilsU_vals < 1) / n_group_members;
    end
    
    mean_NOR = mean(all_NOR(s, :));
    ci_NOR = 1.96 * std(all_NOR(s, :), 0, 2) / sqrt(n_runs);

    disp(['Sigma = ', num2str(sigma_RW, '%.4g')]);
    disp(['mean_NOR = ', num2str(mean_NOR, '%.4g')]);
    disp(['ci_NOR = ', num2str(ci_NOR, '%.4g')]);
    disp(' ');
end

