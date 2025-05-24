library(tseriesEntropy)
library(nonlinearTseries)
library(rstudioapi)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))


# Function to run nonlinearity test multiple times 
#and compute improved Bonferroni bound on Whites results
BB_White <- function(data, num_runs = 5) {
  p_values <- numeric(num_runs)  
  
  for (i in 1:num_runs) {
    test_result <- nonlinearityTest(data, verbose = FALSE) 
    p_values[i] <- test_result$White$p.value 
  }
  
  p_values <- sort(p_values) 
  
  m <- length(p_values)
  alpha <- min((m - (1:m) + 1) * p_values)  
  
  results = list(sorted_p_values = p_values, improved_bonferroni_bound = alpha)
  
  print(results)
  
  return(results)
}




train_close <- read.csv("train_close.csv", header = FALSE)
train_close <- train_close$V1
NL_Test_train_close <- nonlinearityTest(train_close, verbose = TRUE)
BB_White_train_close <- BB_White(train_close, num_runs = 5)

train_logclose <- read.csv("train_logclose.csv", header = FALSE)
train_logclose <- train_logclose$V1
NL_Test_train_logclose <- nonlinearityTest(train_logclose, verbose = TRUE)
BB_White_train_logclose <- BB_White(train_logclose, num_runs = 5)

train_logreturns <- read.csv("train_logreturns.csv", header = FALSE)
train_logreturns <- train_logreturns$V1
NL_Test_train_logreturns <- nonlinearityTest(train_logreturns, verbose = TRUE)
BB_White_train_logreturns <- BB_White(train_logreturns, num_runs = 5)


rawResiduals_ARIMA <- read.csv("rawResiduals_ARIMA.csv", header = FALSE)
rawResiduals_ARIMA <- rawResiduals_ARIMA$V1
NL_Test_rawResiduals_ARIMA <- nonlinearityTest(rawResiduals_ARIMA, verbose = TRUE)
BB_White_rawResiduals_ARIMA <- BB_White(rawResiduals_ARIMA, num_runs = 5)

stanResiduals_ARIMA <- read.csv("stanResiduals_ARIMA.csv", header = FALSE)
stanResiduals_ARIMA <- stanResiduals_ARIMA$V1
NL_Test_stanResiduals_ARIMA <- nonlinearityTest(stanResiduals_ARIMA, verbose = TRUE)
BB_White_stanResiduals_ARIMA <- BB_White(stanResiduals_ARIMA, num_runs = 5)


rawResiduals_ARIMA_t <- read.csv("rawResiduals_ARIMA_t.csv", header = FALSE)
rawResiduals_ARIMA_t <- rawResiduals_ARIMA_t$V1
NL_Test_rawResiduals_ARIMA_t <- nonlinearityTest(rawResiduals_ARIMA_t, verbose = TRUE)
BB_White_rawResiduals_ARIMA_t <- BB_White(rawResiduals_ARIMA_t, num_runs = 5)

stanResiduals_ARIMA_t <- read.csv("stanResiduals_ARIMA_t.csv", header = FALSE)
stanResiduals_ARIMA_t <- stanResiduals_ARIMA_t$V1
NL_Test_stanResiduals_ARIMA <- nonlinearityTest(stanResiduals_ARIMA_t, verbose = TRUE)
BB_White_stanResiduals_ARIMA_t <- BB_White(stanResiduals_ARIMA_t, num_runs = 5)


rawResiduals_ARIMA_GARCH <- read.csv("rawResiduals_ARIMA_GARCH.csv", header = FALSE)
rawResiduals_ARIMA_GARCH <- rawResiduals_ARIMA_GARCH$V1
NL_Test_rawResiduals_ARIMA_GARCH <- nonlinearityTest(rawResiduals_ARIMA_GARCH, verbose = TRUE)
BB_White_rawResiduals_ARIMA_GARCH <- BB_White(rawResiduals_ARIMA_GARCH, num_runs = 5)

stanResiduals_ARIMA_GARCH <- read.csv("stanResiduals_ARIMA_GARCH.csv", header = FALSE)
stanResiduals_ARIMA_GARCH <- stanResiduals_ARIMA_GARCH$V1
NL_Test_stanResiduals_ARIMA_GARCH <- nonlinearityTest(stanResiduals_ARIMA_GARCH, verbose = TRUE)
BB_White_stanResiduals_ARIMA_GARCH <- BB_White(stanResiduals_ARIMA_GARCH, num_runs = 5)


rawResiduals_ARIMA_t_GARCH_t <- read.csv("rawResiduals_ARIMA_t_GARCH_t.csv", header = FALSE)
rawResiduals_ARIMA_t_GARCH_t <- rawResiduals_ARIMA_t_GARCH_t$V1
NL_Test_rawResiduals_ARIMA_t_GARCH_t <- nonlinearityTest(rawResiduals_ARIMA_t_GARCH_t, verbose = TRUE)
BB_White_rawResiduals_ARIMA_t_GARCH_t <- BB_White(rawResiduals_ARIMA_t_GARCH_t, num_runs = 5)

stanResiduals_ARIMA_t_GARCH_t <- read.csv("stanResiduals_ARIMA_t_GARCH_t.csv", header = FALSE)
stanResiduals_ARIMA_t_GARCH_t <- stanResiduals_ARIMA_t_GARCH_t$V1
NL_Test_stanResiduals_ARIMA_GARCH <- nonlinearityTest(stanResiduals_ARIMA_t_GARCH_t, verbose = TRUE)
BB_White_stanResiduals_ARIMA_t_GARCH_t <- BB_White(stanResiduals_ARIMA_t_GARCH_t, num_runs = 5)



rawResiduals_SVR_BO_standardscaler <- read.csv("rawResiduals_SVR_BO-TPE_standardscaler.csv", header = FALSE)
rawResiduals_SVR_BO_standardscaler <- rawResiduals_SVR_BO_standardscaler$V1
NL_Test_rawResiduals_SVR_BO_standardscaler <- nonlinearityTest(rawResiduals_SVR_BO_standardscaler, verbose = TRUE)
BB_White_rawResiduals_SVR_BO_standardscaler <- BB_White(rawResiduals_SVR_BO_standardscaler, num_runs = 5)

rawResiduals_SVR_BO_robustscaler <- read.csv("rawResiduals_SVR_BO-TPE_robustscaler.csv", header = FALSE)
rawResiduals_SVR_BO_robustscaler <- rawResiduals_SVR_BO_robustscaler$V1
NL_Test_rawResiduals_SVR_BO_robustscaler <- nonlinearityTest(rawResiduals_SVR_BO_robustscaler, verbose = TRUE)
BB_White_rawResiduals_SVR_BO_robustscaler <- BB_White(rawResiduals_SVR_BO_robustscaler, num_runs = 5)



rawResiduals_ARIMA_t_SVR_BO_standardscaler <- read.csv("rawResiduals_ARIMA_t_SVR_BO-TPE_standardscaler.csv", header = FALSE)
rawResiduals_ARIMA_t_SVR_BO_standardscaler <- rawResiduals_ARIMA_t_SVR_BO_standardscaler$V1
NL_Test_rawResiduals_ARIMA_t_SVR_BO_standardscaler <- nonlinearityTest(rawResiduals_ARIMA_t_SVR_BO_standardscaler, verbose = TRUE)
BB_White_rawResiduals_ARIMA_t_SVR_BO_standardscaler <- BB_White(rawResiduals_ARIMA_t_SVR_BO_standardscaler, num_runs = 5)

rawResiduals_ARIMA_t_SVR_BO_robustscaler <- read.csv("rawResiduals_ARIMA_t_SVR_BO-TPE_robustscaler.csv", header = FALSE)
rawResiduals_ARIMA_t_SVR_BO_robustscaler <- rawResiduals_ARIMA_t_SVR_BO_robustscaler$V1
NL_Test_rawResiduals_ARIMA_t_SVR_BO_robustscaler <- nonlinearityTest(rawResiduals_ARIMA_t_SVR_BO_robustscaler, verbose = TRUE)
BB_White_rawResiduals_ARIMA_t_SVR_BO_robustscaler <- BB_White(rawResiduals_ARIMA_t_SVR_BO_robustscaler, num_runs = 5)



rawResiduals_FNN_BO_standardscaler <- read.csv("rawResiduals_FNN_BO-TPE_standardscaler.csv", header = FALSE)
rawResiduals_FNN_BO_standardscaler <- rawResiduals_FNN_BO_standardscaler$V1
NL_Test_rawResiduals_FNN_BO_standardscaler <- nonlinearityTest(rawResiduals_FNN_BO_standardscaler, verbose = TRUE)
BB_White_rawResiduals_FNN_BO_standardscaler <- BB_White(rawResiduals_FNN_BO_standardscaler, num_runs = 5)

rawResiduals_FNN_BO_robustscaler <- read.csv("rawResiduals_FNN_BO-TPE_robustscaler.csv", header = FALSE)
rawResiduals_FNN_BO_robustscaler <- rawResiduals_FNN_BO_robustscaler$V1
NL_Test_rawResiduals_FNN_BO_robustscaler <- nonlinearityTest(rawResiduals_FNN_BO_robustscaler, verbose = TRUE)
BB_White_rawResiduals_FNN_BO_robustscaler <- BB_White(rawResiduals_FNN_BO_robustscaler, num_runs = 5)


rawResiduals_LSTM_BO_standardscaler <- read.csv("rawResiduals_LSTM_BO-TPE_standardscaler.csv", header = FALSE)
rawResiduals_LSTM_BO_standardscaler <- rawResiduals_LSTM_BO_standardscaler$V1
NL_Test_rawResiduals_LSTM_BO_standardscaler <- nonlinearityTest(rawResiduals_LSTM_BO_standardscaler, verbose = TRUE)
BB_White_rawResiduals_LSTM_BO_standardscaler <- BB_White(rawResiduals_LSTM_BO_standardscaler, num_runs = 5)

rawResiduals_LSTM_BO_robustscaler <- read.csv("rawResiduals_LSTM_BO-TPE_robustscaler.csv", header = FALSE)
rawResiduals_LSTM_BO_robustscaler <- rawResiduals_LSTM_BO_robustscaler$V1
NL_Test_rawResiduals_LSTM_BO_robustscaler <- nonlinearityTest(rawResiduals_LSTM_BO_robustscaler, verbose = TRUE)
BB_White_rawResiduals_LSTM_BO_robustscaler <- BB_White(rawResiduals_LSTM_BO_robustscaler, num_runs = 5)



rawResiduals_ARIMA_t_FNN_BO_standardscaler <- read.csv("rawResiduals_ARIMA_t_FNN_BO-TPE_standardscaler.csv", header = FALSE)
rawResiduals_ARIMA_t_FNN_BO_standardscaler <- rawResiduals_ARIMA_t_FNN_BO_standardscaler$V1
NL_Test_rawResiduals_ARIMA_t_FNN_BO_standardscaler <- nonlinearityTest(rawResiduals_ARIMA_t_FNN_BO_standardscaler, verbose = TRUE)
BB_White_rawResiduals_ARIMA_t_FNN_BO_standardscaler <- BB_White(rawResiduals_ARIMA_t_FNN_BO_standardscaler, num_runs = 5)

rawResiduals_ARIMA_t_FNN_BO_robustscaler <- read.csv("rawResiduals_ARIMA_t_FNN_BO-TPE_robustscaler.csv", header = FALSE)
rawResiduals_ARIMA_t_FNN_BO_robustscaler <- rawResiduals_ARIMA_t_FNN_BO_robustscaler$V1
NL_Test_rawResiduals_ARIMA_t_FNN_BO_robustscaler <- nonlinearityTest(rawResiduals_ARIMA_t_FNN_BO_robustscaler, verbose = TRUE)
BB_White_rawResiduals_ARIMA_t_FNN_BO_robustscaler <- BB_White(rawResiduals_ARIMA_t_FNN_BO_robustscaler, num_runs = 5)


rawResiduals_ARIMA_t_LSTM_BO_standardscaler <- read.csv("rawResiduals_ARIMA_t_LSTM_BO-TPE_standardscaler.csv", header = FALSE)
rawResiduals_ARIMA_t_LSTM_BO_standardscaler <- rawResiduals_ARIMA_t_LSTM_BO_standardscaler$V1
NL_Test_rawResiduals_ARIMA_t_LSTM_BO_standardscaler <- nonlinearityTest(rawResiduals_ARIMA_t_LSTM_BO_standardscaler, verbose = TRUE)
BB_White_rawResiduals_ARIMA_t_LSTM_BO_standardscaler <- BB_White(rawResiduals_ARIMA_t_LSTM_BO_standardscaler, num_runs = 5)

rawResiduals_ARIMA_t_LSTM_BO_robustscaler <- read.csv("rawResiduals_ARIMA_t_LSTM_BO-TPE_robustscaler.csv", header = FALSE)
rawResiduals_ARIMA_t_LSTM_BO_robustscaler <- rawResiduals_ARIMA_t_LSTM_BO_robustscaler$V1
NL_Test_rawResiduals_ARIMA_t_LSTM_BO_robustscaler <- nonlinearityTest(rawResiduals_ARIMA_t_LSTM_BO_robustscaler, verbose = TRUE)
BB_White_rawResiduals_ARIMA_t_LSTM_BO_robustscaler <- BB_White(rawResiduals_ARIMA_t_LSTM_BO_robustscaler, num_runs = 5)









