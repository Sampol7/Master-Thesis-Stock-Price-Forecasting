# Function that simulates simple trading strategy based on the code from:
# https://github.com/Saswato/Stock-Price-Prediction-and-Trading-Strategy/tree/main

import matplotlib.pyplot as plt

def trading_strategy1(actual, predicted):
    
    signal = 0
    amount = actual[0]*1000
    Amount = []
    balance = 0
    action = []
    portfolio = 0
    Portfolio = []
    stocks = 0
    Stocks = []

    for i in range(len(actual)-1):
        if predicted[i+1] > actual[i]:
            if signal == 0:
                action.append('Buy')
                stocks = int(amount / actual[i])
                balance = int(amount % actual[i])
                portfolio = stocks * actual[i]
                signal = 1
                amount = portfolio + balance
                print('Stock:',actual[i] ,'Action:',action[-1],'Portfolio:',round(portfolio,2),'Stocks:', stocks,'Balance_init:',balance,'total($)',round(amount,2))
                Portfolio.append(round(portfolio,5))
                Amount.append(round(amount,0))
                Stocks.append(stocks)
            else:
                action.append('Bought--Holding')
                portfolio = stocks * actual[i]
                amount = portfolio + balance
                print('Stock:',actual[i],'Action:',action[-1],'Portfolio:',round(portfolio,2),'Stocks:', stocks,'Balance_init:',balance,'total($)',round(amount,2))
                Portfolio.append(round(portfolio,5))
                Amount.append(round(amount,0))
                Stocks.append(stocks)
                
        elif predicted[i+1] < actual[i]:
            if signal == 1:
                action.append('Sell')
                portfolio = stocks * actual[i]
                
                signal = 0
                stocks = 0
                amount = balance + portfolio
                portfolio = 0
                balance = 0
                print('Stock:',actual[i],'Action:',action[-1],'Portfolio:',round(portfolio,2),'Stocks:', stocks,'Balance_init:',balance,'total($)',round(amount,2))
                Portfolio.append(round(portfolio,5))
                Amount.append(round(amount,0))
                Stocks.append(stocks)
            else:
                action.append('Price-Prediction-Already-Lower')
                print('Stock:',actual[i],'Action:',action[-1],'Portfolio:',round(portfolio,2),'Stocks:', stocks,'Balance_init:',balance,'total($)',round(amount,2))
                Portfolio.append(round(portfolio,5))
                Amount.append(round(amount,0))
                Stocks.append(stocks)
                
    plt.figure()
    plt.plot(actual)
    plt.title('Close Test')
    plt.show()   
    print("Final value of stock:")
    print(actual[-1] * 1000)
    
    plt.figure()
    plt.plot(Amount)
    plt.title('Amount')
    plt.show()
    print("Final value of portfolio:")
    print(Amount[-1])
    
    print('TS=')
    print(Amount[-1] / (actual[-1] * 1000))
    
    return 


