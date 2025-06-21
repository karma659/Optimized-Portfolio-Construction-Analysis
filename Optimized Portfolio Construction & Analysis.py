import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import scipy.optimize as optimization

start_date = '2000-01-01'
end_date = '2025-01-01'
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']
stock_data = {}
NUM_TRADING_DAYS=252
NUM_PORTFOLIOS=10000

def download_data():
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data)

def plotdata(data):
    data.plot(figsize=(12,8))
    plt.show()

def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]

def show_statistics(log_daily_returns):
    print("Mean Annualized Return")
    print(log_daily_returns.mean() * NUM_TRADING_DAYS)
    print("Covariance Matrix")
    print( log_daily_returns.cov() * NUM_TRADING_DAYS)

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility,portfolio_return / portfolio_volatility])

def generate_portfolios(log_daily_returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(log_daily_returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(log_daily_returns.cov() * NUM_TRADING_DAYS, w))))
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

# scipy optimize module can find the minimum of a given function
# the maximum of a f(x) is the minimum of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

# what are the constraints? The sum of weights = 1 !!!
# f(x)=0 this is the function to minimize
def optimize_portfolio(weights, returns):
    # the sum of weights is 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # the weights can be 1 at most: 1 when 100% of money is invested into a single stock
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    ans=optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)
    # print(ans)
    return ans


def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio: ", optimum['x'].round(3))
    print("Expected return, volatility and Sharpe ratio: ",
          statistics(optimum['x'].round(3), returns))


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], marker='*', markersize=20.0)
    plt.show()

if __name__ == '__main__':

   dataset=download_data()
   plotdata(dataset)
   log_daily_returns = calculate_return(dataset)
   # show_statistics(log_daily_returns)
   pweights, means, risks = generate_portfolios(log_daily_returns)
   show_portfolios(means, risks)
   optimum = optimize_portfolio(pweights, log_daily_returns)
   print_optimal_portfolio(optimum, log_daily_returns)
   show_optimal_portfolio(optimum, log_daily_returns, means, risks)
