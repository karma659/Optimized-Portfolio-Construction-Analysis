# 📈 Portfolio Optimization and Efficient Frontier Visualizer

A Python-based portfolio optimization tool that leverages historical stock data to simulate 10,000 portfolios and visualize the **Efficient Frontier**. The model uses the **Sharpe Ratio** as the objective function and optimizes asset allocation via the **SLSQP** algorithm.

---

## 🚀 Features

- ✅ Historical price data fetched using `yfinance`
- ✅ Computes log returns, expected return, volatility
- ✅ Simulates 10,000 random portfolios
- ✅ Optimizes weights to **maximize Sharpe Ratio**
- ✅ Plots:
  - Historical stock price trends
  - Efficient Frontier with Sharpe ratio heatmap
  - Highlighted optimal portfolio

---

## 🖼️ Sample Output

### 📊 Historical Stock Prices
![Price Plot](.png)

### 📉 Efficient Frontier
![Efficient Frontier](.png)

### 🌟 Optimal Portfolio Highlighted
![Optimal Portfolio](.png)

> **Optimal portfolio:** `[0.373, 0.246, 0.37, 0.0, 0.011, 0.0]`  
> **Expected return:** `0.246`  
> **Volatility:** `0.307`  
> **Sharpe Ratio:** `0.801`

---

## 📦 Installation

```bash
git clone https://github.com/karma659/portfolio-optimizer.git
cd portfolio-optimizer
pip install -r requirements.txt
