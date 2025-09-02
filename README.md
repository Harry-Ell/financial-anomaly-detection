# Financial Anomaly Detection

This repository contains supplementary code and results for Section 4.2 of the dissertation **"Real-Time Anomaly Detection in Multidimensional Financial Time Series"** supervised by Idris Eckley and Florian Pein.  
The project investigates whether short-lived anomalies—brief periods when residual returns across multiple assets move together in unexpected ways—can be detected in real time and traded profitably.

## Goal

To define a practical notion of “anomaly” in financial time series, detect such events as they occur, and evaluate whether they provide a tradable edge after accounting for latency and slippage.

## Approach

1. **Modeling:** Asset log prices are treated as having a latent mean and noise component. An anomaly is a transient, collective deviation in the noise terms.
2. **Removing common structure:** A rolling PCA strips out dominant market factors, leaving residual returns.
3. **Smoothing:** Residuals are exponentially smoothed to reduce microstructure noise.
4. **Anomaly detection:** The CAPA algorithm flags segments where multiple smoothed residuals move together unexpectedly.
5. **Trading logic:**  
   - *TradeLogic1* seeks reconvergence after sudden negative correlations between correlated stocks.  
   - *TradeLogic2* bets on mean reversion when principal-component trends break out.
6. **Backtesting:** A simplified engine simulates latency and slippage to test strategies (bid–ask spread is not modeled).

## Repository Structure

- **alg_tools/**
  - `smoother.py` – exponential smoothing routine.
  - `capacc_wrapper.py` – Python wrapper for the R implementation of CAPA.
- **algorithms/**
  - `TradeLogic1.py` – anomaly-based statistical arbitrage on correlated names.
  - `TradeLogic2.py` – mean-reversion on principal-component trends.
- **backtesting/**
  - `backtester.py` – orchestrates simulations with latency and slippage.
  - `parameter_optimiser.py` – Bayesian optimization of strategy parameters.
  - `example_backtest.ipynb` – notebook demonstrating end-to-end usage.

## Getting Started

1. Supply historical price data as a pandas DataFrame (datetime index, ticker columns).
2. Instantiate `TradingLogic` with the desired parameters and feed data incrementally.
3. Use `backtester.py` to run simulations; the notebook shows a typical workflow.

## Dependencies and Caveats

- Python packages: NumPy, pandas, scikit-learn, etc.
- R installation (for CAPA) accessible via `capacc_wrapper.py`.
- Results are illustrative; no bid–ask spread is modeled, and nothing here constitutes investment advice.

