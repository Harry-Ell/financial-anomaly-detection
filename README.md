# Anomaly Detection in Financial Time Series 
This repository contains some of my code and additional results from section 4.2 of my dissertation, titled 'Real-Time Anomaly Detection in Multidimensional Financial Time Series' supervised by Idris Eckley and Florian Pein. 

In this section, we postulate a mathematical form for the financial anomalies, and test this prediction on financial data as part of a backtest. This is done within my own, simplified backtesting engine. Here we can account for things such as latency and slippage, however a buy-sell spread is not factored in. 

Initial tests imply that the edge acquired is too small to be accessible given the spreads on many of the traded stocks