'''
This is a script to automate the process of the hyperparameter tuning, whilst guarding against overfitting. 

Here, as these are short term arb strategies, they make a large number of very small trades. We will be trying 
to optimise for linear, monotonic growth of the pnl of the strategy. To that end, total return is not exactly 
the objective here. 

Dataset is also partitioned to give us some out of sample data cross validate with. 
'''

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd 
import numpy as np 
from skopt import gp_minimize # type: ignore
from skopt.space import Real, Integer # type: ignore

# backtest engine and strategies
from backtester import Orchestrator
from algorithms.TradingStrategy import TradingLogic

# load data, filter down to most correlated stocks only.
dataset = pd.read_csv('data/qtm_data.csv', index_col = 'Datetime')
dataset = dataset[['IONQ', 'QBTS','RGTI']]
# to give us some out of sample data
dataset = dataset.loc[dataset.index <  "2025-08-23 14:30:00+00:00"]


def monotonicity_score(pnl_series):
    '''
    Mixes objective of sharpe and also cagr / max drawdown 
    '''
    # downsample to take every 300 data points
    pnl = pnl_series.values[::300]

    returns = pd.Series(pnl).diff().dropna()
    sharpe = returns.mean() / (returns.std() + 1e-9)
    max_dd = (np.maximum.accumulate(pnl) - pnl).max()
    calmar = (pnl[-1] - pnl[0]) / (max_dd + 1e-9)

    # just optimise risk adjusted returns for now, hopefully biasing monotonic growth. 
    score = sharpe 
    print(f'Another run completed, with sharpe = {sharpe * np.sqrt(252)}, calmar = {calmar}')

    return score


def run_backtest(alpha, pcs_removed, unwind_rate, sensitivity, min_len):
    '''
    Objective to be optimised 
    '''
    outs = Orchestrator(
        TradingLogic,
        start_time="2025-08-05 14:30:00+00:00",
        dataset=dataset,
        alpha=alpha,
        remove_pcs=True,
        pcs_removed=int(pcs_removed),
        unwind_rate=unwind_rate,
        sensitivity=sensitivity,
        min_len=int(min_len),
        bt_res=3
    ).run_orchestrator()
    
    # extracting information regarding backtest
    pnl_curve = outs[1]
    costs = np.abs(outs[0]) # type: ignore
    net_per_stock = costs.sum(axis=0).values
    transactions = sum(net_per_stock * dataset.iloc[-1])
    
    # book keeping
    print(f'For param set {alpha, pcs_removed, unwind_rate, sensitivity, min_len}: ')
    print(f'proft = {pnl_curve.iloc[-1]} with {transactions} total traded, for profit factor of {100 * pnl_curve.iloc[-1] / transactions}')
    return pnl_curve.iloc[-1]  / transactions

if __name__ == "__main__":
    space = [
        Real(0.4, 0.8, name='alpha'),
        Integer(1, 3, name='pcs_removed'),
        Integer(1, 20, name='unwind_rate'),
        Real(0.001, 0.1, name='sensitivity'),
        Integer(5, 40, name='min_len'),
    ]

    print('Beginning Optimisation')
    res = gp_minimize(
        func=lambda params: -run_backtest(*params),  # minimise negative score
        dimensions=space,
        n_calls=50,  # number of runs
        random_state=10,
    )
    print('Market Neutral strat:')
    print("Best score:", -res.fun)
    print("Best params:", res.x)

