'''
This is a script to automate the process of the hyperparameter tuning, whilst guarding against overfitting. 

Here, as these are short term arb strategies, they make a large number of very small trades. We will be trying 
to optimise for linear, monotonic growth of the pnl of the strategy. To that end, total return is not exactly 
the objective here
'''
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd 
import numpy as np 
from skopt import gp_minimize # type: ignore
from skopt.space import Real, Integer # type: ignore

# backtest engine and strategies
from backtester import Orchestrator
from algorithms.TradeLogic1 import TradingLogic as TradingLogic1
dataset = pd.read_csv('data/qtm_data.csv', index_col = 'Datetime')

def monotonicity_score(pnl_series):
    '''
    Mixes objective of sharpe and also cagr / max drawdown 
    '''
    # normalise by scale of pnl
    pnl = pnl_series.values
    returns = pd.Series(pnl).diff().dropna()
    
    sharpe = returns.mean() / (returns.std() + 1e-9)
    max_dd = (np.maximum.accumulate(pnl) - pnl).max()
    calmar = (pnl[-1] - pnl[0]) / (max_dd + 1e-9)

    # equal weighted combination
    score = sharpe #+ calmar
    print(f'Another run completed, with sharpe = {sharpe}, calmar = {calmar}')

    return score


def run_backtest(alpha, pcs_removed, slippage, unwind_rate, sensitivity, min_len):
    '''
    Objective to be optimised 
    '''
    outs = Orchestrator(
        TradingLogic1,
        start_time="2025-08-05 14:30:00+00:00",
        dataset=dataset,
        alpha=alpha,
        pcs_removed=int(pcs_removed),
        slippage=slippage,
        unwind_rate=unwind_rate,
        sensitivity=sensitivity,
        min_len=int(min_len),
    ).RunOrchestrator()
    
    pnl_curve = outs[1]
    return monotonicity_score(pnl_curve)

if __name__ == "__main__":
    space = [
        Real(0.4, 0.8, name='alpha'),
        Integer(1, 3, name='pcs_removed'),
        Real(1, 10, name='slippage'),
        Real(1, 20, name='unwind_rate'),
        Real(0.001, 0.05, name='sensitivity'),
        Integer(5, 50, name='min_len'),
    ]
    print('Beginning Optimisation')
    res = gp_minimize(
        func=lambda params: -run_backtest(*params),  # minimise negative score
        dimensions=space,
        n_calls=50,  # number of runs
        random_state=42,
    )

    print("Best score:", -res.fun)
    print("Best params:", res.x)
