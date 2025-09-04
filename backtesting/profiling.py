'''
Script where we can profile calls to backtester and test where the bottlenecks are.  

Main bottleneck is the process of interfacing R and python, but for now we can just 
optimise around this compuationally heavy part. 
'''

import cProfile
import pstats

import pandas as pd 

# my packages + data
from backtesting.backtester import Orchestrator
from algorithms.trading_strategy import TradingLogic
dataset = pd.read_csv('data/qtm_data.csv', index_col = 'Datetime')

# filter this to only include the most correlated stocks 
dataset = dataset[['IONQ', 'QBTS','RGTI']]

# wrapper func
def run():
    print('Entering backtester now', flush = True)
    _ = Orchestrator(
        TradingLogic,
        start_time="2025-08-05 14:30:00+00:00", 
        dataset = dataset,
        alpha = 0.72,  
        pcs_removed = 2,
        unwind_rate=10, 
        sensitivity=0.02, 
        min_len=9, 
        bt_res = 1
        ).run_orchestrator()
    
# run within profiler
cProfile.run("run()", "profile.out")
stats = pstats.Stats("profile.out")
stats.strip_dirs().sort_stats("cumtime").print_stats(30)
