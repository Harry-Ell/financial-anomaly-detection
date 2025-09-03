'''
Script where we can profile calls to backtester and test where the bottlenecks are 
'''

import cProfile
import pstats

import pandas as pd 
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# my packages + data
from backtester import Orchestrator
from algorithms.TradeLogic1 import TradingLogic as TradingLogic1
dataset = pd.read_csv('data/qtm_data.csv', index_col = 'Datetime')


def run():
    outs1 = Orchestrator(
        TradingLogic1,
        start_time="2025-08-05 14:30:00+00:00", 
        dataset = dataset,
        alpha = 0.72,  
        pcs_removed = 2,
        slippage = 8,
        unwind_rate=12, 
        sensitivity=0.0225, 
        min_len=9, 
        bt_res = 4
        ).RunOrchestrator() 
    


cProfile.run("run()", "profile.out")

stats = pstats.Stats("profile.out")
stats.strip_dirs().sort_stats("cumtime").print_stats(30)
