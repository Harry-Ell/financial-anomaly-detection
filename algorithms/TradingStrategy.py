'''
Here we make use of our base class, and only append a small amount of 
additional logic. The idea is that this can be used in an online 
implementation as it should be agnostic to the environment in which 
it is ran. 
'''
import numpy as np 
import pandas as pd 

from algorithms.base import TradingAlgorithm
from alg_tools.capacc_wrapper import capa_cc

class TradingLogic(TradingAlgorithm):

    def TradingStrategy(self, trade_log):
        '''
        - If recent anomalies: place buy orders proportional to anomaly size
          cancel any existing unwind plan for affected tickers.
        - Else: linearly unwind existing executed inventory over `unwind_bars` bars
        using equal-sized chunks.
        '''

        relevant, signals, inv = self.capa_cc_call(trade_log)
        affected_series = []
        # if we have anomalies, begin to populate the list of signals
        if len(relevant) > 0:
            for _, row in relevant.iterrows():
                # relevant contains rows which note which series is affected
                j = int(row["variate"]) - 1
                tkr = self.tickers[j]
                prices = self.dataset.iloc[-1]

                # expensive ones we want to buy less of, hence we rescale. 
                size = (float(row["size"])) 
                rescaling = float(prices.iloc[j]) / 50
                signals[tkr] += size / rescaling # buy amount proportional  to anomaly magnitude
                affected_series.append(tkr)
                # cancel any existing unwind plan; we’re actively trading this name
                if tkr in self.unwind_plan:
                    self.unwind_plan.pop(tkr, None)
            
        # with all other tickers, lets continue the unwinding as though nothing is happening 
        signals = self._unwind_trades(affected_series, signals, inv)
        self.signals = signals.to_dict()
    
    def capa_cc_call(self, trade_log):
        '''
        Calling routine to capa CC wrapper function, defined separately. 

        This process is very computationally intensive, hence we make use of a backtest
        resolution, whereby we only run this every n data points. 
        '''
        # For performance purposes, we will call this only every n data points
        if len(self.smoothed_series) % self.backtest_res == 0 and len(self.smoothed_series) > self.min_anom:
            # sub window in which we detect anomalies, rescaling to normalise
            df = self.smoothed_series.iloc[-50:]
            X = df.to_numpy()
            denom = np.max(np.abs(X)) if np.max(np.abs(X)) != 0 else 1.0
            rescaled = X / denom

            # assume that beforehand, noise was uncorrelated/ diagonal, and call capa cc
            Q = np.eye(len(self.tickers))
            anoms_exp = capa_cc(rescaled, Q, b=self.sensitivity, b_point=10, min_seg_len=self.min_anom)
            # we are interested in anomalies that have ended recently; time to start trading
            relevant = anoms_exp.loc[(anoms_exp["end"] >= (len(df) - 5)) & ~(anoms_exp["end"] >= len(df)-2)] if len(anoms_exp) else anoms_exp
        else:
            relevant = pd.DataFrame()

        # initialise the series which stores the signals, fetch current inventory
        signals = pd.Series(0.0, index=self.tickers)
        inv = self._executed_inventory(trade_log)
        return relevant, signals, inv