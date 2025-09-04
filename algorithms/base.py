'''
Base class from which we will inherit for our other algorithms 
'''

from abc import ABC, abstractmethod

import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA 

# my modules
from alg_tools.smoother import exp_smoother

class TradingAlgorithm(ABC):
    def __init__(self, 
                 dataset, 
                 start_time, 
                 remove_pcs,
                 pcs, 
                 alpha, 
                 slippage,
                 unwind_bars=10, 
                 sensitivity=0.01, 
                 min_len=10, 
                 pca_lookback=300, 
                 bt_res = 1):
        
        # BOOK KEEPING PARAMETERS
        self.tickers = list(dataset.columns)
        self.signals = {}
        self.start_time = start_time
        self.smoothed_values = None
        self.dataset_values = dataset.to_numpy(copy=True)
        self.dataset_index = list(dataset.index)
        self.pca_resids_values = None

        # PREPROCESSING PARAMETERS
        self.alpha = float(alpha)
        self.pca = None
        self.pca_lookback = pca_lookback
        self.remove_pcs = remove_pcs
        self.k = int(pcs)

        # TRADING STRATEDY PARAMETERS
        self.slippage = int(slippage)
        self.unwind_bars = int(unwind_bars)
        self.unwind_plan = {}
        self.sensitivity = sensitivity
        self.min_anom = int(min_len)

        # BACKTEST PARAMETERS
        self.backtest_res = bt_res

###################################################################
######################### ABSTRACT METHODS ######################## 
###################################################################

    @abstractmethod
    def trading_strategy(self, trade_log):
        '''
        Core logic function in which anomalies are detected and potentially 
        acted on 
        '''
        pass

###################################################################
##################### PREPROCESSING FUNCTIONS ##################### 
###################################################################

    def _returns_from_prices(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        '''
        Take in a df, return log ratios of prices. 
    
        Helps transform the heavy tails of distributions to easier ones to work with
        '''
        return np.log(df_prices).diff() # type: ignore

    def _return_top_k_pcs(self, returns: pd.DataFrame, pca: PCA) -> pd.DataFrame:
        '''
        Extract principal components from the selection of series, and trade on them 

        This is useful in reducing noise experienced when trading single stocks, and 
        instead you are only trading on common drivers which are felt by all stocks. 

        Likely to be used with a mean reversion strategy, however this could breakdown 
        in periods of rapid price changes. 
        '''
        Xc = returns.values - pca.mean_
        Z = Xc @ pca.components_.T
        Z[:, :self.k] = 0.0
        Xhat = Z @ pca.components_ + pca.mean_
        resid = returns.values - Xhat
        return pd.DataFrame(resid, index=returns.index, columns=self.tickers)
        
    def _remove_top_k_pcs(self, returns: pd.DataFrame, pca: PCA) -> pd.DataFrame:
        '''
        As opposed to the above method which returns the PCs, this one removes them 
        and returns the residuals.

        This is more in line with the postulated mathematical model postulated, where we 
        now have interest in this `unexplainable' structure which exists in these series. 
        '''
        Xc = returns.values - pca.mean_
        Z = Xc @ pca.components_.T
        Z[:, :self.k] = 0.0
        Xhat = Z @ pca.components_ + pca.mean_
        resid = returns.values - Xhat
        pca_free = returns - resid
        return pd.DataFrame(pca_free, index=returns.index, columns=self.tickers)
    


###################################################################
######################### TRADE EXIT LOGIC ######################## 
###################################################################

    def _executed_inventory(self, trade_log: list[dict]) -> pd.Series:
        '''
        This function tracks all real and pending orders we have made, and 
        allows us to set up an unwinding of trades. 

        In order to plan future trades, 

        '''
        # if we do not have any trades made yet, just say 0
        if not trade_log:
            return pd.Series(0.0, index=self.tickers)

        # else, define a new df to populate
        orders = (pd.DataFrame(trade_log)
                    .set_index("time")
                    .sort_index())
        # Align to our known timestamps
        idx = self.dataset.index  
        orders = (orders.reindex(idx).fillna(0.0))[self.tickers]


        # sum over all the orders until now 
        pos = orders.cumsum().iloc[-1]
        pos = pos.reindex(self.tickers).fillna(0.0)
        return pos
    
    def _unwind_trades(self, affected_series, signals, inv):
        '''
        Given information from the above function, ie net positions, we 
        can begin to either act on plans to reduce positions down to 0, 
        or not act yet since we have just made an order on a particular 
        ticker
        '''
        N = max(1, self.unwind_bars)
        for tkr in self.tickers:
            if tkr not in affected_series:
                # pull inventory for this ticker
                inv_pos = inv.get(tkr, 0.0)

                # If already flat, clear plan
                if abs(inv_pos) < 1e-12:
                    self.unwind_plan.pop(tkr, None)
                    continue

                # Use existing plan if it exists
                plan = self.unwind_plan.get(tkr)
                if plan is None:
                    # Create a plan once, based on current inventory
                    chunk = -inv_pos / N
                    plan = {"chunk": float(chunk), "left": N}
                    self.unwind_plan[tkr] = plan

                # Follow the existing plan exactly, decrease position and 
                # reduce number of remaining steps
                signals[tkr] = plan["chunk"]
                plan["left"] -= 1

                # mild book keeping
                if plan["left"] <= 0:
                    # Force to zero at the end (avoid residuals)
                    signals[tkr] = -inv_pos
                    self.unwind_plan.pop(tkr, None)
        return signals


###################################################################
########################## PUBLIC METHODS ######################### 
###################################################################
    def notify_new_point(self, new_row: pd.DataFrame):
        """
        Append a new observation and update PCA residuals and smoothed series.
        Keeps both NumPy arrays and DataFrame views in sync so downstream logic
        uses fresh data.
        """

        # Append to NumPy storage
        self.dataset_values = np.vstack([self.dataset_values, new_row.to_numpy()])
        self.dataset_index.append(new_row.index[0])

        # Refresh DataFrame view of raw prices
        self.dataset = pd.DataFrame(self.dataset_values,index=self.dataset_index,columns=self.tickers)

        # Need at least two points and a trained PCA model
        if len(self.dataset_index) < 2 or self.pca is None:
            return

        # Compute the latest return
        last_prices = self.dataset.iloc[-2:]
        returns_new = self._returns_from_prices(last_prices).iloc[[-1]]
        if returns_new.isna().any(axis=None):
            return

        # Remove top k PCs to get residual
        if self.remove_pcs:
            resid_df = self._remove_top_k_pcs(returns_new, self.pca)
        else: 
            resid_df = self._return_top_k_pcs(returns_new, self.pca)
        resid_row = resid_df.to_numpy()[0]

        # Store residuals in NumPy + DataFrame form
        if self.pca_resids_values is None:
            self.pca_resids_values = resid_row[None, :]
        else:
            self.pca_resids_values = np.vstack([self.pca_resids_values, resid_row])
        self.pca_resids = pd.DataFrame(
            self.pca_resids_values,
            index=self.dataset_index[-len(self.pca_resids_values):],
            columns=self.tickers,
        )

        # Exponential smoothing of residuals
        if self.smoothed_values is None:
            self.smoothed_values = resid_row[None, :]
        else:
            s_prev = self.smoothed_values[-1]
            s_new = self.alpha * resid_row + (1 - self.alpha) * s_prev
            self.smoothed_values = np.vstack([self.smoothed_values, s_new])

        # Refresh DataFrame view of smoothed series
        self.smoothed_series = pd.DataFrame(
            self.smoothed_values,
            index=self.dataset_index[-len(self.smoothed_values):],
            columns=self.tickers,
        )  

    def retrain_full(self, full_dataset: pd.DataFrame):
        '''
        given all the data the model has now, perform a full retraining. 
        '''
        full_dataset = full_dataset[self.tickers]
        returns = self._returns_from_prices(full_dataset).dropna()

        # fit a new pca, remove the top k pcas and smooth the series. 
        pca_new = PCA().fit(returns.iloc[-self.pca_lookback:])
        if self.remove_pcs:
            resids_new = self._remove_top_k_pcs(returns, pca_new)
        else:
            resids_new = self._return_top_k_pcs(returns, pca_new)
        smoothed_new = resids_new.apply(
            lambda col: pd.Series(exp_smoother(col.to_numpy(), self.alpha), index=col.index)
        )

        # swap out all of the variables/update them with the new data, model and transformed data
        self.dataset = full_dataset
        self.pca = pca_new
        self.pca_resids = resids_new
        self.smoothed_series = smoothed_new

    def run_strategy(self, trade_log):
        self.trading_strategy(trade_log)
        return self.signals
