'''
This algorithm proceeds as is explained in the report. 

Perform PC removal, then smooth, then test for spontaneous negative correlations in noise. 

Trade on these, bet on reconvergence after periods of negative correlation.
'''


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from alg_tools.smoother import exp_smoother
from alg_tools.capacc_wrapper import capa_cc

class TradingLogic:
    def __init__(self, 
                 dataset, start_time, pcs_removed, alpha, slippage,
                 unwind_bars=10, sensitivity=0.01, min_len=10, pca_lookback=300, bt_res = 1):
        self.tickers = list(dataset.columns)
        self.start_time = start_time
        self.slippage = int(slippage)
        self.unwind_bars = int(unwind_bars)
        self.unwind_plan = {}
        self.alpha = float(alpha)
        self.sensitivity = sensitivity
        self.min_anom = int(min_len)
        self.k = int(pcs_removed)
        self.pca_lookback = pca_lookback
        self.backtest_res = bt_res

        # store raw data as numpy
        self.dataset_values = dataset.to_numpy(copy=True)
        self.dataset_index = list(dataset.index)

        # placeholders
        self.pca = None
        self.pca_resids_values = None
        self.smoothed_values = None


    def notify_new_point(self, new_row: pd.DataFrame):
        """
        Append new data point, update PCA residuals & smoothed series.
        """
        # ensure column order matches
        new_row = new_row.reindex(columns=self.tickers)

        # update dataset (numpy storage)
        self.dataset_values = np.vstack([self.dataset_values, new_row.to_numpy()])
        self.dataset_index.append(new_row.index[0])

        # compute last return
        if len(self.dataset_index) < 2 or self.pca is None:
            return

        last_prices = pd.DataFrame(self.dataset_values[-10:], 
                                   index=self.dataset_index[-10:], 
                                   columns=self.tickers)
        returns_new = self._returns_from_prices(last_prices).iloc[[-1]]
        if returns_new.isna().any(axis=None):
            return

        # residual
        resid_df = self._remove_top_k_pcs(returns_new, self.pca)
        resid_row = resid_df.to_numpy()[0]

        if self.pca_resids_values is None:
            self.pca_resids_values = resid_row[None, :]
        else:
            self.pca_resids_values = np.vstack([self.pca_resids_values, resid_row])

        # smoothing
        if self.smoothed_values is None:
            self.smoothed_values = resid_row[None, :]
        else:
            s_prev = self.smoothed_values[-1]
            s_new = self.alpha * resid_row + (1 - self.alpha) * s_prev
            self.smoothed_values = np.vstack([self.smoothed_values, s_new])

    # HELPER FUNCTIONS
    def _returns_from_prices(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        '''
        Simple func which takes in a df and works out log ratio of prices. 
        '''
        return np.log(df_prices).diff() # type: ignore

    def _return_top_k_pcs(self, returns: pd.DataFrame, pca: PCA) -> pd.DataFrame:
        '''
        this returns the top k pcs and we trade on these. this ends up being a mean reversion 
        strategy, except with fewer false positives
        '''
        Xc = returns.values - pca.mean_
        Z = Xc @ pca.components_.T
        Z[:, :self.k] = 0.0
        Xhat = Z @ pca.components_ + pca.mean_
        resid = returns.values - Xhat
        return pd.DataFrame(resid, index=returns.index, columns=self.tickers)
        
    def _remove_top_k_pcs(self, returns: pd.DataFrame, pca: PCA) -> pd.DataFrame:
        '''
        this has been erroneously giving us the pcs and we trade based on that 
        '''
        Xc = returns.values - pca.mean_
        Z = Xc @ pca.components_.T
        Z[:, :self.k] = 0.0
        Xhat = Z @ pca.components_ + pca.mean_
        resid = returns.values - Xhat
        pca_free = returns - resid
        return pd.DataFrame(pca_free, index=returns.index, columns=self.tickers)


    # public methods which we will call from the other script
    def retrain_full(self, full_dataset: pd.DataFrame):
        '''
        given all the data the model has now, perform a full retraining. 
        '''
        full_dataset = full_dataset[self.tickers]
        returns = self._returns_from_prices(full_dataset).dropna()

        # fit a new pca, remove the top k pcas and smooth the series. 
        pca_new = PCA().fit(returns.iloc[-self.pca_lookback:])
        resids_new = self._remove_top_k_pcs(returns, pca_new)
        smoothed_new = resids_new.apply(
            lambda col: pd.Series(exp_smoother(col.to_numpy(), self.alpha), index=col.index)
        )

        # swap out all of the variables/update them with the new data, model and transformed data
        self.dataset = full_dataset
        self.pca = pca_new
        self.pca_resids = resids_new
        self.smoothed_series = smoothed_new



    def _executed_inventory(self, trade_log: list[dict]) -> pd.Series:
        '''
        this is to help us unwind trades like we planned on doing. 

        '''
        # if we do not have any trades made yet, just say 0
        if not trade_log:
            return pd.Series(0.0, index=self.tickers)

        orders = (pd.DataFrame(trade_log)
                    .set_index("time")
                    .sort_index())
        # Align to our known timestamps
        idx = self.dataset.index  # same clock as smoothed_series
        orders = (orders.reindex(idx)
                        .fillna(0.0))[self.tickers]

        # orders placed at t fill at t+1+slippage
        executed = orders#.shift(1 + max(0, self.slippage)).fillna(0.0)

        # Only count fills up to "now"
        now = idx[-1]
        executed = executed.loc[:now]

        # sum over all the orders until now 
        pos = executed.cumsum().iloc[-1]
        pos = pos.reindex(self.tickers).fillna(0.0)
        return pos

    def TradingStrategy(self, trade_log):
        """
        - If recent anomalies: place buy orders proportional to anomaly 'size';
        cancel any existing unwind plan for affected tickers.
        - Else: linearly unwind existing executed inventory over `unwind_bars` bars
        using equal-sized chunks.
        """
        if self.smoothed_series is None or len(self.smoothed_series) == 0:
            return {}
        # to speed up, this is going to be ran only every n data points
        if len(self.smoothed_series) % self.backtest_res == 0:
            # --- anomaly detection window ---
            df = self.smoothed_series.iloc[-50:]
            X = df.to_numpy()
            denom = np.max(np.abs(X)) if np.max(np.abs(X)) != 0 else 1.0
            rescaled = X / denom

            # old cov mat 
            Q = np.eye(len(self.tickers))

            # ROUTINE TO CALL CAPA CC
            anoms_exp = capa_cc(rescaled, Q, b=self.sensitivity, b_point=10, min_seg_len=self.min_anom)
            relevant = anoms_exp.loc[anoms_exp["end"] >= (len(df) - 1)] if len(anoms_exp) else anoms_exp
        else:
            relevant = pd.DataFrame()
        signals = pd.Series(0.0, index=self.tickers)

        # Current executed inventory (accounts for slippage)
        inv = self._executed_inventory(trade_log)

        if len(relevant) > 0:
            # print(relevant, flush = True)
            # 1) Trade the anomalies; cancel unwind plan on affected tickers
            for idx, row in relevant.iterrows():
                j = int(row["variate"]) - 1
                if 0 <= j < len(self.tickers):
                    tkr = self.tickers[j]
                    prices = self.dataset.iloc[-1]
                    # expensive ones we want to buy less of 
                    size = (float(row["size"])) 
                    rescaling = float(prices.iloc[j]) / 50
                    signals[tkr] += size / rescaling # "always buy amount equal to anomaly magnitude"
                    # cancel any existing unwind plan; we’re actively trading this name
                    if tkr in self.unwind_plan:
                        del self.unwind_plan[tkr]
            # For names without anomaly but with inventory, we leave them unchanged *this bar*.
            # (You can also choose to keep unwinding unaffected names—up to you.)
            return signals.to_dict()

       # 2) No anomalies now: linearly unwind to flat over unwind_bars bars
        N = max(1, self.unwind_bars)
        for tkr in self.tickers:
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

            # Follow the existing plan exactly
            signals[tkr] = plan["chunk"]
            plan["left"] -= 1

            if plan["left"] <= 0:
                # Force to zero at the end (avoid residuals)
                signals[tkr] = -inv_pos
                self.unwind_plan.pop(tkr, None)


        return signals.to_dict()


    def RunStrategy(self,trade_log):
        return self.TradingStrategy(trade_log)
