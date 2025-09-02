'''
This algorithm proceeds as is explained in the report. 

Perform PC removal, then smooth, then test for spontaneous negative correlations in noise. 

Trade on these, bet on reconvergence after periods of negative correlation.
'''


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# import sys 
# sys.path.append('..')
from alg_tools.smoother import exp_smoother

# wrapper function which is written initially in r 
from alg_tools.capacc_wrapper import capa_cc

class TradingLogic:
    def __init__(self, 
                 dataset,          # all the data that the model is told about at once
                 start_time,       # this is basically our time index, we need this to make trades log
                 pcs_removed,      # integer, how much structure do we want to take from the series. 
                 alpha,            # used by our smoother 
                 slippage, 
                 unwind_bars = 10, 
                 sensitivity = 0.01,
                 min_len = 10, 
                 pca_lookback=300):
        
        # general setup/ data params
        self.dataset = dataset.copy()
        self.start_time = start_time
        self.tickers = list(dataset.columns)
        self.smoothed_series = None

        # exit/ execution params
        self.slippage = int(slippage)
        self.unwind_bars = int(unwind_bars)
        self.unwind_plan = { }  
        
        # pca related
        self.pca = None
        self.pca_resids = None
        self.pca_lookback = pca_lookback

        # STRAT PARAMS 
        self.alpha = float(alpha)
        self.sensitivity = sensitivity
        self.min_anom = int(min_len)
        self.k = int(pcs_removed)
        self.alpha = float(alpha)

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


    def notify_new_point(self, new_row: pd.DataFrame):
        '''
        Use current pca to fit a new point. given a new row, massage it on to the df, and find its returns
        '''
        new_row = new_row.reindex(columns=self.tickers)
        self.dataset = pd.concat([self.dataset, new_row])
        returns_new = self._returns_from_prices(self.dataset[-10:]).iloc[[-1]]

        # the only role of this is to get rid of the pylance hints 
        if self.pca is None or returns_new.isna().any(axis=None):
            return  # nothing to do yet

        # remove the pcs and add to the df we already have to store these 
        resid_df = self._remove_top_k_pcs(returns_new, self.pca)
        self.pca_resids = pd.concat([self.pca_resids, resid_df])

        # incremental smoothing update, if else is again to get rid of the type hint
        if self.smoothed_series is None or len(self.smoothed_series) == 0:
            self.smoothed_series = resid_df.copy()
        else:
            s_prev = self.smoothed_series.iloc[-1]
            s_new = self.alpha * resid_df.iloc[0] + (1 - self.alpha) * s_prev
            self.smoothed_series = pd.concat([self.smoothed_series, s_new.to_frame().T])


    def _executed_inventory(self, trade_log: list[dict]) -> pd.Series:
        '''
        this is to help us unwind trades like we planned on doing. 
        '''
        if not trade_log:
            return pd.Series(0.0, index=self.tickers)

        orders = (pd.DataFrame(trade_log)
                    .set_index("time")
                    .sort_index())
        # Align to our known timestamps
        idx = self.dataset.index  # same clock as smoothed_series
        orders = (orders.reindex(idx)
                        .fillna(0.0))[self.tickers]

        # Execution: orders placed at t fill at t+1+slippage
        executed = orders.shift(1 + max(0, self.slippage)).fillna(0.0)

        # Only count fills up to "now"
        now = idx[-1]
        executed = executed.loc[:now]

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
        signals = pd.Series(0.0, index=self.tickers)

        # Current executed inventory (accounts for slippage)
        inv = self._executed_inventory(trade_log)

        if len(relevant) > 0:
            # print(relevant, flush = True)
            # 1) Trade the anomalies; cancel unwind plan on affected tickers
            for idx, row in relevant.iterrows():
                # 'variate' indexing in capa is often 1-based; keep your -1 but guard bounds
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
            q = inv.get(tkr, 0.0)

            if abs(q) < 1e-12:
                # flat -> no plan needed
                if tkr in self.unwind_plan:
                    del self.unwind_plan[tkr]
                continue

            plan = self.unwind_plan.get(tkr)
            if plan is None:
                # Start a new linear plan: equal chunks that sum exactly to -q
                chunk = -q / N          # negative if q>0 (sell), positive if q<0 (buy)
                self.unwind_plan[tkr] = {"chunk": float(chunk), "left": N}
                plan = self.unwind_plan[tkr]

            # Emit one equal chunk per bar
            signals[tkr] = plan["chunk"]
            plan["left"] -= 1
            if plan["left"] <= 0:
                del self.unwind_plan[tkr]

        return signals.to_dict()


    def RunStrategy(self,trade_log):
        return self.TradingStrategy(trade_log)
