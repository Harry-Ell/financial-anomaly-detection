'''
Backtesting engine for strategies which deal with trading on correlated stocks. 

Returns logs of trades, orders over time, and pnl. 
'''

import pandas as pd

class Orchestrator:
    '''
    Walk-forward backtester / orchestrator.
    '''
    def __init__(self, 
                 TradingLogic,
                 start_time,
                 alpha, 
                 remove_pcs = True,
                 pcs_removed=2, 
                 block_size=15,
                 dataset=pd.DataFrame(),
                 gap_reset_minutes=60,
                 slippage = 1, 
                 unwind_rate=10, 
                 sensitivity = 0.01,
                 min_len = 10, 
                 bt_res = 1):
        
        # BOOK KEEPING PARAMETERS
        self.start_time = pd.to_datetime(start_time)
        self.dataset_full = dataset  # full prices
        self.data = None             # working prices up to cursor

        # PREPROCESSING PARAMETERS
        self.alpha = float(alpha)
        self.remove_pcs = remove_pcs
        self.k = int(pcs_removed)
        self.block_size = int(block_size)
        self.gap_reset = pd.Timedelta(minutes=gap_reset_minutes)
        
        # TRADING STRATEDY PARAMETERS
        self.slippage = slippage
        self.unwind = unwind_rate
        self.trades_log = []         # list of dicts
        self.sensitivity = sensitivity
        self.min_anom = int(min_len)

        # BACKTEST PARAMETERS
        self.cursor_time = None      # point up to which we are looking 
        self.bot = None              # TradingLogic instance
        self.trading_bot = TradingLogic
        self.bt_res = bt_res

    # setup backtest 
    def first_pull(self):
        '''
        Take the data which has been given, move the index to be a datetime. Then, 
        initialise the trading bot with this in place. 
        '''
        self.dataset_full.index = pd.to_datetime(self.dataset_full.index)
        df = self.dataset_full

        # rows strictly before start_time
        seed = df.loc[df.index < self.start_time]
        if len(seed) < 2:
            raise ValueError("Need at least 2 rows before start_time to form first return.")
        self.data = seed.copy()
        self.cursor_time = seed.index[-1]

        # init bot with seed data
        self.bot = self.trading_bot(dataset=self.data,
                                start_time=self.start_time,
                                remove_pcs = self.remove_pcs,
                                pcs=self.k,
                                alpha=self.alpha, 
                                unwind_bars=self.unwind,
                                slippage=self.slippage, 
                                sensitivity = self.sensitivity, 
                                min_len = self.min_anom, 
                                bt_res = self.bt_res)
        # full fit on seed data
        self.bot.retrain_full(self.data)

    # walk forward block 
    def retraining_cycle(self):
        '''
        pick up the next 15 points out of the time series, store in block times
        '''
        # quick check if we have anything left to do
        after_cursor = self.dataset_full.index > self.cursor_time # type: ignore
        future_times = self.dataset_full.index[after_cursor]# type: ignore
        if len(future_times) == 0:
            return False  # nothing more to process

        # Take next block_size timestamps
        block_times = future_times[:self.block_size]

        prev_ts = self.cursor_time
        for ts in block_times:
            # we need to make sure we are not jumping overnight. if we are, pcs 
            # will likely breakdown a bit and we have to retrain fully
            if (ts - prev_ts) > self.gap_reset:
                # full retrain up to end of day, we may aswell since in reality we will
                # have plenty of time to do this
                # print('changing day')
                self.bot.retrain_full(self.data)# type: ignore

            # get the new row of prices, append to working data
            new_row = self.dataset_full.loc[[ts]]# type: ignore
            self.data = pd.concat([self.data, new_row])
            self.bot.notify_new_point(new_row) # notify the new bot # type: ignore ,

            # get signals and log
            signals = self.bot.run_strategy(self.trades_log) or {}# type: ignore
            self.trades_log.append({"time": ts, **signals})

            # quick step forward for overnight trading logic 
            prev_ts = ts

        # exit loop, move forward cursor and retrain
        self.cursor_time = block_times[-1]
        self.bot.retrain_full(self.data)# type: ignore

        return True
    
    def _compute_pnl(self):
        '''
        Price-units accounting (cash + inventory).
        trades_log rows are orders at decision time, in units of number of shares
        Orders at t fill at t+1+slippage, executed at that bar's price.

        Returns:
            total_pnl 
        '''
        
        if len(self.trades_log) == 0:
            return pd.Series(dtype=float)
            
        # Prices 
        prices = self.dataset_full.astype(float).sort_index()

        # Orders placed at decision times
        orders_raw = pd.DataFrame(self.trades_log).set_index("time").sort_index()

        # Keep only tickers that exist in price table and align
        tickers = [c for c in orders_raw.columns if c in prices.columns]
        prices = prices[tickers]
        orders = orders_raw[tickers].reindex(prices.index).fillna(0)  

        # Simulate execution latency here 
        slip = max(0, int(self.slippage))
        executed = orders.shift(1 + slip).reindex(prices.index).fillna(0)

        # Find out overall positions vs time
        positions = executed.cumsum()
        # Cash flow: buys reduce cash, sells increase cash
        cash_flows = -(executed * prices).sum(axis=1)        # value at EXECUTION bar price
        cash = cash_flows.cumsum()

        # Inventory mark and equity
        inv_value = (positions * prices).sum(axis=1)
        equity = cash + inv_value

        return equity

    # Point of entry to class
    def run_orchestrator(self):
        '''
        walk forward through the entire time series at once
        '''
        # start it off by doing the first data pull, initialise the bot once
        if self.dataset_full is None or self.bot is None:
            self.first_pull()

        # loop until we run out of data
        while self.retraining_cycle():
            pass

        # if no trades are made, return an empty df 
        if len(self.trades_log) == 0:
            return pd.DataFrame(columns=["time"]).set_index("time")

        # else make them into a data frame. 
        log_df = pd.DataFrame(self.trades_log).set_index("time").sort_index()
        total_pnl = self._compute_pnl()

        return log_df, total_pnl
