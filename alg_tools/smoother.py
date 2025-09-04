'''
Simple exponential smoother
'''

import pandas as pd

def exp_smoother(time_series, alpha):
    '''
    vectorised function for doing full time series worth of smoothing, 
    expects pandas series as input
    '''
    return pd.Series(time_series).ewm(alpha=alpha, adjust=False).mean().to_numpy()
