'''
Simple exponential smoother
'''

import numpy as np 

def exp_smoother(time_series, alpha):
    '''
    function for doing full time series worth of smoothing, expects pandas series 
    as input
    '''
    smoothed_series = np.zeros_like(time_series, dtype = float)
    for idx, value in enumerate(time_series):
        if idx < 1:
            smoothed_series[idx] = value
        else:
            smoothed_series[idx] = alpha * value + (1-alpha)* smoothed_series[idx-1]
    return smoothed_series


