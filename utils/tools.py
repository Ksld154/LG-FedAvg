import numpy as np

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.nan
    
    return sum(filter(None, data[-window_size:])) / window_size