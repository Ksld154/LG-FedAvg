import numpy as np
import math

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.nan
    
    return sum(filter(None, data[-window_size:])) / window_size



def convert_size(size_bits):
    if size_bits == 0:
        return "0 B"
    
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    size_bytes = size_bits / 8
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 4)

    #    return "%s %s" % (s, size_name[i])
    return f'{s} {size_name[i]}'
