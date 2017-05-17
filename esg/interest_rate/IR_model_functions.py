import numpy as np

def interpolate1(l):
    out = []
    if l is not list:
        temp = list(l)
    if len(temp) == 1:
        pass
    else:
        for i in range(len(temp)-1):
            out.append(temp[i])
            out.append((temp[i+1] + temp[i])/2)
        out.append(temp[-1])
    return np.asarray(out)

def diff(y, x):
    n = len(y)
    d = np.zeros(n-1,'d')
    
    for i in range(n-1):
        d[i] = (y[i+1] - y[i])/(x[i+1] - x[i])
    return d
    