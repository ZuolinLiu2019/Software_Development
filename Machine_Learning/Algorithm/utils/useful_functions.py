def median(x):
    n = len(x)
    x = sorted(x)
    return x[n//2]

def vote(Y, default=0):
    return default if len(Y) == 0 else max(Y, key=Y.count)
