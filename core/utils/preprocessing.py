def standardize_data(x):
    mean = x.mean(0)
    std = x.std(0)
    eps = 1e-8
    scale = std + eps
    x = (x - mean) / scale
    return x, mean, scale
