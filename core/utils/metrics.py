def score(pred, actual):
    pred = (pred > 0.5).float()
    correct = (pred == actual).float().sum().item()
    total = actual.shape[0]
    accuracy = correct / total
    tp = ((pred == 1).float() * (actual == 1).float()).sum().item()
    fp = ((pred == 1).float() * (actual == 0).float()).sum().item()
    fn = ((pred == 0).float() * (actual == 1).float()).sum().item()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 1 / (1 / precision + 1 / recall)
    return accuracy, precision, recall, f1
