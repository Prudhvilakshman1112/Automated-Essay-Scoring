from sklearn.metrics import mean_squared_error, cohen_kappa_score, r2_score
import numpy as np

def get_kappa_score(predictions, labels, scores):
    bins = np.linspace(min(scores), max(scores), 6)  # 6 edges = 5 bins
    def to_discrete_labels(vals, bins):
        return np.clip(np.digitize(vals, bins) - 1, 0, len(bins) - 2)
    pred_labels = to_discrete_labels(predictions, bins)
    true_labels = to_discrete_labels(labels, bins)
    try:
        return cohen_kappa_score(pred_labels, true_labels, weights='quadratic')
    except ValueError:
        return 0.0  # fallback if only one class predicted

def get_perform(predictions, labels, scores):
    kappa_score = get_kappa_score(predictions, labels, scores)
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return kappa_score, mse, r2
