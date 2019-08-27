
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score


def f1(preds, labels):
    return {
        "f1": f1_score(y_true=labels, y_pred=preds)
    }


def matthews_corr(preds, labels):
    return {
        "matthews_corr": matthews_corrcoef(labels, preds),
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "pearson_spearman_corr": (pearson_corr + spearman_corr) / 2,
    }
