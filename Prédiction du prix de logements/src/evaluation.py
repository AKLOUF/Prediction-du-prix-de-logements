import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.modelselection import crossvalscore
from sklearn.metrics import meansquarederror, r2score

ARTIFACTDIR = Path("artifacts")
ARTIFACTDIR.mkdir(existok=True)

def evaluate(model, Xtrain, ytrain, Xtest, ytest):
    ypred = model.predict(Xtest)
    rmse = float(np.sqrt(meansquarederror(ytest, ypred)))
    r2 = float(r2score(ytest, ypred))

    try:
        cvscores = crossvalscore(model, Xtrain, ytrain, cv=5, scoring="r2")
        cvmean = float(np.mean(cvscores))
        cvstd = float(np.std(cvscores))
    except Exception:
        cvmean, cvstd = None, None

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "cvr2mean": cvmean,
        "cvr2std": cvstd,
    }
    (ARTIFACTDIR / "metrics.json").writetext(json.dumps(metrics, indent=2))
    return metrics

def plotimportance(model, featurenames, title: str):
    plt.figure(figsize=(8, 5))

    if hasattr(model, "featureimportances"):
        importances = model.featureimportances
        order = np.argsort(importances)
        labels = np.array(featurenames)[order]
        plt.barh(labels, importances[order])
    else:
        coefs = getattr(model, "coef", None)
        if coefs is not None:
            coefs = coefs.ravel()
            order = np.argsort(abs(coefs))
            labels = np.array(featurenames)[order]
            plt.barh(labels, coefs[order])

    plt.title(title)
    plt.tightlayout()

    out = ARTIFACTDIR / "feature_importance.png"
    plt.savefig(out)
    plt.close()
    return out
