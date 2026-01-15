import argparse
from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

ARTIFACTDIR = Path("artifacts")

def runinference(inputcsv: Path, outputcsv: Path):
    model = load(ARTIFACTDIR / "model.joblib")
    scaler: StandardScaler = load(ARTIFACTDIR / "scaler.joblib")

    df = pd.readcsv(inputcsv)
    X = scaler.transform(df)
    preds = model.predict(X)

    pd.DataFrame({"prediction": preds}).tocsv(outputcsv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.addargument("--input", required=True)
    parser.addargument("--output", default="preds.csv")
    args = parser.parseargs()
    runinference(Path(args.input), Path(args.output))
