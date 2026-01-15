import argparse
from joblib import dump
from src.preprocessing import loaddata, splitscale
from src.modeltraining import train
from src.evaluation import evaluate, plotimportance, ARTIFACTDIR

def main():
    parser = argparse.ArgumentParser(description="Housing Price Prediction — Pipeline")
    parser.addargument("--model", choices=["lr", "rf"], default="rf")
    parser.addargument("--testsize", type=float, default=0.2)
    parser.addargument("--randomstate", type=int, default=42)
    args = parser.parseargs()

    df = loaddata()
    Xtrain, Xtest, ytrain, ytest, scaler, featurenames = splitscale(
        df, args.testsize, args.randomstate
    )

    model = train(args.model, Xtrain, ytrain)
    metrics = evaluate(model, Xtrain, ytrain, Xtest, ytest)

    title = "Feature Importances — RandomForest" if args.model == "rf" else "Coefficients — Linear Regression"
    plotimportance(model, featurenames, title)

    dump(model, ARTIFACTDIR / "model.joblib")
    dump(scaler, ARTIFACTDIR / "scaler.joblib")

    print("Training complete. Metrics:", metrics)

if __name__ == "__main__":
    main()
