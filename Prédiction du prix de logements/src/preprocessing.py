from typing import List
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"MedHouseVal": "target_price"}, inplace=True)
    return df

def split_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop("target_price", axis=1)
    y = df["target_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    feature_names: List[str] = X.columns.tolist()
    return X_train_s, X_test_s, y_train.values, y_test.values, scaler, feature_names
