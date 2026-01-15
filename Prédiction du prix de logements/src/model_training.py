from typing import Literal
from sklearn.linearmodel import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train(modelname: Literal['lr', 'rf'], Xtrain, ytrain):
    if modelname == "lr":
        model = LinearRegression()
    elif modelname == "rf":
        model = RandomForestRegressor(nestimators=200, randomstate=42)
    else:
        raise ValueError("model must be 'lr' or 'rf'")

    model.fit(Xtrain, ytrain)
    return model
