from flask import request
import pandas as pd

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def predict():
    input_data = [[i] for i in request.args.getlist('inputs')]
    keys = ['MONATSZAHL','AUSPRAEGUNG','JAHR','MONAT','VORJAHRESWERT']
    query = dict(zip(keys,input_data))
    
    df = pd.read_csv('220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv')
    df = df[df.MONAT.isin(['Summe'])==False]
    df = df[df.JAHR.isin([2021,2022])==False]
    df.fillna(method='ffill',inplace=True)
    categorical_features = ['MONATSZAHL','AUSPRAEGUNG','JAHR','MONAT']
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)],remainder='passthrough')
    pipeline = make_pipeline(preprocessor, linear_model.LinearRegression(normalize=True))
    X = df[['MONATSZAHL','AUSPRAEGUNG','JAHR','MONAT','VORJAHRESWERT']]
    y = df['WERT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 52)
    pipeline.fit(X_train, y_train)

    sample = pd.DataFrame(data=query)
    prediction=pipeline.predict(sample)
    return int(prediction)

if __name__=='__main__':
    print(predict())
