import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



df = pd.read_csv('220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv')
df = df[df.MONAT.isin(['Summe'])==False]
df = df[df.JAHR.isin([2021,2022])==False]



df.reset_index(drop=True, inplace=True)



df['MONAT']=df['MONAT'].astype(str).str[-2:].astype(np.int64)



df.fillna(method='ffill',inplace=True)
categorical_features = ['MONATSZAHL','AUSPRAEGUNG','JAHR','MONAT']


categorical_transformer = OneHotEncoder(handle_unknown="ignore")



preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough')




pipeline = make_pipeline(preprocessor, linear_model.LinearRegression(normalize=True))



X = df[['MONATSZAHL','AUSPRAEGUNG','JAHR','MONAT','VORJAHRESWERT']]
y = df['WERT']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 52)



pipeline.fit(X_train, y_train)


# calculate the accuracy score from test data
print("model score: %.3f" % pipeline.score(X_test, y_test))
print("model score: %.3f" % pipeline.score(X_train, y_train))
