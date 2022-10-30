from flask import request
import pickle
import pandas as pd

def predict():
    input_data = [[i] for i in request.args.getlist('inputs')]
    keys = ['MONATSZAHL','AUSPRAEGUNG','JAHR','MONAT','VORJAHRESWERT']
    query = dict(zip(keys,input_data))
    with open('./pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
    sample = pd.DataFrame(data=query)
    prediction=pipeline.predict(sample)
    return int(prediction)

if __name__=='__main__':
    print(predict())
