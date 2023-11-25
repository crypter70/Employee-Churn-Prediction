from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import utils as util
import pandas as pd
import numpy as np

import data_pipeline as data_pipeline
import preprocessing as preprocessing


config_data = util.load_config()

ohe_education = util.pickle_load('../' + config_data['ohe_education_path'])
ohe_city = util.pickle_load('../' + config_data['ohe_city_path'])
ohe_gender = util.pickle_load('../' + config_data['ohe_gender_path'])
ohe_everbenched = util.pickle_load('../' + config_data['ohe_everbenched_path'])

model_data = util.pickle_load('../' + config_data['final_model']['model_directory'] + config_data['final_model']['model_name'])


class ApiData(BaseModel):
    Education : str
    City : str
    Gender : str
    EverBenched : str
    JoiningYear : int
    PaymentTier : int
    Age : int
    ExperienceInCurrentDomain : int


app = FastAPI()

@app.get('/')
def home():
    return "Hello, Employee Churn Prediction up!"


@app.post("/predict/")
def predict(data: ApiData):    
    
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

    # data defence
    try:
        data_pipeline.check_data(data, config_data, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}

    # encoding
    data = preprocessing.ohe_transform(data, "Education", ohe_education)
    data = preprocessing.ohe_transform(data, "City", ohe_city)
    data = preprocessing.ohe_transform(data, "Gender", ohe_gender)
    data = preprocessing.ohe_transform(data, "EverBenched", ohe_everbenched)

    # predict data
    y_pred = model_data.predict(data)

    # inverse transform
    label = ['Not Churn', 'Churn']
    y_pred = label[y_pred[0]]

    return {"res" : y_pred, "error_msg": ""}
    # return data


if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080, reload=True)
