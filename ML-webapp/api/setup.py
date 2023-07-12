from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import json
from joblib import load

ARTIFICATS_DIR = '../ml/artifacts/'

app = FastAPI()
# TODO: enable using other models
model = load(ARTIFICATS_DIR + 'clf_v1.joblib')

# Taking care of serialization/deserializarion - Parser info
class user_input(BaseModel):
    satisfaction_level  : float
    last_evaluation     : float
    number_project      : int
    average_montly_hours: int
    time_spend_company  : int
    Work_accident       : int  
    promotion_last_5years: int
    department          : str
    salary              : str

def predict(X_test):
    department_enc = load(ARTIFICATS_DIR + 'department_enc.joblib')
    salary_enc = load(ARTIFICATS_DIR + 'salary_enc.joblib')
    print(X_test)
    df = pd.DataFrame(X_test)
    df['salary'] = salary_enc.transform(df['salary'])
    df['department'] = department_enc.transform(df['department'])
    print(df)
    preds = model.predict(df)
    probs = model.predict_proba(df)

    return preds, probs

def format_input(inp):
    return {
        'satisfaction_level' : [inp.satisfaction_level],
        'last_evaluation' : [inp.last_evaluation], 
        'number_project' : [inp.number_project],
        'average_montly_hours' : [inp.average_montly_hours], 
        'time_spend_company' : [inp.time_spend_company],
        'Work_accident' : [inp.Work_accident], 
        'promotion_last_5years' : [inp.promotion_last_5years],
        'department' : [inp.department],
        'salary' : [inp.salary]
    }

@app.post('/predict')
async def func(inp:user_input):
    data = format_input(inp)
    preds, probs = predict(data)
    print(preds)
    print(probs)
    output = {
        'prediction':int(preds[0]),
        'probability':float(probs[0][1])}
    return json.dumps(output)

@app.get('/')
async def welcome():
    return f'Welcome to HR api'
