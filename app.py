from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import uvicorn

app=FastAPI()

model=joblib.load("models/model.joblib")

class PredictionInput(BaseModel):
    # Define the input parameters required for making predictions
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return "Welcome"

@app.post("/predict")
def main(input_data : PredictionInput):
    features = [input_data.Time,
                input_data.V1,
                input_data.V2,
                input_data.V3,
                input_data.V4,
                input_data.V5,
                input_data.V6,
                input_data.V7,
                input_data.V8,
                input_data.V9,
                input_data.V10,
                input_data.V11,
                input_data.V12,
                input_data.V13,
                input_data.V14,
                input_data.V15,
                input_data.V16,
                input_data.V17,
                input_data.V18,
                input_data.V19,
                input_data.V20,
                input_data.V21,
                input_data.V22,
                input_data.V23,
                input_data.V24,
                input_data.V25,
                input_data.V26,
                input_data.V27,
                input_data.V28,
                input_data.Amount
                ]
    prediction=model.predict([features])[0].item()
    return {'Preditction ':prediction}

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8080)
