import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, root_validator
from typing import Literal


# Load the saved model pipeline
pipeline = joblib.load("pipeline.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic BaseModel
class ModelInput(BaseModel):
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int
    Loan_Amount_Term: int
    Credit_History: Literal[0, 1]
    Gender: Literal["male", "female"]
    Married: Literal["yes", "no"] 
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["graduate", "not graduate"] 
    Self_Employed: Literal["yes", "no"] 
    Property_Area: Literal["rural", "urban", "semiurban"]


# Convert relevant fields to lowercase and strip whitespace using a root validator
    @root_validator(pre=True)
    def to_lowercase_and_strip(cls, values):
        for field in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip().lower()  # Strip whitespace and convert to lowercase
        return values


@app.post("/predict/")
def predict(data: ModelInput):
    try:
        # Convert the input data to a DataFrame (wrap the dict in a list)
        input_df = pd.DataFrame([data.dict()])
        
        # Use the pipeline to make predictions
        pred = pipeline.predict(input_df)[0]
        if pred==1:
            # Return the prediction
            return {"prediction": "Eligible for Loan"}
        else:
            return {"prediction": "Not Eligible for Loan"}
    
    except Exception as e:
        return {"error": str(e)}