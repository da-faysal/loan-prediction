import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


target_col = "Loan_Status"
df = pd.read_csv("Loan-Approval-Prediction.csv")
df[target_col] = df[target_col].map({"Y": 1, "N": 0})

cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

X = df[num_cols+cat_cols]
y = df[target_col]

# Preprocessing function
def preprocess_data():
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    return preprocessor