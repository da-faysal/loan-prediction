from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from preprocess import preprocess_data, X, y
from train import train_and_validate
from optimize_hyperparameters import optimize
import joblib

# Define models
models = {
        "XGBoost":XGBClassifier(random_state=42),
        "LightGBM":LGBMClassifier(random_state=42, verbose=-1)}

# Preprocess the data
preprocessor = preprocess_data()

# Cross validation to find the best model
best_name = train_and_validate(preprocessor, X, y, models)

# Getting the best model 
best_model = models[best_name]

# Optimized hyperparameters of the best model and use that optimized parameters 
best_params, pipeline = optimize(preprocessor, X, y, best_model)
pipeline.set_params(**best_params)

# Get the best model from the pipeline
best_model = pipeline.named_steps['classifier']

# Sanity check to see if the hyperparameters optimization worked on initial cross validation
train_and_validate(preprocessor, X, y, best_model)

# Save the final model with fine-tuned parameters
pipeline.fit(X, y)

# Save the pipeline
joblib.dump(pipeline, "pipeline.joblib")