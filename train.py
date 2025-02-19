from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
import time

def train_and_validate(preprocessor, X, y, models):
    best_score = -float('inf')
    best_pipeline = None
    best_name = None

    if isinstance(models, dict):
        for name, model in models.items():

            # Create a pipeline for each model with the same preprocessor
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            # Compute cross-validation scores (using f1 score here)
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
            mean_score = scores.mean()
            print(f"{name} CV mean f1 score: {mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_pipeline = pipeline
                best_name = name

        print(f"\nBest Model: {best_name} with a CV f1 Score of {best_score:.4f}")
        return best_name

    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', models)
        ])
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
        mean_score = scores.mean()
        print(f"CV Mean f1 Score after Hyperparameters Opt: {mean_score:.4f}")