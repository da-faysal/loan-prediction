from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Hyperparameter distribution for LGBM
param_dist = {
    'classifier__max_depth': [5, 10, -1],
    'classifier__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'classifier__n_estimators': [50, 80, 90, 100, 200, 500],
    "classifier__random_state" : [43]
}

def optimize(preprocessor, X, y, model):
    # Create a pipeline with the preprocessor and LGBM model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Use RandomizedSearchCV for hyperparameter optimization
    randomized_search = GridSearchCV(pipeline, param_dist, cv=5, scoring='f1', verbose=1, n_jobs=-1)
    randomized_search.fit(X, y)

    # Get the best model and parameters
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Score after Hyperparameters Opt: {best_score}")
    return best_params, pipeline