import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, RandomizedSearchCV
from sklearn import metrics

# Set a constant random seed value for reproducibility
SEED = 20
np.random.seed(SEED)

def fit_model_with_cv(model, X, y, scoring_metrics=['recall', 'f1', 'accuracy', 'precision']):
    """
    Function to fit a model using Stratified K-Fold cross-validation.
    
    Parameters:
    - model: The machine learning model to be trained.
    - X: Features.
    - y: Target variable.
    - scoring_metrics: List of metrics for evaluation.
    
    Returns:
    - cv: Cross-validation results.
    """
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = cross_validate(model, X, y, cv=sk, scoring=scoring_metrics)
    return cv

def get_metrics_from_cross_validate(cv):
    """
    Extract metrics from the cross-validation results.
    
    Parameters:
    - cv: Cross-validation results.
    
    Returns:
    - Tuple containing average fit time, recall, f1, accuracy, and precision.
    """
    return cv['fit_time'].mean(), cv['test_recall'].mean(), cv['test_f1'].mean(), cv['test_accuracy'].mean(), cv['test_precision'].mean()

def fit_multiple_models_with_cv(models, X, y):
    """
    Fit multiple models using cross-validation and return their performances.
    
    Parameters:
    - models: List of machine learning models to be trained.
    - X: Features.
    - y: Target variable.
    
    Returns:
    - DataFrame containing performance metrics of each model.
    """
    results = []
    for model in models:
        cv = fit_model_with_cv(model, X, y)
        fit_time, avg_recall, avg_f1, avg_accuracy, avg_precision = get_metrics_from_cross_validate(cv)
        results.append({
            'Model': model.__class__.__name__,
            'Fitting Time': fit_time,
            'Average Recall': avg_recall,
            'Average F1': avg_f1,
            'Average Accuracy': avg_accuracy,
            'Average Precision': avg_precision
        })
    return pd.DataFrame(results).sort_values(by='Average Recall', ascending=False)

def fit_model(model, X, y):
    """
    Train a model on a training set and evaluate it on a test set.
    
    Parameters:
    - model: The machine learning model to be trained.
    - X: Features.
    - y: Target variable.
    
    Returns:
    - The trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    return model

# Define a dictionary of scorers for model evaluation
scorers = {'accuracy': 'accuracy', 'recall': 'recall', 'precision': 'precision', 'f1': 'f1'}

def randomized_grid_search_wrapper(model, X, y, param_grid, n_iter=10, refit_score='recall'):
    """
    Fit a model using RandomizedSearchCV for hyperparameter tuning.
    
    Parameters:
    - model: The machine learning model to be trained.
    - X: Features.
    - y: Target variable.
    - param_grid: Dictionary containing hyperparameters to be tested.
    - n_iter: Number of parameter settings that are sampled.
    - refit_score: Metric to refit the estimator.
    
    Returns:
    - grid_search: Fitted RandomizedSearchCV object.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_search = RandomizedSearchCV(model, param_grid, scoring=scorers, refit=refit_score, n_iter=n_iter, 
                                     cv=skf, return_train_score=True, n_jobs=-1, random_state=SEED)
    grid_search.fit(X, y)
    return grid_search
