import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

SEED = 20
np.random.seed(SEED)

def fit_model_with_cv(model, X, y, scoring_metrics = ['recall', 'f1', 'accuracy', 'precision']):
    cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=SEED)
    cv = cross_validate(model, X, y, cv=5, scoring=scoring_metrics)

    return cv

def get_metrics_from_cross_validate(cv):
    fit_time = cv['fit_time'].mean()
    avg_recall = cv['test_recall'].mean()
    avg_f1 = cv['test_f1'].mean()
    avg_accuracy = cv['test_accuracy'].mean()
    avg_precision = cv['test_precision'].mean()

    return fit_time, avg_recall, avg_f1, avg_accuracy, avg_precision

def fit_multiple_models_with_cv(models, X, y):
    results = []

    for model in models:
        cross_validate = fit_model_with_cv(model, X, y)
        fit_time, avg_recall, avg_f1, avg_accuracy, avg_precision = get_metrics_from_cross_validate(cross_validate)

        # Append results to the list
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Acurácia:",metrics.accuracy_score(y_test, y_pred))
    print("Precisão:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred)) 
    print("F1:",metrics.f1_score(y_test, y_pred))

    return model


scorers = {'accuracy': 'accuracy', 'recall': 'recall', 'precision': 'precision',' f1': 'f1'}

def randomized_grid_search_wrapper(model, X, y, param_grid, n_iter=10, refit_score='recall'):
    """
    fits a RandomizedSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_search = RandomizedSearchCV(model, param_grid, scoring=scorers, refit=refit_score, n_iter=n_iter, 
                           cv=skf, return_train_score=True, n_jobs=-1, random_state=SEED)
    grid_search.fit(X, y)

    # make the predictions
    y_pred = grid_search.predict(X)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(metrics.confusion_matrix(y, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search