import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, RandomizedSearchCV
from sklearn import metrics
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
import joblib
import os


# Set a constant random seed value for reproducibility
SEED = 20
np.random.seed(SEED)

def split_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """
    Splits the data into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    return X_train, X_test, y_train, y_test

def fit_model(pipe: Pipeline, X_train: pd.DataFrame,  y_train: pd.DataFrame) -> Pipeline:
    """
    Fits the model using the provided pipeline.
    """

    pipe.fit(X_train, y_train)

    return pipe

def predict_score(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
    """
    Predicts using the fitted model and calculates various metrics.
    """

    y_pred = pipe.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc

def fit_compare_multiple_pipelines(pipeline_dict: dict, X: pd.DataFrame,  y: pd.DataFrame, resampler: str) -> pd.DataFrame:
    """
    Fits and compares multiple pipelines on the data.
    """

    # Initiate list to store the results
    results = []

    #  Splitting the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    for model_name, pipeline in pipeline_dict.items():

        # Fitting model
        pipeline = fit_model(pipeline, X_train, y_train)

        # Predicting and getting metrics
        accuracy, precision, recall, f1, roc_auc = predict_score(pipeline, X_test, y_test)

        results.append({
                'Model_name': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'ROC auc': roc_auc,
                'Resampler': resampler
            })
        
    return pd.DataFrame(results).sort_values(by='F1', ascending=False)

def create_cols_trans(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Creates a column transformer with pipelines for numeric and categorical columns.
    """

    # numeric Pipeline
    num_pipeline = Pipeline(steps=[
        ('scale', MinMaxScaler())
    ])

    # categorical Pipeline
    cat_pipeline = Pipeline(steps=[
        ('onehotencoder', OneHotEncoder(drop='first'))
    ])

    # merging cat and num pipelines into a columnstransformer, so it can alter just specific columns
    col_trans = ColumnTransformer(transformers=[
        ('num_pipeline', num_pipeline, num_cols),
        ('cat_pipeline', cat_pipeline, cat_cols), 
        ],
        remainder='passthrough', #remainder=passthrough is specified to ignore other columns in a dataframe.
        n_jobs=-1 #n_job = -1 means using all processors to run in parallel.
    )

    return col_trans

def generate_pipelines(classifiers_list, base_pipeline) -> Pipeline:
    """
    Generates pipelines for a list of classifiers using a base pipeline.
    """

    # Dictionary to store pipelines
    pipelines_dict = {}

    for clf in classifiers_list:
        # Clone the base_pipeline to avoid modifying the original
        new_pipeline = clone(base_pipeline)
        
        # If the pipeline contains an RFECV step, set its estimator
        if 'rfecv' in new_pipeline.named_steps:
            new_pipeline.named_steps['rfecv'].estimator = clf
        
        # Modify the 'classifier' step in the steps list
        steps = list(new_pipeline.steps)
        steps[-1] = ('classifier', clf)
        new_pipeline.steps = steps
        
        # Add the new pipeline to the dictionary
        pipelines_dict[clf.__class__.__name__] = new_pipeline
    
    return pipelines_dict

def train_and_save_model(pre_tuned_pipelines_rfecv, X, y, save_dir="Model"):
    rfecv_model_results = []
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for model_name, pipeline in pre_tuned_pipelines_rfecv.items():
        # Splitting the data into training and testing set
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Using the fit_model function to fit the pipeline with recursive feature elimination
        pipeline = fit_model(pipeline, X_train, y_train)

        # Access the 'classifier' step of the pipeline
        rfecv_step = pipeline.named_steps['rfecv']

        # Get the mask of selected features
        selected_features_mask = rfecv_step.support_

        # Getting the features names from the transformer step
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

        # Filtering the names of the selected features using the mask
        selected_feature_names = feature_names[selected_features_mask]

        # Getting and printing the metrics for the new model
        accuracy, precision, recall, f1, roc_auc = predict_score(pipeline, X_test, y_test)

        rfecv_model_results.append({'Model_name': model_name,
                    'Features Selected': selected_feature_names,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'ROC auc': roc_auc,
                    'Model': 'rfecv'})
        
        # Create a filename based on the model name and its performance metrics
        filename = f"{model_name}_recall_{round(recall, 4)}_f1_{round(f1, 4)}.pkl"
        save_path = os.path.join(save_dir, filename)
        
        # Save the trained model using joblib
        joblib.dump(pipeline, save_path)    

    return pd.DataFrame(rfecv_model_results)