import os
import sys
import shutil

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures,StandardScaler,QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold

import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    auc = roc_auc_score(actual, pred)
    
    return accuracy,f1,precision,recall,auc


def clean_data(dataframe):
    def set_index(df):
        return df
    
    def drop_columns(df):
        return df
    
    def drop_missing_rows(df):
        return df
    
    clean_dataframe=dataframe.pipe(set_index)\
                             .pipe(drop_columns)\
                             .pipe(drop_missing_rows)
    
    return clean_dataframe


def load_data(path):
    print(":::::::LOADING DATA:::::::::::")
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        logger.exception("Unable to download training CSV, check path. Error: %s", e)
        
path_train="../data/processed/Train.csv"
path_test="../data/processed/Test.csv"


if __name__ == "__main__":
    print(":::::::RUN TRAIN::::::::::::::")
    #np.random.seed(40)
    os.makedirs("train_files", exist_ok=True)
    shutil.copy('03_train.py', 'train_files/train.py')

    train_df = load_data(path_train)
    test_df = load_data(path_test)

    X_train = train_df.drop("diabetes_mellitus",axis=1)
    y_train = train_df["diabetes_mellitus"]
    
    X_test = test_df.drop("diabetes_mellitus",axis=1)
    y_test = test_df["diabetes_mellitus"]
    
#    mlflow.create_experiment("WiDS2021")
    mlflow.set_experiment("WiDS2021")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        
        numeric_features = X_train.select_dtypes("number").columns
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                                              ('scaler', StandardScaler())
                                     ]
                              )

        categorical_features = X_train.select_dtypes("category").columns
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                                                  ('encoder', OneHotEncoder(drop="first",handle_unknown='error')),
                                                 ]
                                          )

        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                       ('cat', categorical_transformer, categorical_features)
                                                      ]
                                         )

        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', DummyClassifier(strategy="stratified"))
                             ]
                       )
        
        
        best_clf = clf.fit(X_train, y_train)

        predictions = best_clf.predict(X_test)

        (accuracy,f1,precision,recall,auc) = eval_metrics(y_test, predictions)

        print("Accuracy : {:.3f}".format(accuracy))
        print("F1       : {:.3f}".format(f1))
        print("Precision: {:.3f}".format(precision))
        print("Recall   : {:.3f}".format(recall))
        print("AUC      : {:.3f}".format(auc))

        mlflow.log_artifacts("train_files")
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("AUC", auc)