# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 01:44:57 2024

@author: gaelk
"""

import logging


import pandas as pd


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score,recall_score,log_loss

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def read_data_on_disk(path, display=False):
    
    train_data = pd.read_csv(path + '/data.csv')
    val_data = pd.read_csv(path + '/validation_data.csv')
    
    label_encoder = LabelEncoder()
    label_encoders = {}
    
    categorical_features = ['Cloud Cover', 'Season', 'Location', 'Weather Type']

    for feature in categorical_features:
        le = LabelEncoder()
        train_data[feature] = label_encoder.fit_transform(train_data[feature])
        val_data[feature] = label_encoder.transform(val_data[feature])
        label_encoders[feature] = le
    
    X_train = train_data.drop('Weather Type', axis=1)
    X_val = val_data.drop('Weather Type', axis=1)
    
    # We drop the firsst column that contains only index
    X_train.drop(X_train.columns[0], axis=1, inplace=True)
    X_val.drop(X_val.columns[0], axis=1, inplace=True)
    
    y_train = train_data['Weather Type']
    y_val = val_data['Weather Type']
    
        
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    
    # Sauvegarder le scaler pour une utilisation future
    # joblib.dump(scaler, 'scaler.pkl')
    
    
    # Tio avoid the error below, we change data type to numpy array
    # WARNING mlflow.sklearn: Failed to log training dataset information to MLflow Tracking. Reason: 'Series' object has no attribute 'flatten'
    # https://stackoverflow.com/questions/53319865/series-object-has-no-attribute-flatten-when-i-use-y-flatten
    
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    return X_train, y_train, X_val, y_val




def training_basic_classifier(X_train,y_train, max_iter=300):
    classifier = LogisticRegression(max_iter=max_iter)
    classifier.fit(X_train,y_train)  
    return classifier


def signature_on_test_data(X_val, y_pred):
    signature = infer_signature(X_val, y_pred)
    return signature


def predict_on_test_data(model,X_val):
    y_pred = model.predict(X_val)
    return y_pred

def predict_prob_on_test_data(model,X_val):
    y_pred = model.predict_proba(X_val)
    return y_pred

def get_metrics(y_val, y_pred, y_pred_prob):
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred,average='micro')
    recall = recall_score(y_val, y_pred,average='micro')
    entropy = log_loss(y_val, y_pred_prob)
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}

def create_roc_auc_plot(clf, X_data, y_data):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    metrics.plot_roc_curve(clf, X_data, y_data) 
    plt.savefig('roc_auc_curve.png')
    
def create_confusion_matrix_plot(clf, X_val, y_val):
    ConfusionMatrixDisplay.from_estimator(clf,X_val,y_val)
    plt.savefig('confusion_matrix.png')
    
    
def create_experiment(tracking_uri, experiment_name,run_name, run_metrics,model, confusion_matrix_path = None, 
                      roc_auc_plot_path = None, run_params=None):
    
    mlflow.end_run()
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        
        
        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')
            
        if not roc_auc_plot_path == None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
        
 
        mlflow.sklearn.log_model(model, experiment_name + "_" + run_name + "_" + "model_")
        

        
        
        # Test if the new model performs better than the one in production
        # if new_accuracy >= prod_accuracy:
            # client.transition_model_version_stage(name, to_prod_version, "Production")
        #     deploy the model in production
        

        
        
        joblib.dump(model, experiment_name + "_" + run_name + "_" + "pkl_model" + ".pkl")
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))