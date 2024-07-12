# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:25:14 2024

@author: gaelk
"""

import logging
import warnings

import numpy as np

import mlflow
import mlflow.sklearn


from datetime import datetime
import argparse

import os

from utils import read_data_on_disk, training_basic_classifier, predict_on_test_data, create_confusion_matrix_plot
from utils import predict_prob_on_test_data, get_metrics, signature_on_test_data

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


# Credential for GCP
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'refined-grammar-428414-f3-36c1a78b9de4.json'


# Inputs of the script
parser = argparse.ArgumentParser(
    description='Train a model and deploy it with MLflow')


parser.add_argument("--dataset", default="./MLOps_test", type=str,
                    help='Specify the folder where the dataset is')

parser.add_argument("--experiment", default="default", type=str,
                    help='Experiment name')

parser.add_argument('--max_iter', default=300, type=int, help='Maximum interation number')

parser.add_argument('--random_seed', default=42, type=int, help='random seed')

parser.add_argument('--author', default="Gael", type=str, help='Tag for the author')
parser.add_argument('--purpose', default="assignment", type=str, help='Tag for the purpose of model')

parser.add_argument('--tracking_uri', default="http://127.0.0.1:5000", type=str, help='Tracking URI')



args = parser.parse_args()

# Folder where the dataset is stored. By default ./MLOps_test
path = args.dataset

# Maximum number of iterations for the logistic regression. Just for the purpose of
# having some training parameters
max_iter = args.max_iter

# Random seed, just to have another parameter for the training
random_seed = args.random_seed

# Name of the experiment. On mlflow, run are grouped by experiment in a folder
experiment_name = args.experiment

# URL for the mlflow server
mlflow_tracking_uri = args.tracking_uri

# I created two tags to filter the runs in the database. We can have as much as needed
author = args.author   # author of the run
purpose = args.purpose  # purpose of the run


# For each mlflow run, we need a RUN NAME. 
run_name=experiment_name + "-" + str(datetime.now()).replace(":", "-")

# We insert the training parameters in a dict
run_params = {"max_iter": max_iter, "random_seed":random_seed}

# mlflow_tracking_uri = "http://34.163.11.217:5000"

logging.info(f"Tag author: {author}.")
logging.info(f"Tag purpose: {purpose}.")
logging.info(f"RUNID:  {run_name}.")
logging.info(f"Dataset stored in the folder {path}.")
logging.info(f"Parameters of training are  {run_params}.")
logging.info(f"Experiment name is {experiment_name}.")
logging.info(f"Tracking URI: {mlflow_tracking_uri}.")
logging.info("...............................")
logging.info("...............................")



if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    np.random.seed(random_seed)
    
    # We read and preprocess the data -- It is not done in a optimal way.
    # I just wanted to have a working pipeline
    
    X_train, y_train, X_val, y_val = read_data_on_disk(path, display=False)
    
    # We stop previous run
    mlflow.end_run()
    
    # We set up the name of the experiment
    mlflow.set_experiment(experiment_name)
    
    # We set up the URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    #mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name=run_name):
        
        
        model = training_basic_classifier(X_train,y_train, max_iter)
        
        y_pred = predict_on_test_data(model,X_val)
        
        signature = signature_on_test_data(X_val, y_pred)
        
        y_pred_prob = predict_prob_on_test_data(model,X_val) 
        
        run_metrics = get_metrics(y_val, y_pred, y_pred_prob)
        
        create_confusion_matrix_plot(model, X_val, y_val)
        
        print(f"run_metrics  {run_metrics}")
        
        
        # Logging training parameters
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
           
        # Logging metrics
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        
        # Save confusion matrix as an artefact
        mlflow.log_artifact(local_path='confusion_matrix.png', artifact_path='confusion_matrix')
            

        # Setting tags for this run
        mlflow.set_tags({"author":author, "purpose":purpose})
        
        # Logging the model
        result = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="artifacts-folder",
            signature=signature,
            registered_model_name="registered_model_name-sk-learn",
            )
        
