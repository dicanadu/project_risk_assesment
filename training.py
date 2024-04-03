from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(message)s')
logger = logging.getLogger()

###################Load config.json and get path variables
with open('/home/workspace/config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
input_dataframe = os.path.join(config['output_concatenated_dataframe']) 

#################Function for training the model
def train_model(): 
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    logger.info("Loading data")
    load_data_path = os.path.join(dataset_csv_path, input_dataframe)
    data = pd.read_csv(load_data_path, index_col=1)
    columns = data.columns
    X = data.drop([columns[0], 'exited'], axis=1)
    y = data['exited']
    
    logger.info("Fitting model")
    model.fit(X,y)
    
    logger.info("score predictions")    
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, os.path.join(model_path,'trainedmodel.pkl'))
    
    return model
    

if __name__ == "__main__":
    train_model()
