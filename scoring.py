from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import json
import logging


logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(message)s')
logger = logging.getLogger()

#################Load config.json and get path variables
with open('/home/workspace/config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    logger.info(f"Loading model data from {test_data_path}")
    load_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'),
                           index_col=0)
    X = load_data.drop('exited',axis=1)
    y = load_data['exited']
    
    logger.info(f"Loading model from {output_model_path}")
    model = joblib.load(os.path.join(output_model_path, 'trainedmodel.pkl'))
 
    logger.info(f"Scoring model f1_score")
    preds = model.predict(X)
    score = metrics.f1_score(y, preds)
    
    logger.info(f"F1 score is {score}")
    
    logger.info(f"Saving latest score to {output_model_path}")
    with open(os.path.join(output_model_path, 'latestscore.txt'), "w") as file:
        file.write(str(score))
    return score
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

if __name__ == '__main__':
    score_model()