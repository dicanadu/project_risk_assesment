from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(message)s')
logger = logging.getLogger()

##################Load config.json and correct path variable
with open('/home/workspace/config.json','r') as f:
    config = json.load(f) 
dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

####################function for deployment
def store_model(model='trainedmodel.pkl'):
    logger.info(f"Erasing model to production path: {prod_deployment_path}")
    for file in os.listdir(prod_deployment_path):
        if file.endswith('.pkl'):
            logger.info(f"Removing {file}")
            os.remove(os.path.join(prod_deployment_path, file))
    
    logger.info(f"Copying new information to production path: {prod_deployment_path}")
    latest_files_info = os.path.join(dataset_csv_path, 'ingestedata.txt')
    latest_model_info = os.path.join(output_model_path, 'latestscore.txt')
    latest_model_pkl = os.path.join(output_model_path, model)
    shutil.copy(latest_files_info, prod_deployment_path)
    shutil.copy(latest_model_info, prod_deployment_path)
    shutil.copy(latest_model_pkl, prod_deployment_path)
    pass

        
if __name__ == "__main__":
    store_model()

    
        
        

