
import pandas as pd
import numpy as np
import timeit
import os
import json
import joblib
import logging
import subprocess

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(message)s')
logger = logging.getLogger()
##################Load config.json and get environment variables
with open('/home/workspace/config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config["prod_deployment_path"])

##################Function to get model predictions
def model_predictions(dataframe):
    model_file = [file for file in os.listdir(prod_deployment_path) if file.endswith(".pkl")][0]
    logger.info(f"Loading model {model_file}")
    load_model = os.path.join(prod_deployment_path, model_file)
    model = joblib.load(load_model)
    logger.info("Getting predictions")               
    predictions = model.predict(dataframe)
    logger.info(f"Predictions resulted in {predictions}")
    return predictions

##################Function to get summary statistics
def dataframe_summary(filename='final_data.csv'):
    file_path = os.path.join(dataset_csv_path, filename)
    logger.info(f"Getting statistics from {file_path}")
    df = pd.read_csv(file_path, index_col = 0).set_index('corporation')
    numeric_columns = df.select_dtypes(include=np.number).columns
    output = []
    for col in numeric_columns:
        logger.info(f"Getting mean, median and std for {col}")
        output.append({col: {'mean' : df[col].mean(),
        'median': df[col].median(),
        'std': df[col].std()}})
    
    return output

##################Function to get nas
def missing_data(filename='final_data.csv'):
    file_path = os.path.join(dataset_csv_path, filename)
    logger.info(f"Getting statistics from {file_path}")
    df = pd.read_csv(file_path, index_col = 0).set_index('corporation')
    nas = (df.isna().sum() / len(df) * 100).to_dict()
    return nas

def execution_time():
    files = ['ingestion.py', 'training.py']
    total_time = []
    for file in files:
        logger.info(f"Calculating execution time for {file}")
        start = timeit.default_timer()
        subprocess.run(['python', file])
        total =  timeit.default_timer() - start
        logger.info(f"Total execution time of {file} was {total}")
        total_time.append({file:total}) 
    return total_time

##################Function to check dependencies
def outdated_packages_list():
    with open('requirements.txt', 'r') as file:
        content = file.readlines()
    packages = [package.split('==')[0] for package in content]
    versions = [package.split('==')[1].strip() for package in content]
    latest_version = []
    
    for package in packages:
        latest = subprocess.run(f'pip show {package} | grep "Version"',
                                shell = True, capture_output=True, text=True).stdout.strip()
        latest = latest.split(': ', 1)[-1]
        latest_version.append(latest)
    
    df = pd.DataFrame({'package': packages,
                       'current_version': versions,
                       'latest_version': latest_version})
    return df


if __name__ == '__main__':
    entry = {
    'lastmonth_activity' : [70, 50],
    'lastyear_activity' : [200, 10],
    'number_of_employees': [10, 1]
    }
    df = pd.DataFrame(entry)                    
    model_predictions(df)
    dataframe_summary()
    missing_data()
    #execution_time()
    print(outdated_packages_list())





    
