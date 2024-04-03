import training
import pandas as pd
import json
import scoring
import ingestion
import deployment
import diagnostics
import reporting
import logging
import apicalls
import os
import ast
import joblib
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logging.info('Getting config params and paths')

##################Getting paths and configuration
with open('./config.json', 'r') as file:
    config = json.load(file)

input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
deployed_text_path = os.path.join(prod_deployment_path, 'ingestedata.txt')
deployed_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
deployed_model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
ingested_data_path = os.path.join(config['output_folder_path'], 'final_data.csv')
source_data = sorted(os.listdir(input_folder_path))

##################Check and read new data
logging.info('Reading deployed files')

with open(deployed_text_path, 'r') as file:
    content = file.read()
    content = sorted(ast.literal_eval(content))

with open(deployed_score_path, 'r') as file:
    score = float(file.read())

############## Define model drift function
def checking_model_drift():
    model = joblib.load(deployed_model_path)
    data = pd.read_csv(ingested_data_path, index_col=0).set_index('corporation')
    X = data.drop(['exited'], axis=1)
    y = data['exited']
    preds = model.predict(X)
    new_score = f1_score(y, preds)
    return new_score

############## Define full process
def main():
    try:
        #Decide Step 1
        logger.info(f"Current data is {content}")
        if content != source_data:
            logger.info(f"Ingesting new data {source_data}")
            ingestion.merge_multiple_dataframe() #Ingesteddata
            new_score = checking_model_drift()
            logger.info(f"Latest score was: {score}, and new_score: {new_score}")
            #Decide step 2
            if new_score > score:
                training.train_model() #scores trainedmodel.pkl
                scoring.score_model() #scores latestscore.txt
                deployment.store_model() #files to production_deployment
                reporting.make_reports() #confusionmatrix2.png
                apicalls.api_report()
            else:
                logger.info("Stop step 2: Current model is accurate")
        else:
            logger.info("Stop step 1: Files already analyzed")
    except Exception as e:
        logger.info(e)


if __name__ == '__main__':
    main()
