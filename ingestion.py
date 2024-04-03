import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(message)s')
logger = logging.getLogger()

#############Load config.json and get input and output paths
with open('./config.json','r') as f:
    config = json.load(f)

#home_path = config["home_path"]
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
filename = config['output_concatenated_dataframe']

#############Function for data ingestion
def merge_multiple_dataframe():
    df = pd.DataFrame()
    for file in os.listdir(input_folder_path):
        ingestion_path = os.path.join(input_folder_path, file)
        logger.info(f"Taking ingestion from {ingestion_path}")
        data = pd.read_csv(ingestion_path)
        df = pd.concat([df,data])
    df = df.drop_duplicates()
    logger.info(f"Final shape is {df.shape}")

    save_path = os.path.join(output_folder_path)
    os.makedirs(os.path.join(save_path), exist_ok=True)
    logger.info(f"Saving dataframe to {save_path}/{filename}")
    df.to_csv(os.path.join(save_path, filename))


    with open(os.path.join(save_path, 'ingestedata.txt'), 'w') as file:
        file.write(str(os.listdir(input_folder_path)))

    return df

#check for datasets, compile them together, and write to an output file

if __name__ == '__main__':
    df = merge_multiple_dataframe()
