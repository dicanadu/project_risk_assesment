import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(message)s')
logger = logging.getLogger()


###############Load config.json and get path variables
with open('/home/workspace/config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def make_reports(filename='testdata.csv'):
    path = os.path.join(test_data_path, filename)
    logger.info(f'Reading dataframe from {path}')
    df = pd.read_csv(path, index_col=0)
    X = df.drop('exited', axis=1)
    y_true = df['exited']
    preds = model_predictions(X)
    logger.info(f'Creating confusion matrix')
    cm = metrics.confusion_matrix(y_true, preds)
    disp = metrics.ConfusionMatrixDisplay(cm,
                                  display_labels=['0', '1'])
    disp.plot()
    plt.title("Confusion matrix")
    save_path = os.path.join(output_model_path, 'confusionmatrix.png')
    logger.info(f'Saving confusion matrix to {save_path}')
    plt.savefig(save_path)


if __name__ == '__main__':
    make_reports()
