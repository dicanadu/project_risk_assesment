from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import model_predictions, dataframe_summary
from diagnostics import missing_data, execution_time, outdated_packages_list
from scoring import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    filename=request.args.get('filename')
    dataset_path = os.path.join(filename)
    df = pd.read_csv(dataset_path, index_col=0)
    X = df.drop('exited', axis=1)
    preds = model_predictions(X)
    #call the prediction function you created in Step 3
    return jsonify(preds.tolist())

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    score = score_model()
    return str(score)

# #######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    filename = request.args.get('filename', 'final_data.csv')
    result = dataframe_summary(filename)
    result_json = jsonify(result)
    return result_json

# #######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    nas = missing_data()
    time = execution_time()
    dependencies = outdated_packages_list().to_dict()
    result = {'missing_data': nas, 'execution_time': time, 'dependencies': dependencies}
    return jsonify(result)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
