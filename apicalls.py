import requests
import json
import os

with open('./config.json','r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

def api_report():
#Call each API endpoint and store the responses
    response1 = requests.post(f"{URL}/prediction", params = {"filename": 'testdata/testdata.csv'})
    response1 = response1.content.decode('utf-8')

    response2 = requests.get(f"{URL}/scoring")
    response2 = response2.content.decode('utf-8')

    response3 = requests.get(f"{URL}/summarystats")
    response3 = response3.content.decode('utf-8')

    response4 = requests.get(f"{URL}/diagnostics")
    response4 = response4.content.decode('utf-8')

    # #combine all API responses
    responses = [response1, response2, response3, response4]

    #write the responses to your workspace
    with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as file:
        file.write(str(responses))
