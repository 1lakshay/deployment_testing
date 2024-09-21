"""
FILE just to send data to API & check whether it is working fine or not
"""

import json
import requests

data = {'features': [1,2,3,4]}    # data that is to be sent to the API

url = "http://127.0.0.1:8000/predict/"   # endpoint location

data = json.dumps(data)                # converting data into json
response = requests.post(url, data)   # to send the data
print(response.json())
