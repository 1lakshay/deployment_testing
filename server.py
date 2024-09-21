from fastapi import FastAPI
import joblib
import numpy as np

# loading the saved ML model
model = joblib.load('model.joblib')

# all feature names
# to print hename of output as with the position of the predicted value
class_names = np.array(['setosa','versicolor','virginica'])

app = FastAPI()    # initialization

@app.get('/')    # endpoint
def reed_root():
    return {'message': 'Iris model API'}

@app.post('/predict')    # to send DATA | endpoint for sending DATA
def predict(data: dict):   # function takes in the dictionary datatype
    features = np.array(data['features']).reshape(1,-1)
    prediction = model.predict(features)
    # [prediction] :: the model will return the array like array([0])
    # not the single int even for the single prediction not like: 0, 1
    # class_name[prediction] for prediction = array[0] will return
    # ['setosa'] so for getting 'setosa' we used [prediction][0]
    class_name = class_names[prediction][0]

    return {'predicted_class': class_name}


