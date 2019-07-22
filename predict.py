import pickle
import os
import glob
import pandas as pd
from sklearn.metrics import mean_squared_error

list_of_files = glob.glob('embedding/*')
latest_file = max(list_of_files, key=os.path.getctime)
print ("Loading embedding:", latest_file)
embedding_file = open(latest_file,"rb")
embedding_dict = pickle.load(embedding_file)

def getMSE(a, b):
    return mean_squared_error(a, b)

def getPrediction(embedding):
    """
        For now return min error person as prediction. 
        Later on we will train the model from embedding and use that model for prediction.
    """
    name = []
    error = []
    
    for key, value in embedding_dict.items():
        name.append(key)
        error.append(getMSE(value, embedding))
    
    result = pd.DataFrame({'name': name, 'error': error}).sort_values('error').iloc[:5]
    print (result)
