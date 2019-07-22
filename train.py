import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder

embedding_file = open("embedding/embedding_1563463132.pkl","rb")
embedding_dict = pickle.load(embedding_file)

X = []
Y = []

for key, value in embedding_dict.items():
    Y.append(key)
    X.append(value)

X = np.array(X)
Y = np.array(Y)

le = LabelEncoder().fit(Y)
Y = le.transform(Y)

print (X.shape)
print (Y)
