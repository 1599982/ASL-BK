# train.py
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

DATA_DIR = "dataset"
X, y = [], []

for label in os.listdir(DATA_DIR):
    for file in os.listdir(os.path.join(DATA_DIR, label)):
        data = np.load(os.path.join(DATA_DIR, label, file))
        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

with open("asl_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Modelo entrenado y guardado como asl_model.pkl")
