import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

mfccs = np.load('data/processed/mfccs/mfccs.npy', allow_pickle=True)
labels = np.load('data/processed/labels/labels.npy')

shapes = np.genfromtxt('data/processed/shapes/mfccs.csv', delimiter=",", dtype=int)
reshaped_mfccs = [np.reshape(mfccs[i], (4000)) for i in range(len(mfccs))]

# print(reshaped_mfccs)

X_train, X_test, y_train, y_test = train_test_split(reshaped_mfccs, labels)
# scaler = StandardScaler().fit(list(X_train))
# X_scaled = scaler.transform(X_train)

clf = SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

accuracy = clf.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')