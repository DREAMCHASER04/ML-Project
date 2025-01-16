import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('Heart_disease_prediction.csv')
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(class_weight='balanced', random_state=0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, Y_train, cv=5, scoring='f1')
print("Mean F1-Score:", np.mean(scores))