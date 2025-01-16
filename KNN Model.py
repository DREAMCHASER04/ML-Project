import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # the sklearn only have metric that equals to minkowski, and when p = 2, it will be euclidean method.
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# Define a new data point (e.g., Age = 30, Estimated Salary = 50000)
new_data = np.array([[40, 30000]])

# Scale the new data point using the previously fitted StandardScaler
new_data_scaled = sc.transform(new_data)

# Predict the class of the new data point
predicted_class = classifier.predict(new_data_scaled)

if predicted_class[0] == 0:
    print(f"The predicted class for the input data is: Not Purchase")
else:
    print(f"The predicted class for the input data is: Purchased")

