import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data read + split the dataset
dataset = pd.read_csv("Salary_dataset.csv")
X = dataset.iloc[ : ,  1 : 2 ].values #the first : means select all rows
Y = dataset.iloc[ : , 2 ].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)

#Train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#Predict the result
Y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(mse)
average_value = np.average(Y)
print(average_value)


'''
#visualize the training result
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.title("Training Results")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()  # Display the training plot

#visualize the test result
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.title("Test Results")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()  # Display the test plot
'''

