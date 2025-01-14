
import numpy as np
import pandas as pd


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
X[ : , 3] = labelencoder_x.fit_transform(X[ : , 3])
#Linear regression/logisitc regression/SVM/KNN need Onehotencoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
X = X[: , 1:] #Avoid the dummy variable trap to modify the dummy variable (redundancy in the dataset) --> general strategy to eliminate a column

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

#Test Validation
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(Y_test, Y_pred)



#Input section



new_input = [[160000, 130000, 300000, 'New York']]

# Preprocess the 'State' column (label encode)
new_input[0][3] = labelencoder_x.transform([new_input[0][3]])[0]
# Apply one-hot encoding using the ColumnTransformer
new_input_transformed = ct.transform(new_input)
# Avoid the dummy variable trap (drop the first column of one-hot encoding)
new_input_transformed = new_input_transformed[:, 1:]
# Predict the profit
predicted_profit = regressor.predict(new_input_transformed)
print("Predicted Profit:", predicted_profit[0])





