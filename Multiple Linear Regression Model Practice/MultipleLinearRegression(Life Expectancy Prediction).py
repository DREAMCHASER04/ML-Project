import numpy as np
import pandas as pd

dataset = pd.read_csv('Life Expectancy Data.csv')
dataset = dataset.dropna()
X = dataset.iloc[:, dataset.columns != dataset.columns[3]].values #dataset.columns != dataset.columns[3] means all the columns except for the third column
Y = dataset.iloc[:, 3].values

#Convert the label data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#two different columns need two different label encoder
labelencoder_country = LabelEncoder()
labelencoder_status = LabelEncoder()
# Fit LabelEncoders
labelencoder_country.fit(dataset.iloc[:, 0].unique())  # Country
labelencoder_status.fit(dataset.iloc[:, 2].unique())  # Status
ct_0 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [0,2])], remainder='passthrough')
X = ct_0.fit_transform(X)

#Separate the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

#Train the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#Make Prediction
Y_pred = regressor.predict(X_test)



#Test Section:

# Define a test input based on your dataset's structure and preprocessing
# Format: Country,Year,Status,Adult Mortality,infant deaths,Alcohol,percentage expenditure,Hepatitis B,Measles , BMI ,under-five deaths ,Polio,Total expenditure,Diphtheria , HIV/AIDS,GDP,Population, thinness  1-19 years, thinness 5-9 years,Income composition of resources,Schooling
test_input = [['China', 2023, 'Developing', 200, 50, 5.5, 85.678, 90, 120, 25.4, 70, 90, 9.2, 90, 0.05, 1000.12345, 5000000, 10.5, 10.3, 0.800, 12.5]]

# Preprocess the test input to match the training data
# Step 1: Label encode categorical columns (Country and Status in this case)
test_input[0][0] = labelencoder_country.transform([test_input[0][0]])[0]  # Label encode 'Country'
test_input[0][2] = labelencoder_status.transform([test_input[0][2]])[0]  # Label encode 'Status'

# Step 2: Apply OneHotEncoder for categorical columns (using the existing ColumnTransformer)
test_input_transformed = ct_0.transform(test_input)

# Step 3: Make a prediction using the trained model
predicted_value = regressor.predict(test_input_transformed)

# Output the prediction
print("Predicted Life Expectancy:", predicted_value[0])




