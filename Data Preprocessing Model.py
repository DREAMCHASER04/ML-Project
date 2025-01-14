import numpy as np
import pandas as pd

#Extract data from the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values

#Handling the Missing data

from sklearn.impute import SimpleImputer
# imputer is used to handle the missing value, and replace them with specific target value)

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
# the missing_value represented as NaN in doc,
# and replace the missing values by the mean of the column,
# axis = 0 means the imputation is performed column-wise
imputer = imputer.fit(X[ : , 1:3])
#Select all rows and columns 1 and 2, excluding column 3
# fit method computes the required statistic (the above mean) --> stores in the object for later transformation
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
# transform the mean values computed during the fit step

#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#convert categorical labels into a one-hot encoded format (0,1,2, into [1,0,0],[0,1,0],[0,0,1])
labelencoder_X = LabelEncoder() #creating an instance of the labelencoder assigns it to the variable
#convert label value or categorical value as numeric values
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#transform the data and convert only the first column data

#Creating a dummy variable --> change the original blue to (1,0,1) this is the dummy variable
from sklearn.compose import ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [0])  # Specify column index 0 for one-hot encoding
    ],
    remainder='passthrough'  # Leave other columns as they are
)
X = column_transformer.fit_transform(X)
#convert categorical labels into a one-hot encoded format (0,1,2, into [1,0,0],[0,1,0],[0,0,1])
# label encoder convert the labels into integer while the onehot encoder convert the labels into binary numbers 0 or 1
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)





