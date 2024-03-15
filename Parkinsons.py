import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('C:\\Users\\jashw\\OneDrive\\Desktop\\MDPS\\DataSets\\parkinsons.csv')

# Printing the first 5 rows of the dataframe
print(parkinsons_data.head())

# Number of rows and columns in the dataframe
print(parkinsons_data.shape)

# Getting more information about the dataset
print(parkinsons_data.info())

# Checking for missing values in each column
print(parkinsons_data.isnull().sum())

# Getting some statistical measures about the data
print(parkinsons_data.describe())

# Distribution of target Variable
print(parkinsons_data['status'].value_counts())

# Grouping the data based on the target variable, excluding the 'name' column
print(parkinsons_data.drop(columns=['name'], axis=1).groupby('status').mean())

X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = svm.SVC(kernel='linear')

# Training the SVM model with training data
model.fit(X_train, Y_train)

# Accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# Accuracy score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

input_data = [[122.40000, 148.65000, 113.81900, 0.00968, 0.00008, 0.00465, 0.00696, 0.01394, 0.06134,
               0.62600, 0.03134, 0.04518, 0.04368, 0.09403, 0.01929, 19.08500, 0.458359, 0.819521,
               -4.075192, 0.335590, 2.486855, 0.368674]]

# Changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's")

# Saving the trained model
filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Loading the saved model
loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))

for column in X.columns:
    print(column)
