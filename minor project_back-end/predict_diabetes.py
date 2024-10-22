import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Separate the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Take input data from command-line arguments
input_data = [float(arg) for arg in sys.argv[1:]]

# Convert the input data to a numpy array and reshape for prediction
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_as_numpy_array)

# Make the prediction
prediction = classifier.predict(std_data)

# Output the result
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')


# import sys
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import accuracy_score

# # loading the diabetes dataset into pandas
# diabetes_dataset = pd.read_csv('diabetes.csv')

# # separating the data and labels
# X = diabetes_dataset.drop(columns='Outcome', axis=1)
# Y = diabetes_dataset['Outcome']

# # data standardization
# scaler = StandardScaler()
# scaler.fit(X)

# # Train-test split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# # Training the SVM classifier
# classifier = svm.SVC(kernel='linear')
# classifier.fit(X_train, Y_train)

# # Fetch input data from the Node.js API call
# input_data = sys.argv[1:]

# # Convert input data to numpy array
# input_data = np.array(input_data, dtype=float).reshape(1, -1)

# # Convert input data into a DataFrame with the same column names as the original data
# input_data_df = pd.DataFrame(input_data, columns=X.columns)

# # Standardize the input data
# std_data = scaler.transform(input_data_df)

# # Perform the prediction
# prediction = classifier.predict(std_data)

# # Output the result
# if prediction[0] == 0:
#     print("The person is not diabetic")
# else:
#     print("The person is diabetic")

