# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
%matplotlib inline


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load the training dataset
train_df = pd.read_csv('/kaggle/input/who-is-the-real-winner/train.csv')

# /Display the first few rows of the training dataset
print(train_df.head(20))


train_df.info()

train_df['state'].nunique()

train_df['Party'].nunique()

# Get the count of unique values in the 'Constituency ∇' column
train_df['Constituency ∇'].nunique()

# Define a function to convert amounts to rupees
def convert_to_rupees(amount):
    if isinstance(amount, str):
        if 'Crore' in amount:
            return float(amount.split()[0]) * 1e7  # 1 crore = 10^7 rupees
        elif 'Lac' in amount:
            return float(amount.split()[0]) * 1e5  # 1 lakh = 10^5 rupees
        elif 'Thou' in amount:
            return float(amount.split()[0]) * 1e3  # 1 Thousand = 10^3 rupees
        elif 'Hund' in amount:
            return float(amount.split()[0]) * 1e2  # 1 Thousand = 10^2 rupees
    return float(amount)

# Apply the conversion function to the 'Total Assets' column
train_df['Total Assets'] = train_df['Total Assets'].apply(convert_to_rupees)
train_df['Liabilities'] = train_df['Liabilities'].apply(convert_to_rupees)
train_df['Total Assets']=train_df['Total Assets']/1e5
train_df['Liabilities']=train_df['Liabilities']/1e5

print(train_df.head(20))

train_df.describe()

plt.hist(train_df['Total Assets'], bins=500,edgecolor='Black')

# Count the number of people with assets below 100
num_people_assets_below_100 = len(train_df[train_df['Total Assets'] > 500])

print("Number of people with assets below 100:", num_people_assets_below_100)

# Define the bins for different asset categories
bins = [0, 10, 100, 500, float('inf')]  # Define the bins: [0-10], (10-100], (100-500], (500, inf)

# Define the labels for the categories
labels = [1, 2, 3, 4]  # Label encode as 1, 2, 3, 4

# Create a new column 'asset_category' based on the bins
train_df['Total Assets'] = pd.cut(train_df['Total Assets'], bins=bins, labels=labels, right=False)

# Convert the 'asset_category' column to numeric if necessary
train_df['Total Assets'] = pd.to_numeric(train_df['Total Assets'])

# Check the first few rows of the DataFrame
print(train_df.head())

plt.hist(train_df['Total Assets'], bins=40,edgecolor='Black')

# Count the number of people with assets below 100
num_people_assets_below_100 = len(train_df[train_df['Liabilities'] < 1])

print("Number of people with assets below 100:", num_people_assets_below_100)

# Define the bins for different asset categories
bins = [0, 10, 50, 100, float('inf')]  # Define the bins: [0-10], (10-100], (100-500], (500, inf)

# Define the labels for the categories
labels = [1, 2, 3, 4]  # Label encode as 1, 2, 3, 4

# Create a new column 'asset_category' based on the bins
train_df['Liabilities'] = pd.cut(train_df['Liabilities'], bins=bins, labels=labels, right=False)

# Convert the 'asset_category' column to numeric if necessary
train_df['Liabilities'] = pd.to_numeric(train_df['Liabilities'])

# Check the first few rows of the DataFrame
print(train_df.head())

train_df = train_df[['Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state','Education']]



train_df['Education'].nunique()

train_df['Education'].unique()

train_df['Total Assets']

train_df['Liabilities']

# Encode categorical variables
label_encoder = LabelEncoder()
train_df['Party'] = label_encoder.fit_transform(train_df['Party'])
train_df['state'] = label_encoder.fit_transform(train_df['state'])
train_df['Total Assets'] = label_encoder.fit_transform(train_df['Total Assets'])
train_df['Liabilities'] = label_encoder.fit_transform(train_df['Liabilities'])
# Convert 'Education' to numerical categories
# Define a mapping dictionary for label encoding

# Select features and target variable
X = train_df[['Criminal Case', 'Total Assets', 'Liabilities', 'state']]
y = train_df['Education']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_train


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# One-hot encode categorical variables
#X_train_encoded = pd.get_dummies(X_train, columns=['Party', 'Constituency ∇', 'state'])

# One-hot encode categorical variables in y_train
#y_train_encoded = pd.get_dummies(y_train)

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
# Perform grid search cross-validation after handling missing values
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_rf_classifier = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Train Random Forest classifier with best parameters
best_rf_classifier.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_rf_classifier.predict(X_val)

# Calculate F1-score
f1 = f1_score(y_val, y_pred, average='weighted')
print("F1-score on validation set:", f1)

sns.jointplot(x='Total Assets', y='Criminal Case', data=train_df,color='white',marginal_kws={'color': 'blue'},edgecolor='Purple')

sns.jointplot(data=train_df, x='Party',y='Criminal Case',color='Green')

sns.jointplot(data=train_df, x='Party',y="Total Assets",color='Green')

# Load the test dataset
test_df = pd.read_csv('/kaggle/input/who-is-the-real-winner/test.csv')
test_df.head()

# Define a function to convert amounts to rupees
def convert_to_rupees(amount):
    if isinstance(amount, str):
        if 'Crore' in amount:
            return float(amount.split()[0]) * 1e7  # 1 crore = 10^7 rupees
        elif 'Lac' in amount:
            return float(amount.split()[0]) * 1e5  # 1 lakh = 10^5 rupees
        elif 'Thou' in amount:
            return float(amount.split()[0]) * 1e3  # 1 Thousand = 10^3 rupees
        elif 'Hund' in amount:
            return float(amount.split()[0]) * 1e2  # 1 Thousand = 10^2 rupees
    return float(amount)

# Apply the conversion function to the 'Total Assets' column
test_df['Total Assets'] = test_df['Total Assets'].apply(convert_to_rupees)
test_df['Liabilities'] = test_df['Liabilities'].apply(convert_to_rupees)
test_df['Total Assets']=test_df['Total Assets']/1e5
test_df['Liabilities']=test_df['Liabilities']/1e5

# Encode categorical variables
label_encoder = LabelEncoder()
test_df['Party'] = label_encoder.fit_transform(test_df['Party'])
test_df['state'] = label_encoder.fit_transform(test_df['state'])
test_df['Total Assets'] = label_encoder.fit_transform(test_df['Total Assets'])
test_df['Liabilities'] = label_encoder.fit_transform(test_df['Liabilities'])
# Select features and target variable
X_test = test_df[['Criminal Case', 'Total Assets', 'Liabilities', 'state']]


predictions = best_rf_classifier.predict(X_test)

# Create a DataFrame with predictions
# Assuming 'ID' column contains the IDs of candidates in the test data
# Create a DataFrame with 'ID' and 'Prediction' columns
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Prediction': predictions})

# 4. Save predictions to a CSV file
submission_df.to_csv('predictions.csv', index=False)