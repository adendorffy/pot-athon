import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
test_df = pd.read_csv('data-v6/test/test.csv')
train_df = pd.read_csv('data-v6/train/train.csv')
valid_df = pd.read_csv('data-v6/valid/valid.csv')

# Combine train, validation, and test datasets
full_df = pd.concat([train_df, valid_df, test_df])

# Calculate Z-scores for the pothole area and bags used
full_df['z_area'] = zscore(full_df['pothole_area_mm2'])
full_df['z_bags'] = zscore(full_df['Bags used '])

# Impute with median for 'pothole_area_mm2'
full_df.loc[full_df['z_area'].abs() > 3, 'pothole_area_mm2'] = full_df['pothole_area_mm2'].median()

# Impute with median for 'Bags used'
full_df.loc[full_df['z_bags'].abs() > 3, 'Bags used '] = full_df['Bags used '].median()

# Drop the z-score columns
full_df = full_df.drop(columns=['z_area', 'z_bags'])

# Drop rows with any NaN values
full_df = full_df.dropna()

# Define features and target variable
X = full_df[['pothole_area_mm2']]
y = full_df['Bags used ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use HuberRegressor, which is robust to outliers
model = HuberRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
