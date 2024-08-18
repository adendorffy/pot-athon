import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('all_potholes_cm.csv')

# Filter rows where 'Bags used' is NaN
nan_rows = data[data['Bags used '].isna()]

# Remove those rows from the original DataFrame if needed
data_without_nan = data.dropna(subset=['Bags used '])

data_without_nan.to_csv('no_label_potholes.csv', index=False)

# Fill NaN values (if needed)
data['Bags used '].fillna(data['Bags used '].mean(), inplace=True)

data = data.dropna()

# Calculate the Z-scores
data['z_score_area'] = stats.zscore(data['pothole_area_cm2'])
data['z_score_bags'] = stats.zscore(data['Bags used '])
# Identify outliers (Z-score > 3 or < -3)
outliers_area = data[(data['z_score_area'] > 3) | (data['z_score_area'] < -3)]
outliers_bags = data[(data['z_score_bags'] > 3) | (data['z_score_bags'] < -3)]

# Define Z-score threshold (e.g., 3 standard deviations)
z_threshold = 3

# Identify outliers
outliers = data[data['z_score_bags'].abs() > z_threshold]
non_outliers = data[data['z_score_bags'].abs() <= z_threshold]

# Features and target for non-outliers
X_non_outliers = non_outliers[['pothole_area_mm2', 'pothole_area_cm2']]  # Replace with relevant features
y_non_outliers = non_outliers['Bags used ']

# Split into training and test sets
X_train_non, X_test_non, y_train_non, y_test_non = train_test_split(X_non_outliers, y_non_outliers, test_size=0.2, random_state=42)

# Train a model (e.g., RandomForestRegressor)
model_non_outliers = RandomForestRegressor(random_state=42)
model_non_outliers.fit(X_train_non, y_train_non)

# Evaluate the model on the non-outlier test set
non_outlier_predictions = model_non_outliers.predict(X_test_non)

# Features and target for outliers
X_outliers = outliers[['pothole_area_mm2', 'pothole_area_cm2']]  # Replace with relevant features
y_outliers = outliers['Bags used ']

# Split into training and test sets
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_outliers, y_outliers, test_size=0.2, random_state=42)

# Train a model for outliers (e.g., RandomForestRegressor)
model_outliers = RandomForestRegressor(random_state=42)
model_outliers.fit(X_train_out, y_train_out)

# Evaluate the model on the outlier test set
outlier_predictions = model_outliers.predict(X_test_out)

# Evaluate the non-outlier model on the test set
mae_non = mean_absolute_error(y_test_non, non_outlier_predictions)
mse_non = mean_squared_error(y_test_non, non_outlier_predictions)
r2_non = r2_score(y_test_non, non_outlier_predictions)

print("Non-Outlier Model Performance:")
print(f"Mean Absolute Error: {mae_non}")
print(f"Mean Squared Error: {mse_non}")
print(f"R² Score: {r2_non}")

# Evaluate the outlier model on the test set
mae_out = mean_absolute_error(y_test_out, outlier_predictions)
mse_out = mean_squared_error(y_test_out, outlier_predictions)
r2_out = r2_score(y_test_out, outlier_predictions)

print("Outlier Model Performance:")
print(f"Mean Absolute Error: {mae_out}")
print(f"Mean Squared Error: {mse_out}")
print(f"R² Score: {r2_out}")


# Load the test data from their_test.csv
test_data = pd.read_csv('their_test.csv')

# Calculate 'pothole_area_cm2'
test_data['pothole_area_cm2'] = test_data['pothole_area_mm2'] / 1000

# Calculate Z-scores based on the non-outliers' area statistics
mean_area = non_outliers['pothole_area_cm2'].mean()
std_area = non_outliers['pothole_area_cm2'].std()
test_data['z_score_area'] = (test_data['pothole_area_cm2'] - mean_area) / std_area

# Function to predict 'Bags used' for each row in test data
def predict_for_test_data(row):
    if abs(row['z_score_area']) > z_threshold:  # Using area-based Z-score to decide if it's an outlier
        # Use the outlier model
        return model_outliers.predict([[row['pothole_area_mm2'], row['pothole_area_cm2']]])[0]
    else:
        # Use the non-outlier model
        return model_non_outliers.predict([[row['pothole_area_mm2'], row['pothole_area_cm2']]])[0]

# Apply the prediction function to each row in the test data
test_data['Bags used'] = test_data.apply(predict_for_test_data, axis=1)

test_data['Pothole Number'] = test_data['pothole_id'] 
test_data = test_data[['Pothole Number', 'Bags used']]

# Display the final test data with predictions
print(test_data)

# Save the predictions to a new CSV file
test_data.to_csv('their_test_predictions.csv', index=False)

# Display confirmation
print("Predictions saved to 'their_test_predictions.csv'")
