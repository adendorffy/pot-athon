import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
test_df = pd.read_csv('data-v6/test/test.csv')
train_df = pd.read_csv('data-v6/train/train.csv')
valid_df = pd.read_csv('data-v6/valid/valid.csv')

# Combine train and validation datasets
combined_df = pd.concat([train_df, valid_df])

# Drop rows with NaN values
combined_df = combined_df.dropna()

# Features (X) and target (y)
X = combined_df[['pothole_area_mm2']]
y = combined_df['Bags used ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

final_test = pd.read_csv('their_test.csv')
final_pred = model.predict(final_test[['pothole_area_mm2']])
final_test['Bags used'] = final_pred.flatten()
final_test.to_csv('predictions.csv', index=False)