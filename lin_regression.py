import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
test_df = pd.read_csv('data-v6/test/test.csv')
train_df = pd.read_csv('data-v6/train/train.csv')
valid_df = pd.read_csv('data-v6/valid/valid.csv')

# Combine train and validation datasets
combined_df = pd.concat([train_df, valid_df])

# Drop rows with NaN values
combined_df = combined_df.dropna()

combined_df = combined_df[combined_df['pothole_area_mm2'] <= 10000000]
print(combined_df)

# Plotting area vs. bags used
plt.figure(figsize=(10, 6))
plt.scatter(combined_df['pothole_area_mm2'], combined_df['Bags used '], color='blue', edgecolor='black', alpha=0.7)
plt.title('Pothole Area vs. Bags Used')
plt.xlabel('Pothole Area (mm²)')
plt.ylabel('Bags Used')
plt.grid(True)
plt.show()

# Features (X) and target (y)
X = combined_df[['pothole_area_mm2']]
y = combined_df['Bags used ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


