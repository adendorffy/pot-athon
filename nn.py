import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Define training and testing data
train_data = pd.read_csv('all_potholes.csv')

test_data = pd.read_csv('test_potholes.csv')

# Features and target for training
X_train = train_data[['pothole_area_mm2', 'aspect_ratio']]
y_train = train_data['Bags used ']

# Features for testing
X_test = test_data[['pothole_area_mm2', 'aspect_ratio']]

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, verbose=1)

# Predict on the test data
predictions = model.predict(X_test_scaled)

# Create output DataFrame
output_df = pd.DataFrame({
    'Pothole Number': test_data['filename'],
    'Bags used': predictions.flatten()
})

# Save the output to a CSV file
output_df.to_csv('pothole_predictions.csv', index=False)

print("Predictions saved to 'pothole_predictions.csv'")
print(output_df)
