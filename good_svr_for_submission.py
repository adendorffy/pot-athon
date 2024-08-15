import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

# Define the features and target
X = df[['BoundingBoxArea','Width','Height','Area']]
y = df['Bags']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVR(kernel='rbf', C=0.1, epsilon=0.1)
model.fit(X_scaled, y)

test_df = pd.read_csv('test_area_features.csv')
X_test = test_df[['BoundingBoxArea','Width','Height','Area']]

X_test_scaled = scaler.transform(X_test)

y_pred_test = model.predict(X_test_scaled)

# y_pred_test = np.round(y_pred_test, 2)

submission_df = pd.DataFrame({
    'Pothole number': test_df['ID'],
    'Bags used': y_pred_test
})

submission_df.to_csv('svr_submission.csv', index=False)

print("Submission file 'svr_submission.csv' created successfully.")

