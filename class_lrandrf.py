import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

# Define the features and target
X = df[['BoundingBoxArea', 'Width', 'Height', 'Area']]
y = df['Bags']

# Bin the target variable
bins = np.arange(0.05, 6.05, 0.05)
y_binned = np.digitize(y, bins) - 1

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_train_log = log_reg.predict(X_train_scaled)
y_pred_test_log = log_reg.predict(X_test_scaled)

# Evaluate Logistic Regression
train_accuracy_log = accuracy_score(y_train, y_pred_train_log)
test_accuracy_log = accuracy_score(y_test, y_pred_test_log)

print(f'Logistic Regression - Training Accuracy: {train_accuracy_log:.4f}')
print(f'Logistic Regression - Test Accuracy: {test_accuracy_log:.4f}')

# Save model details
logistic_config = {
    "model": "Logistic Regression",
    "accuracy": {
    "train_accuracy": train_accuracy_log,
    "test_accuracy": test_accuracy_log
    }
}

with open('logistic_reg_results.txt', 'w') as f:
    f.write(str(logistic_config))

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_scaled, y_train)
y_pred_train_rf = rf_clf.predict(X_train_scaled)
y_pred_test_rf = rf_clf.predict(X_test_scaled)

# Evaluate Random Forest
train_accuracy_rf = accuracy_score(y_train, y_pred_train_rf)
test_accuracy_rf = accuracy_score(y_test, y_pred_test_rf)

print(f'Random Forest Classifier - Training Accuracy: {train_accuracy_rf:.4f}')
print(f'Random Forest Classifier - Test Accuracy: {test_accuracy_rf:.4f}')

# Save model details
rf_config = {
    "model": "Random Forest Classifier",
    "accuracy": {
        "train_accuracy": train_accuracy_rf,
        "test_accuracy": test_accuracy_rf
    }
}

with open('rf_clf_results.txt', 'w') as f:
    f.write(str(rf_config))
