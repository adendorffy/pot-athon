import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

# Split the data
X=df[['BoundingBoxArea','Width','Height','Area']]
y = df['Bags']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (50, 50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant','adaptive']
}

# Initialize MLPRegressor
mlp = MLPRegressor(max_iter=500, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

def safe_fit(model, X_train, y_train):
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error occurred: {e}")

# Fit GridSearchCV
safe_fit(grid_search, X_train, y_train)

# Get the best model and parameters
best_mlp = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions
y_pred_train = best_mlp.predict(X_train)
y_pred_test = best_mlp.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f'Best Parameters: {best_params}')
print(f'MLP Regressor Train MSE: {train_mse:.5f}')
print(f'MLP Regressor Test MSE: {test_mse:.5f}')
