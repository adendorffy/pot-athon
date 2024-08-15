import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
import numpy as np

# Load the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

# Split the data
X=df[['BoundingBoxArea','Width','Height','Area']]
y = df['Bags']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists for storing results
best_mse = float('inf')
best_params = None
best_train_mse = None

# Hyperparameter grid
# Define alpha and lambda options for BayesianRidge
alpha_options = [1e-6, 1e-3, 1e-1]
lambda_options = [1e-6, 1e-3, 1e-1]

# Nested for loop to test different hyperparameter combinations
for alpha in alpha_options:
    for lambda_ in lambda_options:
        # Initialize the model
        model = BayesianRidge(alpha_1=alpha, alpha_2=lambda_, lambda_1=alpha, lambda_2=lambda_)

        # Perform cross-validation on the training set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model_cv = BayesianRidge(alpha_1=alpha, alpha_2=lambda_, lambda_1=alpha, lambda_2=lambda_).fit(X_train_cv, y_train_cv)
            y_pred_val_cv = model_cv.predict(X_val_cv)
            cv_scores.append(mean_squared_error(y_val_cv, y_pred_val_cv))
        mean_cv_mse = np.mean(cv_scores)

        # Train the model on the full training set and evaluate on the test set
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)

        y_pred_test = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)

        # Print the MSE for this combination of hyperparameters
        print(f'Params: alpha={alpha}, lambda={lambda_}')
        print(f'Train MSE: {train_mse:.5f}, Test MSE: {test_mse:.5f}\n')

        # Check if this is the best model so far
        if test_mse < best_mse:
            best_mse = test_mse
            best_params = {
                'alpha': alpha,
                'lambda': lambda_
            }
            best_train_mse = train_mse

# Output the best hyperparameters, corresponding test MSE, and the train MSE of the best model
print(f'Best Hyperparameters: {best_params}')
print(f'Best Test MSE: {best_mse}')
print(f'Train MSE of Best Model: {best_train_mse}')
