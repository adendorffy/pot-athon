import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

# Split the data
X = df[['Area']]
y = df['Bags']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists for storing results
best_mse = float('inf')
best_params = None
best_train_mse = None

# Hyperparameter grid
n_estimators_options = [50, 100, 200]
learning_rate_options = [0.01, 0.1, 0.2]
max_depth_options = [3, 5, 7]

# Nested for loop to test different hyperparameter combinations
for n_estimators in n_estimators_options:
    for learning_rate in learning_rate_options:
        for max_depth in max_depth_options:
            # Initialize the model
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )

            # Perform cross-validation on the training set
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
            mean_cv_mse = -cv_scores.mean()

            # Train the model on the full training set and evaluate on the test set
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            train_mse = mean_squared_error(y_train, y_pred_train)

            y_pred_test = model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred_test)

            # Print the MSE for this combination of hyperparameters
            print(f'Params: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}')
            print(f'Train MSE: {train_mse:.5f}, Test MSE: {test_mse:.5f}\n')

            # Check if this is the best model so far
            if test_mse < best_mse:
                best_mse = test_mse
                best_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth
                }
                best_train_mse = train_mse

# Output the best hyperparameters, corresponding test MSE, and the train MSE of the best model
print(f'Best Hyperparameters: {best_params}')
print(f'Best Test MSE: {best_mse}')
print(f'Train MSE of Best Model: {best_train_mse}')
