import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the datasets
test_df = pd.read_csv('data-v6/test/test1.csv')
train_df = pd.read_csv('data-v6/train/train1.csv')
valid_df = pd.read_csv('data-v6/valid/valid1.csv')

# Combine train and validation datasets
train_df = pd.concat([train_df, valid_df, test_df])

train_df = train_df[train_df['pothole_area_mm2'] <= 100000000]

train_df = train_df.dropna()  # Remove rows with NaN values in the training set


# Define the features and target variable
X = train_df[['pothole_area_mm2', 'aspect_ratio']]
y = train_df['Bags used ']

# Perform the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

def neural_net():
    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

    # Assuming 'model' is your trained model
    joblib.dump(model, 'neural_net.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error Neural Net: {mse}")

    # Calculate R^2 Score
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score Neural Net: {r2}")


    # Visualize predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Bags Used')
    plt.ylabel('Predicted Bags Used')
    plt.title('Actual vs Predicted Bags Used')
    plt.grid(True)
    plt.show()

def random_forest():
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=20)
    model.fit(X_train, y_train)\

    # Predict on the test set
    y_pred = model.predict(X_test)
    # Assuming 'model' is your trained model
    joblib.dump(model, 'random_forest.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest MSE: {mse}")
    # Calculate R^2 Score
    r2 = r2_score(y_test, y_pred)

    print(f"R^2 Score Random Forest: {r2}")
    # Visualize predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Bags Used')
    plt.ylabel('Predicted Bags Used')
    plt.title('Actual vs Predicted Bags Used')
    plt.grid(True)
    plt.show()

def gradient_boost():
    # Initialize the GradientBoostingRegressor
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train the model
    gb_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = gb_model.predict(X_test)

    # Assuming 'model' is your trained model
    joblib.dump(gb_model, 'neural_net.pkl')

    # Evaluate the model
    mse_gb = mean_squared_error(y_test, y_pred)
    r2_gb = r2_score(y_test, y_pred)

    print(f"Gradient Boosting MSE: {mse_gb}")
    print(f"Gradient Boosting R^2: {r2_gb}")

    # Visualize predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Bags Used')
    plt.ylabel('Predicted Bags Used')
    plt.title('Actual vs Predicted Bags Used')
    plt.grid(True)
    plt.show()

def stacking_regressor():
    # Define base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]

    # Define meta-model
    meta_model = LinearRegression()

    # Initialize the StackingRegressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    # Train the model
    stacking_model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred_stack = stacking_model.predict(X_test)

    # Assuming 'model' is your trained model
    joblib.dump(stacking_model, 'stacking_model.pkl')

    # Evaluate the model
    mse_stack = mean_squared_error(y_test, y_pred_stack)
    r2_stack = r2_score(y_test, y_pred_stack)

    print(f"Stacking MSE: {mse_stack}")
    print(f"Stacking R^2: {r2_stack}")

    # Visualize predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_stack, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Bags Used')
    plt.ylabel('Predicted Bags Used')
    plt.title('Actual vs Predicted Bags Used')
    plt.grid(True)
    plt.show()

def optimize_random_forest(X_train, X_test, y_train, y_test):
    best_mse = float('inf')
    best_r2 = float('-inf')
    best_params = None

    for n_estimators in [50, 100, 200]:  # Adjust as needed
        for max_depth in [10, 20, 30]:  # Adjust as needed
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"n_estimators: {n_estimators}, max_depth: {max_depth} -> MSE: {mse}, R²: {r2}")

            if mse < best_mse:
                best_mse = mse
                best_r2 = r2
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}

    print(f"Best params: {best_params}, Best MSE: {best_mse}, Best R²: {best_r2}")

    # Train the model with the best parameters
    final_model = RandomForestRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, 'optimized_random_forest.pkl')

    # Evaluate the final model
    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Final Random Forest MSE: {mse}")
    print(f"Final Random Forest R²: {r2}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Bags Used')
    plt.ylabel('Predicted Bags Used')
    plt.title('Actual vs Predicted Bags Used (Optimized Random Forest)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    optimize_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
    neural_net()
    gradient_boost()
    stacking_regressor()