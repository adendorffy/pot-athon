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
X = df[['Area']]
y = df['Bags']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the kernels to try
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Define the range of hyperparameters to test
C_values = [0.1, 1, 10, 100]
epsilon_values = [0.01, 0.1, 1]

best_mse = float('inf')
best_kernel = None
best_C = None
best_epsilon = None

# Loop through the combinations of kernels, C, and epsilon
for kernel in kernels:
    for C in C_values:
        for epsilon in epsilon_values:
            model = SVR(kernel=kernel, C=C, epsilon=epsilon)
            
            # Perform K-Fold Cross-Validation
            kf = KFold(n_splits=20, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
            
            # Fit the model on the training data
            model.fit(X_train_scaled, y_train)
            
            # Predict on the test data
            y_pred = model.predict(X_test_scaled)
            
            # Round the predicted values to two decimal places
            y_pred = np.round(y_pred, 2)
            
            # Calculate MSE on the test data
            mse = mean_squared_error(y_test, y_pred)
            
            print(f'Kernel: {kernel}, C: {C}, epsilon: {epsilon}, Cross-Validation MSE: {-cv_scores.mean()}, Test MSE: {mse}')
            
            # Update the best parameters if this model is better
            if mse < best_mse:
                best_mse = mse
                best_kernel = kernel
                best_C = C
                best_epsilon = epsilon

print(f'\nBest Kernel: {best_kernel}, Best C: {best_C}, Best epsilon: {best_epsilon}, Best Test MSE: {best_mse}')
