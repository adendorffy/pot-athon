import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import matplotlib.pyplot as plt

# Load the datasets
test_df = pd.read_csv('data-v6/test/test1.csv')
train_df = pd.read_csv('data-v6/train/train1.csv')
valid_df = pd.read_csv('data-v6/valid/valid1.csv')

# Combine train and validation datasets
train_df = pd.concat([train_df, valid_df])

# Cap the pothole areas at 100,000 mm^2
train_df['pothole_area_mm2'] = np.where(train_df['pothole_area_mm2'] > 1e6, 1e6, train_df['pothole_area_mm2'])
test_df['pothole_area_mm2'] = np.where(test_df['pothole_area_mm2'] > 1e6, 1e6, test_df['pothole_area_mm2'])

train_df = train_df.dropna()  # Remove rows with NaN values in the training set
test_df = test_df.dropna() 

train_df['log_pothole_area'] = np.log1p(train_df['pothole_area_mm2'])
train_df['log_bags_used'] = np.log1p(train_df['Bags used '])

test_df['log_pothole_area'] = np.log1p(test_df['pothole_area_mm2'])
test_df['log_bags_used'] = np.log1p(test_df['Bags used '])

# Define the features and target variable
X = train_df[['log_pothole_area']]
y = train_df['log_bags_used']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Initialize the SVR model
svm = SVR()

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2, 0.5],
    'kernel': ['linear', 'rbf']  # You can add more kernels if needed
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Number of different subsets to evaluate
num_subsets = 5

results = []

for i in range(num_subsets):
    print(f"\nTraining on subset {i+1} of the training data...")
    
    # Randomly split the data into a training subset and the rest
    X_subset, _, y_subset, _ = train_test_split(X_scaled, y, train_size=0.8, random_state=np.random.randint(1000))
    
    # Fit GridSearchCV on the subset
    grid_search.fit(X_subset, y_subset)
    
    # Best parameters and best score for the subset
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Score (Negative MSE): {best_score}")
    
    # Train the model with the best parameters on the subset
    best_svm_model = grid_search.best_estimator_

    # Save the best SVM model for this subset
    joblib.dump(best_svm_model, f'best_svm_model_subset_{i+1}.pkl')

    # Make predictions with the best model on the test set
    X_test_scaled = scaler.transform(test_df[['log_pothole_area']])
    y_test = test_df['Bags used ']
    y_pred = best_svm_model.predict(X_test_scaled)
    new_df = pd.DataFrame({
        'pothole_number': test_df['pothole_id'], 
        'area': test_df['log_pothole_area'], 
        'y_pred': y_pred, 
        'y_true': y_test
    })
    print(new_df)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Optimized SVM Mean Squared Error: {mse}")
    print(f"Optimized SVM R^2 Score: {r2}")
    
    # Store results
    results.append({'subset': i+1, 'mse': mse, 'r2': r2})

    # Visualize predictions vs actual values for this subset
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Bags Used')
    plt.ylabel('Predicted Bags Used')
    plt.title(f'SVM (Subset {i+1}): Actual vs Predicted Bags Used')
    plt.grid(True)
    plt.show()

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('subset_training_results.csv', index=False)
