import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

# Split the data
X=df[['BoundingBoxArea','Width','Height','Area']]
y = df['Bags']
df[['BoundingBoxArea','Width','Height','Area']] = X
df['Bags'] = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists for storing results
best_mse = float('inf')
best_params = None
best_train_mse = None

# Hyperparameter grid
alpha_options = [0.1, 1.0, 10.0]  # Different alpha values to test
link_functions = [sm.families.links.Log(), sm.families.links.Identity()]  # Link functions to test

# Nested for loop to test different alpha values and link functions
for alpha in alpha_options:
    for link_function in link_functions:
        # Define the formula
        formula = 'Bags ~ Area'

        # Fit the Gamma regression model using statsmodels
        model = smf.glm(formula=formula, data=df, family=sm.families.Gamma(link=link_function))
        result = model.fit()

        # Make predictions and evaluate
        y_pred_train = result.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)

        y_pred_test = result.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)

        # Print the MSE for this combination of hyperparameters
        print(f'Alpha: {alpha}, Link Function: {link_function.__class__.__name__}')
        print(f'Train MSE: {train_mse:.5f}, Test MSE: {test_mse:.5f}\n')

        # Check if this is the best model so far
        if test_mse < best_mse:
            best_mse = test_mse
            best_params = {
                'alpha': alpha,
                'link_function': link_function.__class__.__name__
            }
            best_train_mse = train_mse

# Output the best alpha, link function, corresponding test MSE, and the train MSE of the best model
print(f'Best Alpha: {best_params["alpha"]}')
print(f'Best Link Function: {best_params["link_function"]}')
print(f'Best Test MSE: {best_mse}')
print(f'Train MSE of Best Model: {best_train_mse}')
