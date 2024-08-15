# STACKING REGRESSION

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])
X = df[['Area']]
y = df['Bags']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
base_learners = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42))
]

# Define meta-learner
meta_learner = RidgeCV()

# Initialize and train stacking model
stacking_model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)
stacking_model.fit(X_train, y_train)

# Evaluate the model
y_pred_train = stacking_model.predict(X_train)
y_pred_test = stacking_model.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f'Stacking Train MSE: {train_mse:.5f}')
print(f'Stacking Test MSE: {test_mse:.5f}')

# VOTING REGRESSION

from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])
X = df[['Area']]
y = df['Bags']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR())
]

# Initialize and train voting regressor
voting_regressor = VotingRegressor(estimators=models)
voting_regressor.fit(X_train, y_train)

# Evaluate the model
y_pred_train = voting_regressor.predict(X_train)
y_pred_test = voting_regressor.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f'Voting Regressor Train MSE: {train_mse:.5f}')
print(f'Voting Regressor Test MSE: {test_mse:.5f}')

# BAGGING

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])
X = df[['Area']]
y = df['Bags']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train bagging regressor
bagging_model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# Evaluate the model
y_pred_train = bagging_model.predict(X_train)
y_pred_test = bagging_model.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f'Bagging Regressor Train MSE: {train_mse:.5f}')
print(f'Bagging Regressor Test MSE: {test_mse:.5f}')

# BOOSTING

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])
X = df[['Area']]
y = df['Bags']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train gradient boosting regressor
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train, y_train)

# Evaluate the model
y_pred_train = gbr.predict(X_train)
y_pred_test = gbr.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f'Gradient Boosting Train MSE: {train_mse:.5f}')
print(f'Gradient Boosting Test MSE: {test_mse:.5f}')

# BLENDING

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

# Split the data
X = df[['Area']]
y = df['Bags']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Define base learners
base_learners = {
    'lr': LinearRegression(),
    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
    'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'svr': SVR(),
    'mlp': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
}

# Train base learners
base_predictions_train = pd.DataFrame()
base_predictions_val = pd.DataFrame()

for name, model in base_learners.items():
    model.fit(X_train, y_train)
    base_predictions_train[name] = model.predict(X_train)
    base_predictions_val[name] = model.predict(X_val)

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

# Define and train meta-learner
meta_learner = RidgeCV()

# Train the meta-learner on validation set predictions
meta_learner.fit(base_predictions_val, y_val)

# Predict on the test set using base learners
base_predictions_test = pd.DataFrame()
for name, model in base_learners.items():
    base_predictions_test[name] = model.predict(X_test)

# Predict using the meta-learner
y_pred_test = meta_learner.predict(base_predictions_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, meta_learner.predict(base_predictions_train))
test_mse = mean_squared_error(y_test, y_pred_test)

print(f'Blending Train MSE: {train_mse:.5f}')
print(f'Blending Test MSE: {test_mse:.5f}')
