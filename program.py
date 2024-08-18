import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('train_features.csv')
df = df.dropna(subset=['Bags'])
df['BB'] = df['Width'] * df['Height']
X = df[['Width', 'Height', 'BB']]
y = df['Bags']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X_scaled)


model = Lasso(alpha=0.1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error)
cv_scores = cross_val_score(model, X_poly, y, cv=kf, scoring=mse_scorer)
print(f'Mean Cross-Validation MSE: {cv_scores.mean()}')


model.fit(X_poly, y)

feature_names = poly.get_feature_names_out(X.columns)
coefficients = model.coef_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
importance_df['Importance'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(feature_names, np.abs(coefficients))
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Feature')
plt.title('Feature Importance (Lasso Regression)')
plt.show()

test_df = pd.read_csv('test_features.csv')


test_df['BB'] = test_df['Width'] * test_df['Height']

X_test = test_df[['Width', 'Height', 'BB']]

X_test_scaled = scaler.transform(X_test)

X_test_poly = poly.transform(X_test_scaled)

predictions = model.predict(X_test_poly)

test_df['Predicted_Bags'] = predictions
test_df[['ID', 'Predicted_Bags']].to_csv('pred.csv', index=False)

print("Predictions have been saved to 'test_predictions.csv'.")
