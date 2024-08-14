import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer, mean_squared_error

df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])
X = df[['Area', 'Width', 'Height']]
y = df['Bags']
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error)
cv_scores = cross_val_score(model, X_poly, y, cv=kf, scoring=mse_scorer)
print(f'Mean Cross-Validation MSE: {cv_scores.mean()}')