import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge 
df = pd.read_csv('../train_area_features.csv')
df = df.dropna(subset=['Bags'])
X=df[['BoundingBoxArea','Width','Height','Area']]
y = df['Bags']
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)

model = Ridge(alpha=50.0)
kf = KFold(n_splits=20, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error)
cv_scores = cross_val_score(model, X_poly, y, cv=kf, scoring=mse_scorer)
print(f'Mean Cross-Validation MSE: {cv_scores.mean()}')
