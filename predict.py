import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge 
df = pd.read_csv('features_resNet.csv')
train_data = df[df['type'] == 'train'].drop(columns=['type'])
test_data = df[df['type'] == 'test'].drop(columns=['type'])

df=train_data
df = df.dropna(subset=['Bags'])
X=df[['BoundingBoxArea','Width','Height','feature_1039','feature_1409']]
#'Area','feature_1039','feature_1409','feature_1567','feature_126','feature_1484','feature_975','feature_994','feature_42','feature_1864','feature_1118','feature_445','feature_1118', 'feature_409'
y = df['Bags']
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)

model = Ridge(alpha=10.0)
kf = KFold(n_splits=20, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error)
cv_scores = cross_val_score(model, X_poly, y, cv=kf, scoring=mse_scorer)
print(f'Mean Cross-Validation MSE: {cv_scores.mean()}')

# use entire set for train
model.fit(X_poly, y)
test_df = pd.read_csv('test_area_features_resNet.csv')
test_df=test_data
X_test = test_df[['BoundingBoxArea', 'Width', 'Height','feature_1039','feature_1409']]


X_test_poly = poly.transform(X_test)
predictions = model.predict(X_test_poly)
result_df = pd.DataFrame({
    'Pothole number': test_df['ID'],
    'Bags used': predictions
})

result_df.to_csv("results.csv", index=False)