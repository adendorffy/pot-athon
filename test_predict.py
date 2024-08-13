import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('train_features.csv')
df = df.dropna(subset=['Bags'])
X = df[['Width', 'Height']]
y = df['Bags']
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

test_df = pd.read_csv('test_features.csv')
X_test = test_df[['Width', 'Height']]
X_test_poly = poly.transform(X_test)
predictions = model.predict(X_test_poly)

output_df = pd.DataFrame()
output_df['Pothole number']=test_df[['ID']]
output_df['Bags used'] = predictions
output_df.to_csv('test_predictions.csv', index=False)
