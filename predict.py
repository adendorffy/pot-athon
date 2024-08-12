import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv('train_features.csv')
df = df.dropna(subset=['Bags'])
X = df[['Width', 'Height']]
y = df['Bags']
model = LinearRegression()
model.fit(X, y)

test_df = pd.read_csv('test_features.csv')
predictions = model.predict(test_df[['Width', 'Height']])

output_df = pd.DataFrame()
output_df['Pothole number']=test_df[['ID']]
output_df['Bags used'] = predictions
output_df.to_csv('test_predictions.csv', index=False)
