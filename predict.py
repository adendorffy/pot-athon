import joblib
import pandas as pd

# Load the test data
df = pd.read_csv('their_test.csv')

# Extract the feature column as a DataFrame
X = df[['pothole_area_mm2']]  # Note the double brackets to keep it as a DataFrame

scaler = joblib.load('scaler.pkl')

# Load the saved model
loaded_model = joblib.load('best_svm_model.pkl')

X_new_scaled = scaler.transform(X)  

# Make predictions
predictions = loaded_model.predict(X_new_scaled)

# Print or analyze the predictions
print("Predictions:", predictions)

df['Bags used'] = predictions
print(df)

new_df = pd.DataFrame({
    'Pothole number' : df['pothole_id'], 
    'Bags used': df['Bags used']

})
new_df.to_csv('my_predictions.csv', index=False)