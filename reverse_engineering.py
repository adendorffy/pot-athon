import pandas as pd
import numpy as np
import os

# Load R² scores
r2_scores = pd.read_csv('kaggle_r2scores.csv', index_col='Submission')

# Initialize a dictionary to store the combined predictions
combined_predictions = {}

# Loop over each submission file (1.csv to 32.csv)
for i in range(1, 33):
    # Load the current submission file
    file_name = f'submission_data/{i}.csv'
    df = pd.read_csv(file_name)
    
    # Get the R² score for this submission
    r2_score = r2_scores.loc[i, ' R2Score']
    
    # Process each row in the current submission
    for _, row in df.iterrows():
        print(row)
        identifier = row['Pothole number']
        prediction = row['Bags used']
        
        # If this identifier is not yet in the combined dictionary, initialize it
        if identifier not in combined_predictions:
            combined_predictions[identifier] = {'weighted_sum': 0, 'total_r2': 0}
        
        # Update the weighted sum and total R² score for this identifier
        combined_predictions[identifier]['weighted_sum'] += prediction * r2_score
        combined_predictions[identifier]['total_r2'] += r2_score

# Calculate the actual values (weighted average)
actual_values = {identifier: data['weighted_sum'] / data['total_r2']
                 for identifier, data in combined_predictions.items()}

# Convert the results to a DataFrame
actual_values_df = pd.DataFrame(list(actual_values.items()), columns=['Pothole number', 'Estimated Bags used'])

# Save the results to a new CSV file
actual_values_df.to_csv('estimated_actual_values.csv', index=False)

print("Estimated actual values saved to 'estimated_actual_values.csv'")
