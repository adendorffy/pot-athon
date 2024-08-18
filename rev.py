import pandas as pd

# Load R² scores
r2_scores_df = pd.read_csv('r2_scores.csv')

# Initialize a DataFrame to store combined predictions
combined_df = pd.DataFrame()

# Normalize the R² scores to sum to 1 (for weighted average)
min_score = r2_scores_df['score'].min()
r2_scores_df['normalized_score'] = (r2_scores_df['score'] - min_score) / (r2_scores_df['score'] - min_score).sum()

# Iterate through each file based on filename in r2_scores_df
for index, row in r2_scores_df.iterrows():
    filename = row['filename']
    print(filename)
    normalized_r2_score = row['normalized_score']
    
    # Load the corresponding prediction file
    pred_df = pd.read_csv(f"submission_data/{filename}")
    
    # Strip any leading or trailing whitespace from column names
    pred_df.columns = pred_df.columns.str.strip()
    print(pred_df.columns)
    if index < 33:
        # Multiply the "Bags used" by the normalized R² score for weighted average
        pred_df['Weighted Bags used'] = pred_df['Bags used'] * normalized_r2_score
    else:
        pred_df['Weighted Bags used'] = 0
    
    # Store the weighted predictions in the combined DataFrame
    if combined_df.empty:
        combined_df = pred_df[['Pothole number', 'Weighted Bags used']]
    else:
        combined_df = pd.merge(combined_df, pred_df[['Pothole number', 'Weighted Bags used']],
                               on='Pothole number', how='outer', suffixes=('', f'_{index}'))

# Fill NaN values with 0 (in case some pothole numbers are missing from certain files)
combined_df = combined_df.fillna(0)

# Sum the weighted predictions across all files to get the final prediction
combined_df['Bags used'] = combined_df.filter(like='Weighted Bags used').sum(axis=1)

# Drop intermediate weighted columns
combined_df = combined_df[['Pothole number', 'Bags used']]

# Save the final predictions to a CSV file
combined_df.to_csv('final_predicted_bags_used.csv', index=False)

print(combined_df)
