import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# List of result files (update with the actual filenames)
result_files = [
    "bayesregression_results.json",
    "elasticnetregression_results.json",
    "gammaregression_results.json",
    "gradientboostingregression_results.json",
    "knnregression_results.json",
    "quantileregression_results.json",
    "randomforestregression_results.json",
    "supportvectorregression_results.json",
    "torchneuralnetwork_results.json"
    # Add all other model result files here
]

# Initialize a list to store the compiled data
compiled_data = []

# Loop over each result file
for file in result_files:
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
            compiled_data.append(data)
    else:
        print(f"Warning: {file} not found. Skipping.")

# Convert the compiled data into a DataFrame
df = pd.DataFrame(compiled_data)

# Save the DataFrame to a CSV file
df.to_csv("compiled_results.csv", index=False)

print("Compiled results saved to compiled_results.csv")

# Visualize the DataFrame as a table
fig, ax = plt.subplots(figsize=(10, 4)) # Adjust the size as needed
ax.axis('off') # Hide the axes

# Create a table
tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))

# Style the table
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2) # Adjust scale as needed

# Save the table as an image
plt.savefig('results_table.png', bbox_inches='tight')
print("Table saved as results_table.png")