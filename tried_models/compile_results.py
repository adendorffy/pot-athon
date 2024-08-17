import json
import os
import pandas as pd

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
