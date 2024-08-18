import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

# Load the datasets
train_df = pd.read_csv('data-v6/train/train1.csv')
valid_df = pd.read_csv('data-v6/valid/valid1.csv')
test_df = pd.read_csv('data-v6/test/test1.csv')

# Combine train, validation, and test datasets
full_df = pd.concat([train_df, valid_df, test_df])

# Cap the pothole areas at 100,000 mm^2
full_df['pothole_area_mm2'] = np.where(full_df['pothole_area_mm2'] > 1e6, 1e6, full_df['pothole_area_mm2'])

# Drop NaN values
full_df = full_df.dropna()

full_df.to_csv('all_potholes.csv', index=False)

correlation = full_df['pothole_area_mm2'].corr(full_df['Bags used '])
print(f"Correlation between pothole area and bags used: {correlation}")

correlation = full_df['aspect_ratio'].corr(full_df['Bags used '])
print(f"Correlation between aspect ratio and bags used: {correlation}")

# Calculate Z-scores for the pothole area and bags used
full_df['z_area'] = zscore(full_df['pothole_area_mm2'])
full_df['z_bags'] = zscore(full_df['Bags used '])
full_df['z_aspect_ratio'] = zscore(full_df['aspect_ratio'])

# Identify outliers
outliers_area = full_df[np.abs(full_df['z_area']) > 5]
outliers_bags = full_df[np.abs(full_df['z_bags']) > 5]
outliers_ar = full_df[np.abs(full_df['z_aspect_ratio']) > 5]

print("Outliers based on pothole area:")
print(outliers_area)

print("\nOutliers based on bags used:")
print(outliers_bags)

print("\nOutliers based on aspect ratio used:")
print(outliers_ar)


# Box plot for pothole area
plt.figure(figsize=(10, 6))
plt.boxplot(full_df['pothole_area_mm2'], vert=False)
plt.title('Box Plot for Pothole Area')
plt.xlabel('Pothole Area (mm²)')
plt.show()

# Box plot for bags used
plt.figure(figsize=(10, 6))
plt.boxplot(full_df['Bags used '], vert=False)
plt.title('Box Plot for Bags Used')
plt.xlabel('Bags Used')
plt.show()

# Plotting the relationship between pothole area and bags used
plt.figure(figsize=(10, 6))
plt.scatter(full_df['pothole_area_mm2'], full_df['Bags used '], alpha=0.5, edgecolors='w', linewidth=0.5)
plt.xlabel('Pothole Area (mm²)')
plt.ylabel('Bags Used')
plt.title('Pothole Area vs. Bags Used')
plt.grid(True)
plt.show()