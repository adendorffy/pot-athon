import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('all_potholes.csv')

data['pothole_area_cm2'] = data['pothole_area_mm2']/1000

data.to_csv('all_potholes_cm.csv', index=False)

data.describe()

# Distribution of pothole area
plt.figure(figsize=(10, 6))
sns.histplot(data['pothole_area_mm2'], bins=30, kde=True)
plt.title('Distribution of Pothole Area (mm²)')
plt.xlabel('Pothole Area (mm²)')
plt.ylabel('Frequency')
plt.show()

# Distribution of Bags used
plt.figure(figsize=(10, 6))
sns.histplot(data['Bags used '], bins=30, kde=True)
plt.title('Distribution of Bags Used')
plt.xlabel('Bags Used')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pothole_area_mm2', y='Bags used ', data=data)
plt.title('Pothole Area vs. Bags Used')
plt.xlabel('Pothole Area (mm²)')
plt.ylabel('Bags Used')
plt.show()

# Correlation matrix
corr_matrix = data.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Boxplot for Pothole Area
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['pothole_area_mm2'])
plt.title('Boxplot of Pothole Area (mm²)')
plt.show()

# Boxplot for Bags used
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Bags used '])
plt.title('Boxplot of Bags Used')
plt.show()

# Pairplot
sns.pairplot(data)
plt.show()
