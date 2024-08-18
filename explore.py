import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_area_features.csv')

df = df.dropna(subset=['Bags'])
df = df['Bags'].value_counts()

plt.figure(figsize=(10, 6))
plt.scatter(df['BoundingBoxArea'],df['Area'], alpha=0.7)
plt.title('Area vs. BoxArea')
plt.xlabel('Area')
plt.ylabel('BoxArea')
plt.grid(True)
plt.savefig('area_vs_bags.png')  # Save the plot as a PNG file






