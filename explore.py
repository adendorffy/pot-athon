import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_area_features.csv')

df = df.dropna(subset=['Bags'])

plt.figure(figsize=(10, 6))
plt.scatter(df['Area'], df['Bags'], alpha=0.7)
plt.title('Area vs. Bags')
plt.xlabel('Area')
plt.ylabel('Bags')
plt.grid(True)
plt.savefig('area_vs_bags.png')  # Save the plot as a PNG file


