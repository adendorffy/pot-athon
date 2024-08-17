import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('data-v6/train/train.csv')
test_df = pd.read_csv('data-v6/test/test.csv')
valid_df = pd.read_csv('data-v6/valid/valid.csv')

# Combine train and valid datasets for training
train_df = pd.concat([train_df, valid_df, test_df])

train_df = train_df.dropna()

plt.figure(figsize=(10, 6))
plt.scatter(train_df['pothole_area_mm2'], train_df['Bags used '], alpha=0.7)
plt.title('Area vs. Bags')
plt.xlabel('Area')
plt.ylabel('Bags')
plt.grid(True)
plt.savefig('area_vs_bags.png')  # Save the plot as a PNG file