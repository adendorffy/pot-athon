import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load the data
df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

# Split the data
X = df[['BoundingBoxArea','Width','Height','Area']].values
y = df['Bags'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Hyperparameters to test
hidden_dims = [(64, 32), (128, 64), (64, 64)]
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [10, 20, 30]
num_epochs = 100

# Initialize lists for storing results
results = []

# Loop over different hyperparameters
for hidden_dim1, hidden_dim2 in hidden_dims:
    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Create a dataset and data loader with the current batch size
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Define the neural network model
            class SimpleNN(nn.Module):
                def __init__(self, input_dim, hidden_dim1, hidden_dim2):
                    super(SimpleNN, self).__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim1)
                    self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
                    self.fc3 = nn.Linear(hidden_dim2, 1)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x

            # Initialize the model
            input_dim = X_train_scaled.shape[1]
            model = SimpleNN(input_dim, hidden_dim1, hidden_dim2)

            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train the model
            for epoch in range(num_epochs):
                for X_batch, y_batch in train_loader:
                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Evaluate the model
            with torch.no_grad():
                y_pred_train = model(X_train_tensor).squeeze().numpy()
                y_pred_test = model(X_test_tensor).squeeze().numpy()

            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)

            # Store results
            results.append({
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'learning_rate': lr,
                'batch_size': batch_size,
                'train_mse': train_mse,
                'test_mse': test_mse
            })

            print(f'Hidden Layers: {hidden_dim1}, {hidden_dim2} | LR: {lr} | Batch Size: {batch_size}')
            print(f'Train MSE: {train_mse:.5f}, Test MSE: {test_mse:.5f}\n')

# Find the best hyperparameters
best_result = min(results, key=lambda x: x['test_mse'])
print(f'Best Hyperparameters: {best_result}')

import json

model_name = "TorchNeuralNetwork"

optimal_config = {
    'hidden_dim1': hidden_dim1,
    'hidden_dim2': hidden_dim2,
    'learning_rate': lr,
    'batch_size': batch_size
}

output_file = f"{model_name.lower()}_results.json"

# Prepare the results dictionary
results = {
    "Regression Model": model_name,
    "Optimal Configuration": optimal_config,
    "Training MSE": train_mse,
    "Test MSE": test_mse
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")