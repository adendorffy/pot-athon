import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('train_area_features.csv')
df = df.dropna(subset=['Bags'])

X = df[['BoundingBoxArea','Width','Height','Area']].values
y = df['Bags'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

batch_size = 20
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
    
input_dim = X_scaled.shape[1]
model = SimpleNN(input_dim, hidden_dim1=64, hidden_dim2=64)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_df = pd.read_csv('test_area_features.csv')
X_test = test_df[['BoundingBoxArea', 'Width', 'Height', 'Area']].values

X_test_scaled = scaler.transform(X_test)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

with torch.no_grad():
    y_pred_test = model(X_test_tensor).squeeze().numpy()

#y_pred_test = y_pred_test.round(2)

submission_df = pd.DataFrame({
    'Pothole number': test_df['ID'],
    'Bags used': y_pred_test
})

submission_df.to_csv('torchnn_submission.csv', index=False)

print("Submission file 'torchnn_submission.csv' created successfully.")
