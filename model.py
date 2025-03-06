import torch
import torch.nn as nn
import torch.optim as optim


class HeuristicNet(nn.Module):
    def __init__(self, input_size=28, hidden_size=40):
        super(HeuristicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def train_model(model, dataset, targets, num_epochs=20, batch_size=32, learning_rate=0.001):
    """
    Trains the network on the provided dataset using MSE loss.
    Returns a list of average training losses per epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    dataset = torch.tensor(dataset, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    n_samples = dataset.shape[0]

    for epoch in range(num_epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_states = dataset[indices]
            batch_targets = targets[indices]
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / (n_samples / batch_size)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return losses
