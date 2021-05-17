import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GINEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, global_sort_pool

from all_2_states import all_goal_states_np, all_nongoal_states_np, train_loader

class FCModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, fc1_dim: int, fc2_dim: int):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim

        self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc_out = nn.Linear(fc2_dim, 1)

    def forward(self, states_nnet, training=False):
        x = states_nnet

        # preprocess input
        x = torch.tensor(x).float()
        x = x.view(-1, self.state_dim * self.one_hot_depth)

        # first two hidden layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # output
        x = self.fc_out(x)
        # x = F.sigmoid(x)
        return x


class GNModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNModel, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GINEConv(nn.Linear(6, hidden_channels))
        self.conv2 = GINEConv(nn.Linear(hidden_channels, hidden_channels))
        self.conv3 = GINEConv(nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


fc_model = FCModel(state_dim=24, one_hot_depth=6, fc1_dim=100, fc2_dim=100)
graph_model = GNModel(hidden_channels=60)



model = graph_model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def run_train(loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    print(loss.item())

def run_test(loader):
    model.eval()
    misclass = []

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        misclass += list(data[pred != data.y])
    return correct / len(loader.dataset), misclass  # Derive ratio of correct predictions.

print(train_loader)

for epoch in range(1, 101):
    run_train(train_loader)
    train_acc, trM = run_test(train_loader)
#     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
    if epoch % 1 == 0:
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
