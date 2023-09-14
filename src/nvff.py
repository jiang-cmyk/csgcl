import torch_geometric.transforms as T
 
from torch_geometric.datasets import Planetoid, Amazon
import torch

import torch.nn as nn



import random
# from torch_geometric.utils import to_networkx
# import networkx as nx
 


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(dataset.num_node_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, dataset.num_classes)
        )

    def forward(self, data):
         x = data.x  # only using node features (x)
         output = self.layers(x)
         return output
     
def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
 
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
 
        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, graph, graph.val_mask)
 
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')
 
    return model
 
 
def eval_node_classifier(model, graph, mask):
 
     model.eval()
     pred = model(graph).argmax(dim=1)
     correct = (pred[mask] == graph.y[mask]).sum()
     acc = int(correct) / int(mask.sum())
     return acc

device = 'cuda' if torch.cuda.is_available() else 'cpu'     

dataset = Amazon(root='../datasets/Amazon-Computers', name='computers', transform=T.NormalizeFeatures())
print(len(dataset))
graph = dataset[0]
split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
graph = split(graph).to(device)
print(graph) 
mlp = MLP().to(device)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
mlp = train_node_classifier(mlp, graph, optimizer_mlp, criterion, n_epochs=150)
 
test_acc = eval_node_classifier(mlp, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}')