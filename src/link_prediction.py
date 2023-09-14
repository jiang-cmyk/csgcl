
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
 
from torch_geometric.datasets import Planetoid, Amazon
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn



import random
 
class Net(torch.nn.Module):
     def __init__(self, in_channels, hidden_channels, out_channels):
         super().__init__()
         self.conv1 = GCNConv(in_channels, hidden_channels)
         self.conv2 = GCNConv(hidden_channels, out_channels)
 
     def encode(self, x, edge_index):
         x = self.conv1(x, edge_index).relu()
         return self.conv2(x, edge_index)
 
     def decode(self, z, edge_label_index):
         return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
             dim=-1
         )  # product of a pair of nodes on each edge
 
     def decode_all(self, z):
         prob_adj = z @ z.t()
         return (prob_adj > 0).nonzero(as_tuple=False).t()
     
 
 
def train_link_predictor(
     model, train_data, val_data, optimizer, criterion, n_epochs=100
 ):
 
     for epoch in range(1, n_epochs + 1):
 
         model.train()
         optimizer.zero_grad()
         z = model.encode(train_data.x, train_data.edge_index)
 
         # sampling training negatives for every training epoch
         neg_edge_index = negative_sampling(
             edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
             num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
 
         edge_label_index = torch.cat(
             [train_data.edge_label_index, neg_edge_index],
             dim=-1,
         )
         edge_label = torch.cat([
             train_data.edge_label,
             train_data.edge_label.new_zeros(neg_edge_index.size(1))
         ], dim=0)
 
         out = model.decode(z, edge_label_index).view(-1)
         loss = criterion(out, edge_label)
         loss.backward()
         optimizer.step()
 
         with torch.no_grad():
             val_auc = eval_link_predictor(model, val_data)
             # test_val = roc_auc_score(val_data.edge_label.cpu().numpy(), model.decode(z, val_data.edge_label_index).view(-1).sigmoid().cpu().numpy())
             if epoch % 10 == 0:
                 print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
 
     return model
 
 
@torch.no_grad()
def eval_link_predictor(model, data):
 
     model.eval()
     z = model.encode(data.x, data.edge_index)
     out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
 
     return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
 
 
import torch_geometric.transforms as T
 
split = T.RandomLinkSplit(
     num_val=0.0,
     num_test=0.1,
     is_undirected=True,
     add_negative_train_samples=False,
     neg_sampling_ratio=1.0,
)
# Data(x=[13752, 767], edge_index=[2, 393378], y=[13752], edge_label=[196689], edge_label_index=[2, 196689])
# Data(x=[13752, 767], edge_index=[2, 393378], y=[13752], edge_label=[49172], edge_label_index=[2, 49172])
# Data(x=[13752, 767], edge_index=[2, 442550], y=[13752], edge_label=[49172], edge_label_index=[2, 49172])

device = 'cuda' if torch.cuda.is_available() else 'cpu'     

dataset = Amazon(root='../datasets/Amazon-Computers', name='computers', transform=T.NormalizeFeatures())\

graph = dataset[0].to(device)
train_data, val_data, test_data = split(graph)
print(train_data)
print(val_data)
print(test_data)
print(test_data.edge_label[:10])
pass
model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data, val_data, optimizer, criterion)
 
test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}, test_auc AUC: {test_auc:.3f}")