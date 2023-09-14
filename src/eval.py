from typing import Optional
import torch
from torch.optim import Adam
import torch.nn as nn
from src.model import LogReg
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T

def get_idx_split(dataset, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')
    
    
def log_regx(z,dataset,device: Optional[str] = None, n_epochs=800):

    device = z.device if device is None else device
    z = z.detach().to(device)
    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    # graph = dataset.to(device)
    graph = split(dataset).to(device)
    mlp = MLP(dataset.y.max().item() + 1, z.shape[1]).to(device)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    acc = train_node_classifier(mlp, z, graph, graph.train_mask, optimizer_mlp, criterion, n_epochs)
 
    return acc
    
    
    
def log_regwiki(z,dataset, i,device: Optional[str] = None, n_epochs=800):

    device = z.device if device is None else device
    z = z.detach().to(device)
    # split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    graph = dataset.to(device)
    # print(dataset) 
    # print(graph.train_mask)
    train_mask = dataset.train_mask[:,i]
    mlp = MLP(dataset.y.max().item() + 1, z.shape[1]).to(device)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    acc = train_node_classifier(mlp, z, graph, train_mask, optimizer_mlp, criterion, n_epochs)
 
    return acc

def log_reglp(z, train_data, test_data,device: Optional[str] = None, n_epochs=800):

    device = z.device if device is None else device
    z = z.detach().to(device)
    
    
    model = Net(z.shape[1], 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    auc = train_link_predictor(model, z, train_data, test_data, optimizer, criterion)
    # eval_link_predictor(model, z, test_data)
    return auc
    
def log_regression(z,
                   dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
    
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    y = dataset[0].y.view(-1).to(test_device)
    num_classes = dataset[0].y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    split = get_idx_split(dataset, split, preload_split)
    split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()
    best_test_acc = 0
    best_val_acc = 0
    # print(len(split['train']))
    # print(len(split['test']))
    # print(len(split['val']))
    # print(split['test'])
    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(z[split['train']])
        loss = nll_loss(f(output), y[split['train']])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                
            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                
                if best_test_acc < acc:
                    best_test_acc = acc
               
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}, '
                     )

    return {'acc': best_test_acc}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()
 
    def eval(self, res):
        return {'acc': self._eval(**res)}



class MLP(nn.Module):
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(num_node_features,
                    64),
                    nn.ReLU(),
                    # nn.Linear(64, 32),
                    # nn.ReLU(),
                    nn.Linear(64, 
                    num_classes
                  )
        )

    def forward(self, data):
         x = data#.x  # only using node features (x)
         output = self.layers(x)
         return output


 
class Net(torch.nn.Module):
     def __init__(self, in_channels, hidden_channels, out_channels):
         super().__init__()
         self.conv1 = nn.Linear(in_channels, 
         #                        hidden_channels)
         # self.conv2 = nn.Linear(hidden_channels, 
                                out_channels)
 
     def encode(self, x):
         # x = self.conv1(x).relu()
         return self.conv1(x)
 
     def decode(self, z, edge_label_index):
         return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
             dim=-1
         )  # product of a pair of nodes on each edge
 
     def decode_all(self, z):
         prob_adj = z @ z.t()
         return (prob_adj > 0).nonzero(as_tuple=False).t()
     

def train_link_predictor(
     model, em, train_data, test_data, optimizer, criterion, n_epochs=100
 ):
     for epoch in range(1, n_epochs + 1):
 
         model.train()
         optimizer.zero_grad()
         z = model.encode(em)
 
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

         # best_test_acc = 0
         # if not epoch % 20:
         #    with torch.no_grad():
         #
         #        auc = eval_link_predictor(model, z, test_data)
         #
         #        if best_test_acc < auc:
         #            best_test_acc = auc
     auc = 0
     with torch.no_grad():
        auc = eval_link_predictor(model, em, test_data)
 
     return auc
 
 
def eval_link_predictor(model, em, data):
 
     model.eval()
     z = model.encode(em)
     out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
 
     return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
 
def train_node_classifier(model, z, graph, train_mask, optimizer, criterion, n_epochs=3000):
 
    best_test_acc = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(z)
        loss = criterion(out[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()
        # print(loss.item())
        if not epoch % 20:
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc = eval_node_classifier(pred, graph, graph.test_mask)
            
                if best_test_acc < acc:
                    best_test_acc = acc
    return best_test_acc
 
 
def eval_node_classifier(pred, graph, mask):
 
     correct = (pred[mask] == graph.y[mask]).sum()
     acc = int(correct) / int(mask.sum())
     return acc
