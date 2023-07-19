import argparse
import os.path as osp
import random
from typing import Dict

import torch
from torch_geometric.utils import to_networkx

import os
from src import *



def train(epoch: int) -> int:
    model.train()
    optimizer.zero_grad()

    edge_index_1 = ced(data.edge_index, edge_weight, p=param['ced_drop_rate_1'], threshold=args.ced_thr)
    edge_index_2 = ced(data.edge_index, edge_weight, p=param['ced_drop_rate_2'], threshold=args.ced_thr)
    if args.dataset == 'WikiCS':
        x1 = cav_dense(data.x, node_cs, param["cav_drop_rate_1"], max_threshold=args.cav_thr)
        x2 = cav_dense(data.x, node_cs, param["cav_drop_rate_2"], max_threshold=args.cav_thr)
    else:
        x1 = cav(data.x, node_cs, param["cav_drop_rate_1"], max_threshold=args.cav_thr)
        x2 = cav(data.x, node_cs, param['cav_drop_rate_2'], max_threshold=args.cav_thr)
    z1 = model(x1, edge_index_1)
    z2 = model(x2, edge_index_2)
    loss = model.team_up_loss(z1, z2,
                              cs=node_cs,
                              current_ep=epoch,
                              t0=param['t0'],
                              gamma_max=param['gamma'],
                              batch_size=args.batch_size if args.dataset in ['Coauthor-CS'] else None)
    loss.backward()
    optimizer.step()
    return loss.item()


def test() -> Dict:
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    res = {}
    seed = np.random.randint(0, 32767)
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1,
                           generator=torch.Generator().manual_seed(seed))
    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            cls_acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}',
                                     num_epochs=800)
            accs.append(cls_acc['acc'])
        acc = sum(accs) / len(accs)
    else:
        cls_acc = log_regression(z, dataset, evaluator, split='rand:0.1',
                                 num_epochs=3000, preload_split=split)
        acc = cls_acc['acc']
    res["acc"] = acc
    return res


@torch.no_grad()
def testn() ->Dict:
    
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
    return acc


@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.axis('off')
    plt.show()

    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
    plot_points(colors)
    # test()
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--dataset_path', type=str, default="./datasets")
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--verbose', type=str, default='train,eval')
    parser.add_argument('--cls_seed', type=int, default=12345)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--cd', type=str, default='leiden')
    parser.add_argument('--ced_thr', type=float, default=1.)
    parser.add_argument('--cav_thr', type=float, default=1.)

    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'ced_drop_rate_1': 0.3,
        'ced_drop_rate_2': 0.4,
        'cav_drop_rate_1': 0.1,
        'cav_drop_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        't0': 500,
        'gamma': 1.,
    }
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)
    comment = f'{args.dataset}_node_{param["cav_drop_rate_1"]}_{param["cav_drop_rate_2"]}'\
              f'_edge_{param["ced_drop_rate_1"]}_{param["ced_drop_rate_2"]}'\
              f'_t0_{param["t0"]}_gamma_{param["gamma"]}'
    if not args.device == 'cpu':
        args.device = 'cuda'

    print(f"training settings: \n"
          f"data: {args.dataset}\n"
          f"community detection method: {args.cd}\n"
          f"device: {args.device}\n"
          f"batch size if used: {args.batch_size}\n"
          f"communal edge dropping (ced) rate: {param['ced_drop_rate_1']}/{param['ced_drop_rate_2']}\n"
          f"communal attr voting (cav) rate: {param['cav_drop_rate_1']}/{param['cav_drop_rate_2']}\n"
          f"gamma: {param['gamma']}\n"
          f"t0: {param['t0']}\n"
          f"epochs: {param['num_epochs']}\n"
          )

    random.seed(12345)
    torch.manual_seed(args.seed)
    # for node classification branch
    if args.cls_seed is not None:
        np.random.seed(args.cls_seed)

    device = torch.device(args.device)
    path = osp.join(args.dataset_path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = data.to(device)
    p1 = './log/'+args.dataset+'_edge_weight.pt'
    p2 = "./log/"+args.dataset+'_node_cs'
    if os.path.isfile(p1) and os.path.isfile(p2+'.npy'):
        edge_weight = torch.load(p1)
        node_cs = np.load(p2+'.npy')
    else:
        print('Detecting communities...')
        g = to_networkx(data, to_undirected=True)
        communities = community_detection(args.cd)(g).communities
        com = transition(communities, g.number_of_nodes())
        com_cs, node_cs = community_strength(g, communities)
        edge_weight = get_edge_weight(data.edge_index, com, com_cs)
        com_size = [len(c) for c in communities]
        print(f'Done! {len(com_size)} communities detected. \n')
        torch.save(edge_weight, p1)
        np.save(p2, node_cs)

    encoder = Encoder(dataset.num_features,
                      param['num_hidden'],
                      get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']),
                      k=param['num_layers']).to(device)
    model = CSGCL(encoder,
                  param['num_hidden'],
                  param['num_proj_hidden'],
                  param['tau']).to(device)
    optimizer = torch.optim.Adam(model.parameters(),  
                                 lr=param['learning_rate'],
                                 weight_decay=param['weight_decay'])
    last_epoch = 0
    log = args.verbose.split(',')

    for epoch in range(1 + last_epoch, param['num_epochs'] + 1):
        loss = train(epoch)
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        if epoch % args.val_interval == 0:
            res = test()
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {res["acc"]}')
