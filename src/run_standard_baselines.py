"""
Standard GNN baselines + random-partition control for Adaptive-OCGCN.

Addresses the reviewer gap: the paper currently compares only Cluster-GCN
variants. This script adds:

  1. Full-batch GCN            (torch_geometric GCNConv, 3 hidden x 16 units)
  2. Full-batch GraphSAGE      (SAGEConv, same capacity)
  3. Full-batch GAT            (GATConv, 8 heads)
  4. Random-partition Cluster-GCN (no overlap, same cluster count as DANMF)

Protocol (mirrors the paper): 4-layer GCN, 16 hidden units, ReLU, dropout 0.5,
Adam lr 0.01; 200 epochs for full-batch models; 10 epochs per subgraph for the
random-partition Cluster-GCN (same as the paper); 70/30 train/test split per
seed; micro + macro F1 on the test set; 20 seeds per dataset.

The full-batch models use a single global 70/30 split per seed; the
Cluster-GCN control splits within each cluster (identical to the paper's
protocol for the DANMF no-overlap baseline).

Run: python run_standard_baselines.py [--seeds 20] [--epochs 200]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import (
    NUM_LABELS,
    SimpleArgs,
    load_dataset,
    fit_danmf_cached,
    run_single,
)
from clustergcn import seed_everything

DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']


# ---------------------------------------------------------------------------
# Full-batch models
# ---------------------------------------------------------------------------
class FullBatchGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.conv4(x, edge_index), dim=1)


class FullBatchSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.conv4 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.conv4(x, edge_index), dim=1)


class FullBatchGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(in_dim, hidden, heads=8, dropout=dropout)
        self.conv2 = GATConv(hidden * 8, hidden, heads=8, dropout=dropout)
        self.conv3 = GATConv(hidden * 8, hidden, heads=8, dropout=dropout)
        self.conv4 = GATConv(hidden * 8, out_dim, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.conv4(x, edge_index), dim=1)


def run_full_batch(graph, features, target, seed, model_name, epochs):
    """Run a full-batch model with a global 70/30 split."""
    seed_everything(seed)

    x = torch.FloatTensor(features)
    y = torch.LongTensor(target.squeeze())
    n = x.shape[0]

    edges = set()
    for u, v in graph.edges():
        edges.add((u, v))
        edges.add((v, u))
    edge_index = torch.LongTensor(list(edges)).t().contiguous()

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=seed)
    train_idx = torch.LongTensor(train_idx)
    test_idx = torch.LongTensor(test_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'GCN':
        model = FullBatchGCN(x.shape[1], 16, int(y.max()) + 1, 0.5)
    elif model_name == 'GraphSAGE':
        model = FullBatchSAGE(x.shape[1], 16, int(y.max()) + 1, 0.5)
    else:
        model = FullBatchGAT(x.shape[1], 8, int(y.max()) + 1, 0.5)
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    start = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index)[test_idx].argmax(dim=1).cpu().numpy()
    tgt = y[test_idx].cpu().numpy()
    micro = f1_score(tgt, pred, average='micro')
    macro = f1_score(tgt, pred, average='macro')
    return {'f1_micro': micro, 'f1_macro': macro, 'runtime': time.time() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--datasets', nargs='*', default=DATASETS)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')

    all_rows = []
    total = len(args.datasets) * len(seeds) * 4
    done = 0

    for ds in args.datasets:
        print(f'\n{"="*60}\n  Dataset: {ds}\n{"="*60}', flush=True)
        graph, features, target = load_dataset(ds, ds_root)

        for seed in seeds:
            for model_name in ['GCN', 'GraphSAGE', 'GAT']:
                res = run_full_batch(graph, features, target, seed, model_name, args.epochs)
                res.update({'dataset': ds, 'method': model_name, 'seed': seed})
                all_rows.append(res)
                done += 1
                print(f'  [{done}/{total}] {ds} {model_name} seed={seed}: micro={res["f1_micro"]:.4f}', flush=True)

            # Random-partition Cluster-GCN (no overlap), same cluster count as DANMF = 2x classes
            k = NUM_LABELS[ds] * 2
            args0 = SimpleArgs(
                dataset_name=ds, ds_root=ds_root, clustering_method='random',
                epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
                learning_rate=0.01, cluster_number=k, num_trial=1,
                layers=[16, 16, 16], overlap_strategy='no_overlap',
                clustering_overlap=False,
            )
            res = run_single(graph, features, target, args0, 'no_overlap')
            res.update({'dataset': ds, 'method': 'RandomPartition', 'seed': seed})
            all_rows.append(res)
            done += 1
            print(f'  [{done}/{total}] {ds} RandomPartition seed={seed}: micro={res["f1_micro"]:.4f}', flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(results_dir, 'standard_baselines_raw.csv'), index=False)

    summary = df.groupby(['dataset', 'method']).agg(
        f1_micro_mean=('f1_micro', 'mean'), f1_micro_std=('f1_micro', 'std'),
        f1_macro_mean=('f1_macro', 'mean'), f1_macro_std=('f1_macro', 'std'),
        runtime_mean=('runtime', 'mean'),
    ).round(4)
    summary.to_csv(os.path.join(results_dir, 'standard_baselines_summary.csv'))
    print('\n===== SUMMARY =====')
    print(summary.to_string())
    print(f'\nSaved: {results_dir}/standard_baselines_raw.csv and _summary.csv')


if __name__ == '__main__':
    main()
