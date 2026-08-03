# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    BSD license.
#    This repository is based on Cluster-GCN
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer, seed_everything
from utils import tab_printer, dataset_reader


def main():
    """
    Parsing command line parameters, reading data, graph decomposition,
    fitting a ClusterGCN and scoring the model.
    """
    args = parameter_parser()
    seed_everything(args.seed)
    tab_printer(args)

    graph, features, target = dataset_reader(args)
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()} | "
          f"Features: {features.shape[1]} | Classes: {int(np.max(target) + 1)}")

    clustering_machine = ClusteringMachine(args, graph, features, target)
    clustering_machine.decompose()

    start = time.time()
    gcn_trainer = ClusterGCNTrainer(args, clustering_machine)
    gcn_trainer.train()
    scores = gcn_trainer.test()
    run_time = time.time() - start

    print(f"\nF-1 micro score: {scores['micro']:.4f}")
    print(f"F-1 macro score: {scores['macro']:.4f}")
    print(f"Run time: {run_time:.2f}s")

    # Save a small report alongside the other result CSVs.
    report = {
        "membership_closeness": args.membership_closeness,
        "dataset_name": args.dataset_name,
        "f1_micro": scores["micro"],
        "f1_macro": scores["macro"],
        "overlap_ratio": float(np.sum(clustering_machine.ClusterNodes)) / len(graph.nodes()),
        "run_time": run_time,
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame([report])
    df.to_csv(os.path.join(results_dir, "main_single_run.csv"), index=False)
    print(f"\nReport saved to: {results_dir}/main_single_run.csv")


if __name__ == "__main__":
    main()
