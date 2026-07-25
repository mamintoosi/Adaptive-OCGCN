"""
Data loaders for heterogeneous datasets (ACM, DBLP, IMDB).
Converts them to homogeneous graphs for use with Overlapping Cluster-GCN.
"""
import torch
import numpy as np
import networkx as nx
from collections import defaultdict


def load_acm(ds_root):
    """Load ACM dataset: paper-paper citation graph."""
    path = f'{ds_root}/ACM/acm/processed/data.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    store = data[0]

    features = store['paper']['x'].numpy()
    target = store['paper']['y'].numpy()
    edges = store[('paper', 'cite', 'paper')]['edge_index']

    G = nx.Graph()
    G.add_nodes_from(range(features.shape[0]))
    for i in range(edges.shape[1]):
        src, dst = edges[0, i].item(), edges[1, i].item()
        G.add_edge(src, dst)

    print(f"ACM: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{np.max(target)+1} classes")
    return G, features, target.reshape(-1, 1)


def load_dblp(ds_root):
    """Load DBLP dataset: author-author graph via shared papers."""
    path = f'{ds_root}/DBLP/dblp/processed/data.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    store = data[0]

    features = store['author']['x'].numpy()
    target = store['author']['y'].numpy()
    ap_edges = store[('author', 'to', 'paper')]['edge_index']

    # Build author-author graph: two authors are connected if they share a paper
    paper_to_authors = defaultdict(set)
    for i in range(ap_edges.shape[1]):
        author, paper = ap_edges[0, i].item(), ap_edges[1, i].item()
        paper_to_authors[paper].add(author)

    G = nx.Graph()
    G.add_nodes_from(range(features.shape[0]))
    for paper, authors in paper_to_authors.items():
        authors = list(authors)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                if G.has_edge(authors[i], authors[j]):
                    G[authors[i]][authors[j]]['weight'] += 1
                else:
                    G.add_edge(authors[i], authors[j], weight=1)

    print(f"DBLP: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{np.max(target)+1} classes")
    return G, features, target.reshape(-1, 1)


def load_imdb(ds_root):
    """Load IMDB dataset: movie-movie graph via shared actors/directors."""
    path = f'{ds_root}/IMDB/imdb/processed/data.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    store = data[0]

    features = store['movie']['x'].numpy()
    y = store['movie']['y']
    # IMDB has multi-label (5 classes), convert to single label
    if y.dim() == 2:
        target = y.argmax(dim=1).numpy()
    else:
        target = y.numpy()

    md_edges = store[('movie', 'to', 'director')]['edge_index']
    ma_edges = store[('movie', '>actorh', 'actor')]['edge_index']

    # Build movie-movie graph: two movies share a director or actor
    G = nx.Graph()
    G.add_nodes_from(range(features.shape[0]))

    # Via shared directors
    director_to_movies = defaultdict(set)
    for i in range(md_edges.shape[1]):
        movie, director = md_edges[0, i].item(), md_edges[1, i].item()
        director_to_movies[director].add(movie)

    for director, movies in director_to_movies.items():
        movies = list(movies)
        for i in range(len(movies)):
            for j in range(i + 1, len(movies)):
                G.add_edge(movies[i], movies[j])

    # Via shared actors
    actor_to_movies = defaultdict(set)
    for i in range(ma_edges.shape[1]):
        movie, actor = ma_edges[0, i].item(), ma_edges[1, i].item()
        actor_to_movies[actor].add(movie)

    for actor, movies in actor_to_movies.items():
        movies = list(movies)
        for i in range(len(movies)):
            for j in range(i + 1, len(movies)):
                G.add_edge(movies[i], movies[j])

    print(f"IMDB: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{np.max(target)+1} classes")
    return G, features, target.reshape(-1, 1)


HETERO_LOADERS = {
    'ACM': load_acm,
    'DBLP': load_dblp,
    'IMDB': load_imdb,
}
