import pandas as pd
import networkx as nx

def build_nodes(df: pd.DataFrame, node_id_col: str, node_feature_cols: list) -> dict:
    """
    Create a dictionary of node features.
    """
    nodes = {}
    for _, row in df.iterrows():
        node_id = row[node_id_col]
        features = {col: row[col] for col in node_feature_cols}
        nodes[node_id] = features
    return nodes

def build_edges(edge_df: pd.DataFrame, src_col: str, dst_col: str, edge_feature_cols: list = None) -> list:
    """
    Create a list of edges (with optional features).
    """
    edges = []
    for _, row in edge_df.iterrows():
        src = row[src_col]
        dst = row[dst_col]
        if edge_feature_cols:
            features = {col: row[col] for col in edge_feature_cols}
            edges.append((src, dst, features))
        else:
            edges.append((src, dst))
    return edges

def build_graph(node_dict: dict, edge_list: list) -> nx.Graph:
    """
    Build a NetworkX graph from nodes and edges.
    """
    G = nx.Graph()
    for node_id, features in node_dict.items():
        G.add_node(node_id, **features)
    for edge in edge_list:
        if len(edge) == 3:
            G.add_edge(edge[0], edge[1], **edge[2])
        else:
            G.add_edge(edge[0], edge[1])
    return G

def build_temporal_graphs(node_dfs: dict, edge_dfs: dict, time_points: list, node_id_col: str, node_feature_cols: list, src_col: str, dst_col: str, edge_feature_cols: list = None) -> dict:
    """
    Build a dictionary of graphs, one per time point.
    """
    graphs = {}
    for t in time_points:
        node_dict = build_nodes(node_dfs[t], node_id_col, node_feature_cols)
        edge_list = build_edges(edge_dfs[t], src_col, dst_col, edge_feature_cols)
        graphs[t] = build_graph(node_dict, edge_list)
    return graphs