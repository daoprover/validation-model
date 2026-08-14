import os
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx
import numpy as np
import random
import logging

from utils.graph import GraphHelper
from utils.features import model_edge_feature_vector, model_node_feature_vector


class GraphDatasetLoader(Dataset):
    def __init__(self, base_dir, label_encoder, logger: logging.Logger, dataset_size=None):
        self.logger = logger
        self.base_dir = base_dir
        if not os.path.isdir(base_dir):
            raise ValueError(f"Graph dataset directory does not exist: {base_dir}")
        self.all_files = sorted(f for f in os.listdir(base_dir) if f.endswith('.gexf'))
        requested_size = len(self.all_files) if dataset_size is None else dataset_size
        n = min(requested_size, len(self.all_files)) if self.all_files else 0
        self.dataset_size = n
        self.files = (
            list(self.all_files)
            if n == len(self.all_files)
            else random.sample(self.all_files, n)
        )
        self.graph_helper = GraphHelper(self.logger)
        self.label_encoder = label_encoder
        self._metadata_cache = {}

    def __len__(self):
        return len(self.files)

    def metadata(self, idx):
        if idx not in self._metadata_cache:
            filepath = os.path.join(self.base_dir, self.files[idx])
            graph, label = self.graph_helper.load_transaction_graph_from_gexf(filepath)
            transaction_ids = {
                str(node)
                for node, attrs in graph.nodes(data=True)
                if str(node).startswith("transaction:") or str(attrs.get("node_type")) == "1"
            }
            self._metadata_cache[idx] = {
                "label": "anomaly" if label != "white" else "white",
                "transaction_ids": transaction_ids,
                "path": filepath,
            }
        return self._metadata_cache[idx]

    def __getitem__(self, idx):
        if idx >= len(self.files):
            raise IndexError("Out of available files")

        filepath = os.path.join(self.base_dir, self.files[idx])
        try:
            graph, label = self.graph_helper.load_transaction_graph_from_gexf(filepath)
        except Exception as exc:
            raise RuntimeError(f"Failed to load graph {filepath}") from exc

        if graph.number_of_nodes() == 0:
            raise ValueError(f"Graph is empty: {filepath}")

        graph_pyg = from_networkx(graph)
        node_features = [model_node_feature_vector(attrs) for _, attrs in graph.nodes(data=True)]
        graph_pyg.x = torch.tensor(np.array(node_features), dtype=torch.float)

        edge_features = [model_edge_feature_vector(attrs) for _, _, attrs in graph.edges(data=True)]
        if edge_features:
            graph_pyg.edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
        else:
            graph_pyg.edge_attr = torch.zeros((0, 4), dtype=torch.float)

        if not label:
            raise ValueError(f"Label is missing for graph: {filepath}")

        binary_label = "anomaly" if label != "white" else "white"
        try:
            encoded_label = self.label_encoder.transform([binary_label])
        except ValueError as exc:
            raise ValueError(f"Failed to encode label '{binary_label}' for {filepath}") from exc

        graph_pyg.y = torch.tensor([encoded_label[0]], dtype=torch.long)
        return graph_pyg

    def shuffle(self):
        n = min(self.dataset_size, len(self.all_files)) if self.all_files else 0
        self.files = random.sample(self.all_files, n) if n else []
