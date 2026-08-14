import os
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx
import numpy as np
import random
import logging

from utils.graph import GraphHelper
from utils.features import model_edge_feature_vector, model_node_feature_vector, timestamp_bucket


class GraphDatasetLoader(Dataset):
    def __init__(self, base_dir, label_encoder, logger: logging.Logger, dataset_size=3200):
        self.logger = logger
        self.base_dir = base_dir
        self.all_files = [f for f in os.listdir(base_dir) if f.endswith('.gexf')]
        n = min(dataset_size, len(self.all_files)) if self.all_files else 0
        self.dataset_size = n
        self.files = random.sample(self.all_files, n) if n else []
        self.label_encoder = label_encoder
        self.graph_helper = GraphHelper(self.logger)

    def __len__(self):
        return len(self.files)

    def timestamp_to_label(self, timestamp):
        return timestamp_bucket(timestamp)

    def _node_feature_vector(self, attrs):
        return np.asarray(model_node_feature_vector(attrs), dtype=np.float32)

    def __getitem__(self, idx):
        while True:
            if idx >= len(self.files):
                raise IndexError("Out of available files")

            filepath = os.path.join(self.base_dir, self.files[idx])

            try:
                graph, label = self.graph_helper.load_transaction_graph_from_gexf(filepath)
            except Exception as e:
                self.logger.warning("Error loading file %s: %s", filepath, e)
                idx += 1
                continue

            if graph.number_of_nodes() == 0:
                self.logger.warning("Graph is empty: %s. Skipping...", filepath)
                idx += 1
                continue

            graph_pyg = from_networkx(graph)

            node_features = []
            node_time_labels = []
            for _nid, attrs in graph.nodes(data=True):
                node_features.append(self._node_feature_vector(attrs))
                ts = attrs.get('last_transaction_time')
                node_time_labels.append(self.timestamp_to_label(ts))

            graph_pyg.x = torch.tensor(np.stack(node_features), dtype=torch.float)
            graph_pyg.node_time_label = torch.tensor(node_time_labels, dtype=torch.long)

            edge_features = []
            edge_time_labels = []
            for _u, _v, edge_data in graph.edges(data=True):
                feature = model_edge_feature_vector(edge_data)
                edge_features.append(feature)
                edge_time_labels.append(self.timestamp_to_label(edge_data.get('timestamp')))

            if edge_features:
                graph_pyg.edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
            else:
                self.logger.debug("Edge attributes are missing for graph %s, initializing zeros.", filepath)
                graph_pyg.edge_attr = torch.zeros((graph.number_of_edges(), 4), dtype=torch.float)

            graph_pyg.edge_time_label = torch.tensor(edge_time_labels, dtype=torch.long)

            if not label:
                self.logger.warning("Label is missing for file: %s. Skipping...", filepath)
                idx += 1
                continue

            label = "anomaly" if label != "white" else "white"
            try:
                encoded_label = self.label_encoder.transform([label])
            except ValueError:
                self.logger.error("Failed to encode label '%s' for file %s. Skipping.", label, filepath)
                idx += 1
                continue

            graph_pyg.y = torch.tensor([encoded_label[0]], dtype=torch.long)

            return graph_pyg

    def shuffle(self):
        n = min(self.dataset_size, len(self.all_files)) if self.all_files else 0
        self.files = random.sample(self.all_files, n) if n else []
        random.shuffle(self.files)
