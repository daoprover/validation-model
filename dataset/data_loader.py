import os
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx
import numpy as np
import random
import logging


from utils.graph import GraphHelper


class GraphDatasetLoader(Dataset):
    def __init__(self, base_dir, label_encoder, logger: logging.Logger, dataset_size=1000):
        self.logger = logger
        self.dataset_size = dataset_size
        self.base_dir = base_dir
        self.all_files = [f for f in os.listdir(base_dir) if f.endswith('.gexf')]
        self.files = random.sample(self.all_files, self.dataset_size)
        self.graph_helper = GraphHelper(self.logger)
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        while True:  # Keep trying until a valid graph is found
            if idx >= len(self.files):
                raise IndexError("Out of available files")

            filepath = os.path.join(self.base_dir, self.files[idx])

            # Load the graph and label from GEXF file
            try:
                graph, label = self.graph_helper.load_transaction_graph_from_gexf(filepath)
            except Exception as e:
                self.logger.error(f"Error loading file {filepath}: {e}")
                idx += 1
                continue

            # Convert the graph to PyG format
            graph_pyg = from_networkx(graph)

            # Extract node features, use default random features if missing
            node_features = []
            for node in graph.nodes(data=True):
                attvalues = node[1].get('attvalues')
                feature = np.array(attvalues) if attvalues is not None else np.random.rand(
                    5)  # Default to random features
                node_features.append(feature)

            graph_pyg.x = torch.tensor(np.array(node_features), dtype=torch.float)

            # Extract edge features, with defaults if missing
            edge_features = []
            for edge in graph.edges(data=True):
                attvalues = edge[2].get('attvalues', {})
                if attvalues:
                    feature = [float(att['value']) for att in attvalues]  # Convert attributes to float
                else:
                    feature = [0.0, 0.0, 0.0, 0.0]  # Default to zero features
                edge_features.append(feature)

            if edge_features:
                graph_pyg.edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
            else:
                self.logger.debug(f"Edge attributes are missing for graph {filepath}, initializing zeros.")
                graph_pyg.edge_attr = torch.zeros((graph.number_of_edges(), 4),
                                                  dtype=torch.float)  # Default shape is [num_edges, 4]

            # Handle missing or incorrect labels
            if not label:
                self.logger.warning(f"Label is missing for file: {filepath}. Skipping...")
                os.remove(filepath)  # Remove problematic files
                idx += 1
                continue

            label = "anomaly" if label != "white" else "white"
            try:
                encoded_label = self.label_encoder.transform([label])
            except ValueError:
                self.logger.error(f"Failed to encode label '{label}' for file {filepath}. Removing file.")
                os.remove(filepath)  # Remove file if label encoding fails
                idx += 1
                continue

            # Assign label as target
            graph_pyg.y = torch.tensor([encoded_label[0]], dtype=torch.long)

            return graph_pyg

    def shuffle(self):
        random.shuffle(self.files)
        self.files =  random.sample(self.all_files, self.dataset_size)
        random.shuffle(self.files)
