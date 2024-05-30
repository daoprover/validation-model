import os
import numpy as np
import pandas as pd
import requests
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import torch

class GraphHelper:
    def __init__(self):
        pass

    def build_transaction_graph(self, transactions):
        G = nx.DiGraph()
        for tx in transactions:
            timestamp = tx['time']
            fee = tx['fee']
            size = tx['size']

            input_addresses = [input_tx['prev_out']['addr']
                               for input_tx in tx['inputs']
                               if 'prev_out' in input_tx
                               and 'addr' in input_tx['prev_out']]
            output_addresses = [output_tx['addr'] for output_tx in tx['out'] if 'addr' in output_tx]

            for input_address in input_addresses:
                for output_address in output_addresses:
                    value = sum(output_tx['value']
                                for output_tx in tx['out']
                                if 'addr' in output_tx
                                and output_tx['addr'] == output_address)
                    edge_attrs = {
                        'amount': value,
                        'fee': fee,
                        'size': size,
                        'timestamp': timestamp
                    }
                    G.add_node(input_address)
                    G.add_node(output_address)
                    G.add_edge(input_address, output_address, **edge_attrs)
        return G

    def save_transaction_graph_to_gexf(self, G, filepath, label=None):
        print("label: ", label)
        if label is not None:
            print("Label is added")
            G.graph['name'] = label
        else:
            print("Default_label is added")
            G.graph['name'] = "default_label"
        nx.write_gexf(G, filepath )

    def load_transaction_graph_from_gexf(self, filepath):
        G = nx.read_gexf(filepath)
        label = G.graph.get('name')
        return G, label

    # def build_graph_from_files(self, filepath):
    #     # Read node and edge data from files
    #     nodes_df = pd.read_csv(filepath + '_nodes.csv')
    #     edges_df = pd.read_csv(filepath + '_edges.csv')
    #
    #     # Extract node IDs and edge information
    #     nodes = nodes_df['node_id'].tolist()
    #     edges = [(source, target) for source, target in zip(edges_df['source'], edges_df['target'])]
    #
    #     # Build adjacency matrix
    #     n_nodes = len(nodes)
    #     a = np.zeros((n_nodes, n_nodes))
    #     for source, target in edges:
    #         source_idx = nodes.index(source)
    #         target_idx = nodes.index(target)
    #         a[source_idx, target_idx] = 1
    #
    #     # Extract node features (dummy features for demonstration)
    #     x = torch.tensor(np.random.rand(n_nodes, 2), dtype=torch.float)
    #
    #     # Extract edge features (if available)
    #     edge_attrs = ['amount', 'fee', 'size', 'timestamp']
    #     edge_features = edges_df[edge_attrs].values
    #     e = torch.tensor(edge_features, dtype=torch.float)
    #
    #     # Create the PyTorch Geometric Data object
    #     data = Data(x=x, edge_index=torch.tensor(edges).t().contiguous(), edge_attr=e)
    #
    #     return data

    # def load_labels_from_file(self, filepath):
    #     with open(filepath + '_label.txt', 'r') as f:
    #         labels = f.read().splitlines()
    #     return labels

    def show(self, graph):
        layout = nx.spring_layout(graph)

        node_colors = ['skyblue' for _ in graph.nodes()]
        node_sizes = [10 for _ in graph.nodes()]

        nx.draw(graph, layout, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=10)
        edge_labels = nx.get_edge_attributes(graph, 'amount')
        nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, font_size=8)
        plt.title("Bitcoin Transaction Graph")
        plt.show()

    def get_transactions(self, address):
        url = f"https://blockchain.info/rawaddr/{address}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            transactions = data['txs']
            return transactions
        else:
            print("Error fetching data:", response.status_code)
            return None
