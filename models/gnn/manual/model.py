
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax


class GraphGNNWithEmbeddings(MessagePassing):
    def __init__(self, node_input_dim, edge_input_dim, embedding_dim, hidden_dim, num_time_labels):
        super(GraphGNNWithEmbeddings, self).__init__(aggr='add')  # Add aggregation
        self.node_time_embedding = nn.Embedding(num_time_labels, embedding_dim)
        self.edge_time_embedding = nn.Embedding(num_time_labels, embedding_dim)

        self.node_transform = nn.Linear(node_input_dim + embedding_dim, hidden_dim)
        self.edge_transform = nn.Linear(edge_input_dim + embedding_dim, hidden_dim)
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        if hasattr(data, 'node_time_label') and data.node_time_label is not None:
            node_time_label = data.node_time_label.long()
        else:
            node_time_label = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if hasattr(data, 'edge_time_label') and data.edge_time_label is not None:
            edge_time_label = data.edge_time_label.long()
        else:
            edge_time_label = torch.zeros(edge_attr.size(0), dtype=torch.long, device=x.device)

        embedded_node_time = self.node_time_embedding(node_time_label)
        embedded_edge_time = self.edge_time_embedding(edge_time_label)

        x = torch.cat([x, embedded_node_time], dim=-1)

        edge_attr = torch.cat([edge_attr, embedded_edge_time], dim=-1)
        x = self.node_transform(x)
        edge_attr = self.edge_transform(edge_attr)

        if edge_index.numel() > 0 and int(edge_index.max().item()) >= x.size(0):
            raise ValueError(
                f"Invalid edge_index: max index {edge_index.max().item()} >= number of nodes {x.size(0)}"
            )
        self._msg_num_nodes = int(x.size(0))
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        graph_embedding = global_mean_pool(x, batch)
        return self.graph_classifier(graph_embedding)

    def message(self, x_i, x_j, edge_attr, edge_index):
        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        attention_logits = self.attention_mlp(combined)
        attention = softmax(attention_logits, edge_index[1], num_nodes=self._msg_num_nodes)
        return attention * x_j

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(combined)
