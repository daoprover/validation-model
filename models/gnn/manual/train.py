import logging
import os

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder

import sys


sys.path.insert(1, os.path.join(sys.path[0], "../../.."))
from models.gnn.manual.data_loader import GraphDatasetLoader
from models.gnn.manual.model import GraphGNNWithEmbeddings


class ManualGNNTrainer():
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def train(self):

        labels = ["anomaly", "white"]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        dataset = GraphDatasetLoader(
            base_dir='data/train', label_encoder=label_encoder, logger=self.logger
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        num_time_labels = 4
        embedding_dim = 8
        hidden_dim = 16
        model = GraphGNNWithEmbeddings(
            node_input_dim=5,
            edge_input_dim=4,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_time_labels=num_time_labels
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.BCEWithLogitsLoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        self.logger.info("device: %s", device)

        epochs = 50
        for epoch in range(epochs):
            self.logger.info("Epoch %s/%s", epoch + 1, epochs)
            model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                try:
                    output = model(batch)
                except RuntimeError as e:
                    self.logger.warning("RuntimeError in batch: %s", e)
                    continue
                y = batch.y.float().view(-1, 1)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            denom = n_batches if n_batches else 1
            self.logger.info("Epoch %s/%s, Loss: %.4f", epoch + 1, epochs, total_loss / denom)

        torch.save(model.state_dict(), "model.h5")

        test_dir = 'data/test'
        if os.path.isdir(test_dir) and any(f.endswith('.gexf') for f in os.listdir(test_dir)):
            test_ds = GraphDatasetLoader(
                base_dir=test_dir, label_encoder=label_encoder, logger=self.logger
            )
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        else:
            self.logger.warning("No data/test with .gexf files; skipping eval on held-out data.")
            test_loader = None

        model.eval()
        with torch.no_grad():
            if test_loader is not None:
                for batch in test_loader:
                    batch = batch.to(device)
                    test_output = model(batch)
                    self.logger.debug("Predicted graph label logits: %s", test_output)
            else:
                for batch in loader:
                    batch = batch.to(device)
                    test_output = model(batch)
                    self.logger.debug("Predicted graph label logits (train loader): %s", test_output)
                    break
