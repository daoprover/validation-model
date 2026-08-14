import os
import sys
from torch_geometric.loader import DataLoader
import torch
import logging
from sklearn.preprocessing import LabelEncoder

from dataset.data_loader import GraphDatasetLoader
from dataset.group_split import transaction_overlap
from models.gnn.gat.hyperparams import GatHyperParams
from tester.tester import Tester

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGATConv


class TestGAT(Tester):
    def __init__(self, hyperparams: GatHyperParams, logger: logging.Logger, labels=None):
        if labels is None:
            labels = ["anomaly", "white"]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.hyperparams = hyperparams
        dataset = GraphDatasetLoader(base_dir=hyperparams.dataset.test, label_encoder=self.label_encoder, logger=logger)
        if len(dataset) == 0:
            raise ValueError(f"No .gexf test graphs found in {hyperparams.dataset.test}")
        if os.path.isdir(hyperparams.dataset.train):
            training_dataset = GraphDatasetLoader(
                base_dir=hyperparams.dataset.train,
                label_encoder=self.label_encoder,
                logger=logger,
            )
            overlap = transaction_overlap(
                [training_dataset.metadata(index)["transaction_ids"] for index in range(len(training_dataset))],
                [dataset.metadata(index)["transaction_ids"] for index in range(len(dataset))],
            )
            if overlap:
                examples = ", ".join(sorted(overlap)[:3])
                raise ValueError(
                    f"Train/test transaction leakage detected ({len(overlap)} shared; examples: {examples})"
                )
        self.loader = DataLoader(dataset, batch_size=self.hyperparams.testing.batch_size, shuffle=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = GraphGATConv(in_channels=5, edge_in_channels=4, num_classes=len(labels)).to(self.device)
        self.model.load_state_dict(torch.load(
            self.hyperparams.testing.model_path,
            map_location=self.device,
            weights_only=True,
        ))
        self.model.eval()
        self.logger = logger

        super().__init__(device=self.device, logger=self.logger, model=self.model)

    def test(self):
        return super().test(
            self.loader,
            self.label_encoder,
            show_plots=self.hyperparams.testing.show_plots,
        )
