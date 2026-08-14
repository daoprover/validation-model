import logging
import os
from collections import Counter

import torch
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import LabelEncoder
import sys

from dataset.data_loader import GraphDatasetLoader
from dataset.group_split import grouped_train_validation_split, transaction_overlap
from models.gnn.gat.hyperparams import GatHyperParams

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))

from models.gnn.gat.model import GraphGATConv


class GatTrainer():
    def __init__(self,hyperparams: GatHyperParams, logger: logging.Logger):
        self.hyperparams = hyperparams
        self.logger = logger

    def train_gat(self):
        torch.manual_seed(self.hyperparams.training.seed)
        labels = ["anomaly", "white"]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        dataset = GraphDatasetLoader(base_dir=self.hyperparams.dataset.train, label_encoder=label_encoder, logger=self.logger)
        if len(dataset) == 0:
            raise ValueError(f"No .gexf training graphs found in {self.hyperparams.dataset.train}")

        fraction = float(self.hyperparams.dataset.dataset_fraction)
        if not 0 < fraction <= 1:
            raise ValueError("dataset_fraction must be in the interval (0, 1]")

        generator = torch.Generator().manual_seed(self.hyperparams.training.seed)
        selected_size = max(1, round(len(dataset) * fraction))
        selected_indices = torch.randperm(len(dataset), generator=generator)[:selected_size].tolist()

        validation_fraction = float(self.hyperparams.training.validation_split)
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_split must be in the interval [0, 1)")
        selected_metadata = [dataset.metadata(index) for index in selected_indices]
        validation_directory = getattr(self.hyperparams.dataset, "valid", None)
        has_external_validation = (
            validation_directory
            and os.path.isdir(validation_directory)
            and any(name.endswith(".gexf") for name in os.listdir(validation_directory))
        )
        if has_external_validation:
            validation_base_dataset = GraphDatasetLoader(
                base_dir=validation_directory,
                label_encoder=label_encoder,
                logger=self.logger,
            )
            validation_indices = list(range(len(validation_base_dataset)))
            validation_metadata = [
                validation_base_dataset.metadata(index) for index in validation_indices
            ]
            overlap = transaction_overlap(
                [item["transaction_ids"] for item in selected_metadata],
                [item["transaction_ids"] for item in validation_metadata],
            )
            if overlap:
                examples = ", ".join(sorted(overlap)[:3])
                raise ValueError(
                    f"Train/validation transaction leakage detected ({len(overlap)} shared; "
                    f"examples: {examples})"
                )
            if any(not item["transaction_ids"] for item in selected_metadata + validation_metadata):
                raise ValueError(
                    "Every graph must contain explicit transaction nodes; regenerate legacy graphs first"
                )
            training_indices = selected_indices
            training_metadata = selected_metadata
            training_dataset = torch.utils.data.Subset(dataset, training_indices)
            validation_dataset = torch.utils.data.Subset(
                validation_base_dataset, validation_indices
            )
        else:
            training_indices, validation_indices = grouped_train_validation_split(
                selected_indices,
                [item["label"] for item in selected_metadata],
                [item["transaction_ids"] for item in selected_metadata],
                validation_fraction,
                self.hyperparams.training.seed,
            )
            metadata_by_index = dict(zip(selected_indices, selected_metadata))
            training_metadata = [metadata_by_index[index] for index in training_indices]
            validation_metadata = [metadata_by_index[index] for index in validation_indices]
            training_dataset = torch.utils.data.Subset(dataset, training_indices)
            validation_dataset = torch.utils.data.Subset(dataset, validation_indices)

        validation_size = len(validation_indices)
        required_labels = set(label_encoder.classes_)
        training_label_counts = Counter(item["label"] for item in training_metadata)
        validation_label_counts = Counter(item["label"] for item in validation_metadata)
        if set(training_label_counts) != required_labels:
            raise ValueError(
                f"Training split must contain both classes; got {dict(training_label_counts)}"
            )
        if validation_indices and set(validation_label_counts) != required_labels:
            raise ValueError(
                f"Validation split must contain both classes; got {dict(validation_label_counts)}"
            )
        training_loader = DataLoader(
            training_dataset,
            batch_size=self.hyperparams.training.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.hyperparams.testing.batch_size,
            shuffle=False,
        ) if validation_size else None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_classes = len(label_encoder.classes_)
        model = GraphGATConv(in_channels=5, edge_in_channels=4, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.hyperparams.training.learning_rate, weight_decay=5e-4)
        if self.hyperparams.training.class_weighted_loss:
            total_training_samples = sum(training_label_counts.values())
            class_weights = torch.tensor(
                [
                    total_training_samples / (num_classes * training_label_counts[label])
                    for label in label_encoder.classes_
                ],
                dtype=torch.float,
                device=device,
            )
        else:
            class_weights = torch.ones(num_classes, dtype=torch.float, device=device)
        criterion = torch.nn.NLLLoss(weight=class_weights)
        reporting_criterion = torch.nn.NLLLoss(weight=class_weights, reduction="sum")

        def train(loader):
            i = 0
            model.train()
            total_loss = 0.0
            total_weight = 0.0
            for data in loader:

                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss encountered")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hyperparams.training.clipnorm)
                optimizer.step()
                total_loss += reporting_criterion(out.detach(), data.y).item()
                total_weight += class_weights[data.y].sum().item()
                del data
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                self.logger.info(f"step {i}")
                i += 1
            return total_loss / total_weight

        def evaluate(loader):
            model.eval()
            total_loss = 0.0
            total_weight = 0.0
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data)
                    loss = reporting_criterion(out, data.y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite validation loss encountered")
                    total_loss += loss.item()
                    total_weight += class_weights[data.y].sum().item()
            return total_loss / total_weight

        best_loss = float("inf")
        best_model_path = f"gat_model_best_{self.hyperparams.meta.version}.h5"
        history = []

        # model.load_state_dict(
        #     (torch.load(self.hyperparams.training., weights_only=True)))

        for epoch in range(self.hyperparams.training.epochs_number):
            self.logger.info("Starting epoch %s", epoch)
            training_loss = train(training_loader)
            selection_loss = evaluate(validation_loader) if validation_loader is not None else training_loss
            if selection_loss < best_loss:
                best_loss = selection_loss
                torch.save(model.state_dict(), best_model_path)

            self.logger.info(
                "Epoch %s, training loss: %.4f, selection loss: %.4f",
                epoch,
                training_loss,
                selection_loss,
            )
            history.append({
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": selection_loss if validation_loader is not None else None,
            })

        final_model_path = f"gat_model_{self.hyperparams.meta.version}.h5"
        torch.save(model.state_dict(), final_model_path)
        return {
            "best_loss": best_loss,
            "best_model_path": best_model_path,
            "final_model_path": final_model_path,
            "training_size": len(training_dataset),
            "validation_size": len(validation_dataset),
            "training_label_counts": dict(training_label_counts),
            "validation_label_counts": dict(validation_label_counts),
            "validation_source": "directory" if has_external_validation else "training_split",
            "history": history,
        }
