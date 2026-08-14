import logging

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
import numpy as np


class Tester:
    def __init__(self, device: torch.device, model: torch.nn.Module, logger: logging.Logger):
        self.device = device
        self.model = model
        self.logger = logger

    @staticmethod
    def _probs_and_preds(out: torch.Tensor):
        if out.dim() == 2 and out.size(1) == 1:
            prob = torch.sigmoid(out)
            preds = (prob > 0.5).long().squeeze(1)
        else:
            prob = torch.exp(out)
            preds = out.argmax(dim=1)
        return prob, preds

    def test(self, loader: DataLoader, label_encoder: LabelEncoder, show_plots=False):
        if len(loader.dataset) == 0:
            raise ValueError("Cannot evaluate an empty dataset")
        correct = 0
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                prob, preds = self._probs_and_preds(out)
                y = data.y.view(-1).long()
                correct += int((preds == y).sum())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())

        accuracy = correct / len(loader.dataset)
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        self.logger.info("Accuracy: %.4f; balanced accuracy: %.4f", accuracy, balanced_accuracy)

        class_ids = np.arange(len(label_encoder.classes_))
        cm = confusion_matrix(all_labels, all_preds, labels=class_ids)
        if show_plots:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = class_ids
            plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
            plt.yticks(tick_marks, label_encoder.classes_)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
            plt.close()

        self.logger.debug(classification_report(
            all_labels,
            all_preds,
            labels=class_ids,
            target_names=label_encoder.classes_,
            zero_division=0,
        ))

        all_probs = np.array(all_probs)
        if all_probs.ndim == 1:
            all_probs = all_probs.reshape(-1, 1)
        all_labels_np = np.array(all_labels)

        if len(class_ids) != 2 or len(np.unique(all_labels_np)) < 2:
            self.logger.warning("ROC AUC requires a binary test set containing both classes; skipping.")
            return {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "confusion_matrix": cm.tolist(),
                "roc_auc": None,
                "average_precision": None,
            }

        anomaly_id = int(label_encoder.transform(["anomaly"])[0])
        anomaly_scores = all_probs[:, anomaly_id]
        y_true_anomaly = (all_labels_np == anomaly_id).astype(int)
        y_pred_anomaly = (np.array(all_preds) == anomaly_id).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_anomaly,
            y_pred_anomaly,
            average="binary",
            zero_division=0,
        )

        fpr, tpr, _ = roc_curve(y_true_anomaly, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        average_precision = average_precision_score(y_true_anomaly, anomaly_scores)

        if show_plots:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(fpr, tpr, label=f'Anomaly ROC (AUC = {roc_auc:0.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.show()
            plt.close()
        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "anomaly_precision": float(precision),
            "anomaly_recall": float(recall),
            "anomaly_f1": float(f1),
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "confusion_matrix": cm.tolist(),
        }
