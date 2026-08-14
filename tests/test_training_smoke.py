import logging
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    __import__("torch_geometric")
    __import__("sklearn")
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch, PyG, and scikit-learn are not installed")
class GatTrainingSmokeTests(unittest.TestCase):
    def test_real_loader_forward_backward_and_checkpoint_cycle(self):
        from models.gnn.gat.model import GraphGATConv
        from models.gnn.gat.train_gat import GatTrainer
        from utils.graph import GraphHelper

        logger = logging.getLogger("training-smoke-test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_directory = root / "train"
            training_directory.mkdir()
            helper = GraphHelper(logger)

            for index in range(8):
                anomaly = index % 2 == 0
                amount = 10_000 if anomaly else 10
                transaction = {
                    "hash": f"tx-{index}",
                    "time": 1_700_000_000 + index,
                    "fee": 10 if anomaly else 1,
                    "size": 250,
                    "inputs": [{"prev_out": {"addr": f"sender-{index}", "value": amount + 1}}],
                    "out": [{"addr": f"receiver-{index}", "value": amount}],
                }
                graph = helper.build_transaction_graph([transaction])
                helper.save_transaction_graph_to_gexf(
                    graph,
                    training_directory / f"graph-{index}.gexf",
                    "ransomware" if anomaly else "white",
                )

            hyperparams = SimpleNamespace(
                meta=SimpleNamespace(version="smoke"),
                dataset=SimpleNamespace(train=str(training_directory), dataset_fraction=1.0),
                training=SimpleNamespace(
                    seed=42,
                    validation_split=0.25,
                    batch_size=2,
                    learning_rate=1e-3,
                    clipnorm=1.0,
                    epochs_number=10,
                    class_weighted_loss=True,
                ),
                testing=SimpleNamespace(batch_size=2, show_plots=False),
            )

            previous_directory = os.getcwd()
            os.chdir(root)
            try:
                result = GatTrainer(hyperparams, logger).train_gat()
                state = torch.load(result["best_model_path"], map_location="cpu", weights_only=True)
                repeated_result = GatTrainer(hyperparams, logger).train_gat()
                repeated_state = torch.load(
                    repeated_result["best_model_path"], map_location="cpu", weights_only=True
                )
            finally:
                os.chdir(previous_directory)

            model = GraphGATConv(in_channels=5, edge_in_channels=4, num_classes=2)
            model.load_state_dict(state)
            self.assertEqual(result["training_size"], 6)
            self.assertEqual(result["validation_size"], 2)
            self.assertEqual(len(result["history"]), 10)
            self.assertTrue(math.isfinite(result["best_loss"]))
            self.assertLess(
                min(item["training_loss"] for item in result["history"][1:]),
                result["history"][0]["training_loss"],
            )
            self.assertEqual(set(result["training_label_counts"]), {"anomaly", "white"})
            self.assertEqual(set(result["validation_label_counts"]), {"anomaly", "white"})
            self.assertEqual(result["history"], repeated_result["history"])
            for key in state:
                self.assertTrue(torch.equal(state[key], repeated_state[key]), key)

            validation_directory = root / "valid"
            validation_directory.mkdir()
            for index in range(4):
                anomaly = index % 2 == 0
                transaction = {
                    "hash": f"validation-tx-{index}",
                    "time": 1_700_100_000 + index,
                    "fee": 10 if anomaly else 1,
                    "size": 250,
                    "inputs": [{"prev_out": {"addr": f"v-sender-{index}", "value": 101}}],
                    "out": [{"addr": f"v-receiver-{index}", "value": 100}],
                }
                helper.save_transaction_graph_to_gexf(
                    helper.build_transaction_graph([transaction]),
                    validation_directory / f"graph-{index}.gexf",
                    "ransomware" if anomaly else "white",
                )

            hyperparams.dataset.valid = str(validation_directory)
            hyperparams.training.epochs_number = 1
            os.chdir(root)
            try:
                external_result = GatTrainer(hyperparams, logger).train_gat()
            finally:
                os.chdir(previous_directory)
            self.assertEqual(external_result["training_size"], 8)
            self.assertEqual(external_result["validation_size"], 4)
            self.assertEqual(external_result["validation_source"], "directory")

            test_directory = root / "test"
            test_directory.mkdir()
            for index in range(4):
                anomaly = index % 2 == 0
                transaction = {
                    "hash": f"test-tx-{index}",
                    "time": 1_700_200_000 + index,
                    "fee": 10 if anomaly else 1,
                    "size": 250,
                    "inputs": [{"prev_out": {"addr": f"t-sender-{index}", "value": 101}}],
                    "out": [{"addr": f"t-receiver-{index}", "value": 100}],
                }
                helper.save_transaction_graph_to_gexf(
                    helper.build_transaction_graph([transaction]),
                    test_directory / f"graph-{index}.gexf",
                    "ransomware" if anomaly else "white",
                )

            from models.gnn.gat.test_gat import TestGAT

            hyperparams.dataset.test = str(test_directory)
            hyperparams.testing.model_path = str(root / external_result["best_model_path"])
            metrics = TestGAT(hyperparams, logger).test()
            self.assertEqual(len(metrics["confusion_matrix"]), 2)
            self.assertTrue(math.isfinite(metrics["balanced_accuracy"]))
            self.assertTrue(math.isfinite(metrics["average_precision"]))


if __name__ == "__main__":
    unittest.main()
