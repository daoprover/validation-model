import logging
import unittest

try:
    import torch
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and scikit-learn are not installed")
class TesterMetricTests(unittest.TestCase):
    def test_anomaly_class_zero_is_used_as_the_positive_class(self):
        from tester.tester import Tester

        class Batch:
            def __init__(self):
                self.y = torch.tensor([0, 1, 0, 1])
                self.log_probabilities = torch.log(torch.tensor([
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.8, 0.2],
                    [0.1, 0.9],
                ]))

            def to(self, device):
                self.y = self.y.to(device)
                self.log_probabilities = self.log_probabilities.to(device)
                return self

        class Model(torch.nn.Module):
            def forward(self, batch):
                return batch.log_probabilities

        class Loader(list):
            @property
            def dataset(self):
                return range(4)

        encoder = LabelEncoder().fit(["anomaly", "white"])
        metrics = Tester(
            device=torch.device("cpu"),
            model=Model(),
            logger=logging.getLogger("tester-metrics"),
        ).test(Loader([Batch()]), encoder)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["anomaly_precision"], 1.0)
        self.assertEqual(metrics["anomaly_recall"], 1.0)
        self.assertEqual(metrics["anomaly_f1"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
