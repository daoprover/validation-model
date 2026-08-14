import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from index.index import Indexer


class FakeGraphHelper:
    saved_labels = []
    existing_label = "white"

    def __init__(self, logger=None):
        self.logger = logger

    def get_transactions(self, address):
        return [{"address": address}]

    def build_transaction_graph(self, transactions):
        return {"transactions": transactions}

    def save_transaction_graph_to_gexf(self, graph, path, label):
        self.saved_labels.append(label)

    def load_transaction_graph_from_gexf(self, path):
        return {"existing": True}, self.existing_label


class IndexerTests(unittest.TestCase):
    def setUp(self):
        FakeGraphHelper.saved_labels = []
        FakeGraphHelper.existing_label = "white"

    def test_csv_labels_are_propagated_to_saved_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "labels.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["address", "label"])
                writer.writerow(["address-1", "ransomware"])

            with patch("index.index.GraphHelper", FakeGraphHelper):
                Indexer(logger=unittest.mock.Mock(), sleep_time=0).index_black_addresses(
                    Path(directory) / "graphs", csv_path
                )

        self.assertEqual(FakeGraphHelper.saved_labels, ["ransomware"])

    def test_white_indexing_does_not_downgrade_an_existing_anomaly_label(self):
        FakeGraphHelper.existing_label = "ransomware"
        with tempfile.TemporaryDirectory() as directory:
            graph_directory = Path(directory) / "graphs"
            graph_directory.mkdir()
            graph_path = graph_directory / "address-1.gexf"
            graph_path.write_text("existing", encoding="utf-8")

            indexer = Indexer(logger=unittest.mock.Mock(), sleep_time=0)
            indexer._Indexer__process_address_info(
                graph_directory,
                "address-1",
                FakeGraphHelper(),
                "white",
            )

        self.assertEqual(FakeGraphHelper.saved_labels, [])

    def test_existing_graph_is_relabeled_when_csv_label_is_stronger(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_directory = Path(directory) / "graphs"
            graph_directory.mkdir()
            (graph_directory / "address-1.gexf").write_text("existing", encoding="utf-8")
            csv_path = Path(directory) / "labels.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["address", "label"])
                writer.writerow(["address-1", "ransomware"])

            with patch("index.index.GraphHelper", FakeGraphHelper):
                Indexer(logger=unittest.mock.Mock(), sleep_time=0).index_black_addresses(
                    graph_directory, csv_path
                )

        self.assertEqual(FakeGraphHelper.saved_labels, ["ransomware"])


if __name__ == "__main__":
    unittest.main()
