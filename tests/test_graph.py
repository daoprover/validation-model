import tempfile
import unittest
from pathlib import Path

try:
    import networkx  # noqa: F401
except ImportError:
    networkx = None


@unittest.skipIf(networkx is None, "networkx is not installed")
class TransactionGraphTests(unittest.TestCase):
    def setUp(self):
        from utils.graph import GraphHelper

        self.helper = GraphHelper()

    @staticmethod
    def transaction(tx_hash="tx-1"):
        return {
            "hash": tx_hash,
            "time": 3600,
            "fee": 5,
            "size": 250,
            "inputs": [
                {"prev_out": {"addr": "alice", "value": 70}},
                {"prev_out": {"addr": "bob", "value": 30}},
            ],
            "out": [
                {"addr": "carol", "value": 90},
                {"addr": "change", "value": 5},
            ],
        }

    def test_transaction_node_prevents_cartesian_product_inflation(self):
        graph = self.helper.build_transaction_graph([self.transaction()])

        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 4)
        self.assertEqual(graph.nodes["address:alice"]["total_sent"], 70)
        self.assertEqual(graph.nodes["address:bob"]["total_sent"], 30)
        self.assertEqual(graph.nodes["address:carol"]["total_received"], 90)

        incoming = graph.in_edges("transaction:tx-1", data=True)
        outgoing = graph.out_edges("transaction:tx-1", data=True)
        self.assertEqual(sum(edge[2]["amount"] for edge in incoming), 100)
        self.assertEqual(sum(edge[2]["amount"] for edge in outgoing), 95)
        self.assertAlmostEqual(sum(edge[2]["fee"] for edge in incoming), 5)

    def test_repeated_address_relationships_are_preserved_by_transaction_nodes(self):
        graph = self.helper.build_transaction_graph([
            self.transaction("tx-1"),
            self.transaction("tx-2"),
        ])
        self.assertTrue(graph.has_edge("address:alice", "transaction:tx-1"))
        self.assertTrue(graph.has_edge("address:alice", "transaction:tx-2"))
        self.assertEqual(graph.nodes["address:alice"]["total_sent"], 140)
        self.assertEqual(graph.nodes["address:alice"]["num_transactions"], 2)

    def test_gexf_round_trip_preserves_label_and_numeric_features(self):
        graph = self.helper.build_transaction_graph([self.transaction()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.gexf"
            self.helper.save_transaction_graph_to_gexf(graph, path, "ransomware")
            restored, label = self.helper.load_transaction_graph_from_gexf(path)

        self.assertEqual(label, "ransomware")
        self.assertEqual(float(restored.nodes["address:alice"]["total_sent"]), 70)
        self.assertEqual(float(restored["address:alice"]["transaction:tx-1"]["amount"]), 70)


if __name__ == "__main__":
    unittest.main()
