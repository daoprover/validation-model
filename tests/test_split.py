import tempfile
import unittest
from pathlib import Path

from dataset.split import split_dataset

try:
    __import__("networkx")
    networkx_available = True
except ImportError:
    networkx_available = False


@unittest.skipUnless(networkx_available, "networkx is not installed")
class DatasetSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_non_destructive_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            from utils.graph import GraphHelper

            helper = GraphHelper()
            for index in range(20):
                graph = helper.build_transaction_graph([{
                    "hash": f"tx-{index}",
                    "time": index,
                    "fee": 1,
                    "size": 100,
                    "inputs": [{"prev_out": {"addr": f"sender-{index}", "value": 11}}],
                    "out": [{"addr": f"receiver-{index}", "value": 10}],
                }])
                helper.save_transaction_graph_to_gexf(
                    graph,
                    source / f"{index:02}.gexf",
                    "ransomware" if index % 2 else "white",
                )

            counts = split_dataset(source, root / "output", seed=7)

            self.assertEqual(counts, {"train": 14, "valid": 3, "test": 3})
            self.assertEqual(len(list(source.glob("*.gexf"))), 20)
            output_names = {
                path.relative_to(root / "output").as_posix()
                for path in (root / "output").glob("*/*.gexf")
            }

            split_dataset(source, root / "second-output", seed=7)
            second_names = {
                path.relative_to(root / "second-output").as_posix()
                for path in (root / "second-output").glob("*/*.gexf")
            }
            self.assertEqual(output_names, second_names)


if __name__ == "__main__":
    unittest.main()
