import unittest

from dataset.group_split import grouped_train_validation_split, transaction_overlap


class GroupSplitTests(unittest.TestCase):
    def test_shared_transactions_never_cross_the_split(self):
        indices = list(range(6))
        labels = ["white", "white", "anomaly", "anomaly", "white", "anomaly"]
        transactions = [
            {"a", "shared"},
            {"b", "shared"},
            {"c"},
            {"d"},
            {"e"},
            {"f"},
        ]

        train, validation = grouped_train_validation_split(
            indices, labels, transactions, validation_fraction=0.33, seed=42
        )

        self.assertFalse(transaction_overlap(
            [transactions[index] for index in train],
            [transactions[index] for index in validation],
        ))
        self.assertEqual(set(train) | set(validation), set(indices))

    def test_fully_connected_samples_fail_instead_of_leaking(self):
        with self.assertRaisesRegex(ValueError, "leakage-free"):
            grouped_train_validation_split(
                [0, 1], ["white", "anomaly"], [{"shared"}, {"shared"}], 0.5, 42
            )

    def test_legacy_graphs_without_transaction_nodes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "transaction nodes"):
            grouped_train_validation_split([0, 1], ["white", "anomaly"], [set(), {"tx"}], 0.5, 42)


if __name__ == "__main__":
    unittest.main()
