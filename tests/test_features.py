import math
import unittest

from utils.features import (
    edge_feature_vector,
    model_edge_feature_vector,
    model_node_feature_vector,
    node_feature_vector,
    timestamp_bucket,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_named_node_attributes_have_stable_order(self):
        attrs = {
            "avg_transaction_value": 5,
            "total_fees": 4,
            "num_transactions": 3,
            "total_received": 2,
            "total_sent": 1,
        }
        self.assertEqual(node_feature_vector(attrs), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_missing_features_are_zero_not_random(self):
        self.assertEqual(node_feature_vector({}), [0.0] * 5)
        self.assertEqual(edge_feature_vector({}), [0.0] * 4)

    def test_legacy_attvalues_remain_supported(self):
        attrs = {"attvalues": [{"value": "1"}, {"value": "2"}]}
        self.assertEqual(edge_feature_vector(attrs), [1.0, 2.0, 0.0, 0.0])

    def test_timestamp_uses_utc_six_hour_buckets(self):
        self.assertEqual(timestamp_bucket(7 * 3600), 0)
        self.assertEqual(timestamp_bucket(13 * 3600), 1)
        self.assertEqual(timestamp_bucket(19 * 3600), 2)
        self.assertEqual(timestamp_bucket(None), 3)

    def test_model_features_scale_large_values_deterministically(self):
        node = model_node_feature_vector({"total_sent": 1_000_000_000})
        edge = model_edge_feature_vector({"amount": 99, "timestamp": 1_700_000_000})
        self.assertAlmostEqual(node[0], math.log1p(1_000_000_000))
        self.assertAlmostEqual(edge[0], math.log1p(99))
        self.assertEqual(edge[3], 1.7)


if __name__ == "__main__":
    unittest.main()
