"""Deterministic feature extraction shared by the graph datasets."""

import math

NODE_FEATURE_KEYS = (
    "total_sent",
    "total_received",
    "num_transactions",
    "total_fees",
    "avg_transaction_value",
)

EDGE_FEATURE_KEYS = ("amount", "fee", "size", "timestamp")


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _legacy_attvalues(attrs):
    """Read old hand-authored ``attvalues`` structures, if present."""
    values = attrs.get("attvalues")
    if isinstance(values, dict):
        return [_as_float(values[key]) for key in sorted(values)]
    if isinstance(values, (list, tuple)):
        parsed = []
        for value in values:
            if isinstance(value, dict):
                parsed.append(_as_float(value.get("value")))
            else:
                parsed.append(_as_float(value))
        return parsed
    return None


def feature_vector(attrs, keys):
    """Return a fixed-width numeric vector with stable, documented ordering."""
    if any(key in attrs for key in keys):
        return [_as_float(attrs.get(key)) for key in keys]

    legacy = _legacy_attvalues(attrs)
    if legacy is None:
        return [0.0] * len(keys)
    return (legacy + [0.0] * len(keys))[: len(keys)]


def node_feature_vector(attrs):
    return feature_vector(attrs, NODE_FEATURE_KEYS)


def edge_feature_vector(attrs):
    return feature_vector(attrs, EDGE_FEATURE_KEYS)


def _signed_log1p(value):
    return math.copysign(math.log1p(abs(value)), value)


def model_node_feature_vector(attrs):
    """Scale non-negative count/value features without fitting on test data."""
    return [_signed_log1p(value) for value in node_feature_vector(attrs)]


def model_edge_feature_vector(attrs):
    """Scale amount/fee/size logarithmically and Unix time to billions."""
    amount, fee, size, timestamp = edge_feature_vector(attrs)
    return [
        _signed_log1p(amount),
        _signed_log1p(fee),
        _signed_log1p(size),
        timestamp / 1_000_000_000.0,
    ]


def timestamp_bucket(timestamp):
    """Map a Unix timestamp to a UTC six-hour bucket."""
    try:
        hour = (float(timestamp) % 86400) // 3600
    except (TypeError, ValueError):
        return 3
    if 6 <= hour < 12:
        return 0
    if 12 <= hour < 18:
        return 1
    if 18 <= hour < 24:
        return 2
    return 3
