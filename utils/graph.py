import logging
import os
from collections import defaultdict

import networkx as nx


class GraphHelper:
    """Build and persist address/transaction graphs for anomaly classification."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _number(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _address_node(address):
        return f"address:{address}"

    @staticmethod
    def _new_address_attributes():
        return {
            "node_type": 0,
            "total_sent": 0.0,
            "total_received": 0.0,
            "num_transactions": 0.0,
            "total_fees": 0.0,
            "avg_transaction_value": 0.0,
            "last_transaction_time": 0.0,
        }

    def build_transaction_graph(self, transactions):
        """Build a lossless address -> transaction -> address flow graph.

        Bitcoin does not define a unique mapping from individual inputs to
        individual outputs. A transaction node therefore avoids inventing that
        mapping and prevents the Cartesian-product inflation of the old graph.
        """
        graph = nx.DiGraph()

        for tx_index, tx in enumerate(transactions):
            timestamp = self._number(tx.get("time"))
            fee = self._number(tx.get("fee"))
            size = self._number(tx.get("size"))

            inputs = defaultdict(float)
            for tx_input in tx.get("inputs", []):
                previous_output = tx_input.get("prev_out") or {}
                address = previous_output.get("addr")
                if address:
                    inputs[address] += self._number(previous_output.get("value"))

            outputs = defaultdict(float)
            for tx_output in tx.get("out", []):
                address = tx_output.get("addr")
                if address:
                    outputs[address] += self._number(tx_output.get("value"))

            tx_id = tx.get("hash") or tx.get("tx_index") or tx_index
            tx_node = f"transaction:{tx_id}"
            if graph.has_node(tx_node):
                tx_node = f"{tx_node}:{tx_index}"

            total_input = sum(inputs.values())
            total_output = sum(outputs.values())
            graph.add_node(
                tx_node,
                node_type=1,
                total_sent=total_output,
                total_received=total_input,
                num_transactions=1.0,
                total_fees=fee,
                avg_transaction_value=(total_output / len(outputs)) if outputs else 0.0,
                last_transaction_time=timestamp,
            )

            participating_addresses = set(inputs) | set(outputs)
            for address in participating_addresses:
                address_node = self._address_node(address)
                if not graph.has_node(address_node):
                    graph.add_node(address_node, **self._new_address_attributes())

                attrs = graph.nodes[address_node]
                attrs["total_sent"] += inputs.get(address, 0.0)
                attrs["total_received"] += outputs.get(address, 0.0)
                attrs["num_transactions"] += 1.0
                attrs["last_transaction_time"] = max(attrs["last_transaction_time"], timestamp)

            input_count = len(inputs)
            for address, amount in inputs.items():
                if total_input > 0:
                    fee_share = fee * amount / total_input
                else:
                    fee_share = fee / input_count if input_count else 0.0
                address_node = self._address_node(address)
                graph.nodes[address_node]["total_fees"] += fee_share
                graph.add_edge(
                    address_node,
                    tx_node,
                    amount=amount,
                    fee=fee_share,
                    size=size,
                    timestamp=timestamp,
                )

            for address, amount in outputs.items():
                graph.add_edge(
                    tx_node,
                    self._address_node(address),
                    amount=amount,
                    fee=0.0,
                    size=size,
                    timestamp=timestamp,
                )

        for _, attrs in graph.nodes(data=True):
            count = attrs["num_transactions"]
            attrs["avg_transaction_value"] = (
                (attrs["total_sent"] + attrs["total_received"]) / count if count else 0.0
            )

        return graph

    def save_transaction_graph_to_gexf(self, graph, filepath, label=None):
        label = label or "white"
        self.logger.debug("Saving graph label: %s", label)
        graph.graph["name"] = label
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        nx.write_gexf(graph, filepath)

    def rebuild_transaction_graph(self, filepath, new_path):
        """Recalculate aggregate features in a legacy address-to-address graph."""
        graph, label = self.load_transaction_graph_from_gexf(filepath)
        for node in graph.nodes():
            graph.nodes[node].update(self._new_address_attributes())

        for sender, receiver, data in graph.edges(data=True):
            value = self._number(data.get("amount"))
            fee = self._number(data.get("fee"))
            timestamp = self._number(data.get("timestamp"))

            graph.nodes[sender]["total_sent"] += value
            graph.nodes[sender]["total_fees"] += fee
            graph.nodes[sender]["num_transactions"] += 1.0
            graph.nodes[sender]["last_transaction_time"] = max(
                graph.nodes[sender]["last_transaction_time"], timestamp
            )

            graph.nodes[receiver]["total_received"] += value
            graph.nodes[receiver]["num_transactions"] += 1.0
            graph.nodes[receiver]["last_transaction_time"] = max(
                graph.nodes[receiver]["last_transaction_time"], timestamp
            )

        for _, attrs in graph.nodes(data=True):
            count = attrs["num_transactions"]
            attrs["avg_transaction_value"] = (
                (attrs["total_sent"] + attrs["total_received"]) / count if count else 0.0
            )

        self.save_transaction_graph_to_gexf(graph, new_path, label)
        self.logger.info("Rebuilt legacy graph %s as %s", filepath, new_path)

    def load_transaction_graph_from_gexf(self, filepath):
        graph = nx.read_gexf(filepath)
        return graph, graph.graph.get("name")

    def show(self, graph):
        import matplotlib.pyplot as plt

        layout = nx.spring_layout(graph)
        nx.draw(
            graph,
            layout,
            with_labels=True,
            node_size=[10 for _ in graph.nodes()],
            node_color=["skyblue" for _ in graph.nodes()],
            font_size=10,
        )
        edge_labels = nx.get_edge_attributes(graph, "amount")
        nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, font_size=8)
        plt.title("Bitcoin Address/Transaction Graph")
        plt.show()

    def get_transactions(self, address):
        import requests

        url = f"https://blockchain.info/rawaddr/{address}"
        transactions = []
        offset = 0
        expected_count = None
        while expected_count is None or offset < expected_count:
            try:
                response = requests.get(
                    url,
                    params={"offset": offset, "limit": 50},
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                self.logger.error("Error fetching transactions for %s: %s", address, exc)
                return []

            payload = response.json()
            page = payload.get("txs", [])
            expected_count = int(payload.get("n_tx", len(page)))
            transactions.extend(page)
            if not page:
                break
            offset += len(page)
        return transactions

    def get_white_addresses(self, block_id):
        import requests

        url = f"https://blockchain.info/rawblock/{block_id}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            self.logger.error("Error fetching block %s: %s", block_id, exc)
            return []

        addresses = []
        seen = set()
        for tx in response.json().get("tx", []):
            candidates = [
                tx_input.get("prev_out", {}).get("addr")
                for tx_input in tx.get("inputs", [])
            ]
            candidates.extend(output.get("addr") for output in tx.get("out", []))
            for address in candidates:
                if address and address not in seen:
                    seen.add(address)
                    addresses.append(address)
        return addresses
