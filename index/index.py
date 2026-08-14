import time as t
import csv
from pathlib import Path
import sys
import os
import logging

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils.graph import GraphHelper


class Indexer:
    def __init__(self, logger: logging.Logger, sleep_time=4):
        self.sleep_time = sleep_time
        self.logger = logger

    def index_white(self, save_path, block_numbers: list[int], tx_per_block: int = 50):
        graph_helper = GraphHelper(self.logger)
        os.makedirs(save_path, exist_ok=True)
        i = 0

        for block in block_numbers:
            addresses = graph_helper.get_white_addresses(block)[:tx_per_block]
            for address in addresses:
                self.__process_address_info(save_path, address, graph_helper, "white")

                self.logger.info(f"i: {i}", )
                i += 1
                t.sleep(self.sleep_time)

    def index_black_addresses(self, save_path: Path, csv_path: Path):
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)
            address_column = header.index("address") if "address" in header else 0
            label_column = header.index("label") if "label" in header else 9
            labeled_addresses = [
                (row[address_column], row[label_column])
                for row in reader
                if len(row) > max(address_column, label_column)
            ]
        graph_helper = GraphHelper(self.logger)
        os.makedirs(save_path, exist_ok=True)
        self.logger.info("Loaded %s labeled addresses", len(labeled_addresses))

        i = 0

        for address, label in labeled_addresses:
            self.__process_address_info(save_path, address, graph_helper, str(label))

            self.logger.debug("i: %s", i)
            i += 1
            t.sleep(self.sleep_time)

    def __process_address_info(
        self, save_path: Path, address: str, graph_helper: GraphHelper, label: str
    ):
        address_file = f"{save_path}/{address}.gexf"

        if os.path.exists(address_file):
            graph, existing_label = graph_helper.load_transaction_graph_from_gexf(address_file)
            if label == "white" and existing_label != "white":
                self.logger.info(
                    "Keeping existing anomaly label %s for %s", existing_label, address
                )
            elif existing_label != label:
                self.logger.info(
                    "Updating label for %s from %s to %s", address, existing_label, label
                )
                graph_helper.save_transaction_graph_to_gexf(graph, address_file, label)
            else:
                self.logger.debug("File for address %s already exists. Skipping...", address)
            return

        self.logger.debug("address: %s", address)
        transactions = graph_helper.get_transactions(address)

        if transactions:
            self.logger.debug("Transactions for address: %s", address)
            graph = graph_helper.build_transaction_graph(transactions)
            graph_helper.save_transaction_graph_to_gexf(graph, address_file, label)
        else:
            self.logger.error("No transactions found.")
