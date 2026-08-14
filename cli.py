import typer
import sys
import os
from enum import IntEnum
import logging

from rich.logging import RichHandler
import absl.logging

from typing_extensions import Annotated

from index.index import Indexer
from models.gnn.gat.hyperparams import GatHyperParams
from models.gnn.gat.test_gat import TestGAT
from models.gnn.gat.train_gat import GatTrainer
from models.gnn.manual.train import ManualGNNTrainer
from utils.graph import GraphHelper

sys.path.insert(1, os.path.join(sys.path[0], "../.."))


app = typer.Typer(help="CLI for indexing Bitcoin graphs and training GNN classifiers")


class VerboseMode(IntEnum):
    """
    Enum for logging verbosity
    """

    WARNING = 0
    INFO = 1
    DEBUG = 2

    def log_level(self) -> str:
        """
        Returns the string representation of the enum
        """
        return {0: "WARNING", 1: "INFO", 2: "DEBUG"}.get(self.value, "INFO")

@app.command()
def train_manual(
        verbose: Annotated[
            int,
            typer.Option(
                help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
            ),
        ] = 1,
) -> None:
    """
    Train the model
    ### Args:
        See the help for each argument
    """

    print("train_manual")

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        trainer = ManualGNNTrainer(  logger)
        trainer.train()

    except Exception as e:
        logger.exception("Error while training the model. Aborting...")
        raise typer.Exit(code=1) from e

    logger.info("Done! Exiting...")


@app.command()
def index_white_addresses(
        verbose: Annotated[
            int,
            typer.Option(
                help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
            ),
        ] = 1,
) -> None:
    """
    Insert new transactions in dataset
    ### Args:
        See the help for each argument
    """

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        hyperparams = GatHyperParams()
    except Exception as e:
        logger.exception("Failed to initialize the ModelTester instance. Aborting...")
        raise typer.Exit(code=1) from e

    try:
        indexer = Indexer(logger)
        indexer.index_white(
            hyperparams.dataset.raw_dataset,
            list(range(hyperparams.dataset.start_block, hyperparams.dataset.end_block, hyperparams.dataset.step)),
            hyperparams.dataset.txs_per_block
        )

    except Exception as e:
        logger.exception("Error while indexing new transactions. Aborting...")
        raise typer.Exit(code=1) from e

    logger.info("Done! Exiting...")


@app.command()
def index_marked_addresses(
        verbose: Annotated[
            int,
            typer.Option(
                help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
            ),
        ] = 1,
) -> None:
    """
    Insert new transactions in dataset
    ### Args:
        See the help for each argument
    """

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        hyperparams = GatHyperParams()
    except Exception as e:
        logger.exception("Failed to initialize the ModelTester instance. Aborting...")
        raise typer.Exit(code=1) from e

    try:
        indexer = Indexer(logger)
        indexer.index_black_addresses(hyperparams.dataset.raw_dataset, hyperparams.dataset.raw_dataset_csv)

    except Exception as e:
        logger.exception("Error while indexing new transactions. Aborting...")
        raise typer.Exit(code=1) from e

    logger.info("Done! Exiting...")

@app.command()
def test_gat(
        verbose: Annotated[
            int,
            typer.Option(
                help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
            ),
        ] = 1,
) -> None:
    """
    Test the model
    ### Args:
        See the help for each argument
    """

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        hyperparams = GatHyperParams()
    except Exception as e:
        logger.exception("Failed to initialize the ModelTester instance. Aborting...")
        raise typer.Exit(code=1) from e

    try:
        indexer = TestGAT(hyperparams,  logger)
        indexer.test()

    except Exception as e:
        logger.exception("Error while testing the model. Aborting...")
        raise typer.Exit(code=1) from e

    logger.info("Done! Exiting...")

@app.command()
def train_gat(
        verbose: Annotated[
            int,
            typer.Option(
                help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
            ),
        ] = 1,
) -> None:
    """
    Train the model
    ### Args:
        See the help for each argument
    """

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        hyperparams = GatHyperParams()
    except Exception as e:
        logger.exception("Failed to initialize the ModelTester instance. Aborting...")
        raise typer.Exit(code=1) from e

    try:
        indexer = GatTrainer(hyperparams,  logger)
        indexer.train_gat()

    except Exception as e:
        logger.exception("Error while training the model. Aborting...")
        raise typer.Exit(code=1) from e

    logger.info("Done! Exiting...")


@app.command()
def rebuild_graph(verbose: Annotated[
    int,
    typer.Option(
        help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
    ),
] = 1,):

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        hyperparams = GatHyperParams()
        gh = GraphHelper(logger)
    except Exception as e:
        logger.exception("Failed to initialize the ModelTester instance. Aborting...")
        raise typer.Exit(code=1) from e

    for file_path in [f for f in os.listdir(hyperparams.dataset.train) if f.endswith('.gexf')]:
        filepath = os.path.join(hyperparams.dataset.train, file_path)
        new_path = os.path.join(hyperparams.dataset.raw_dataset, file_path)
        gh.rebuild_transaction_graph(filepath,  new_path)




@app.command()
def check(verbose: Annotated[
    int,
    typer.Option(
        help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
    ),
] = 1,):

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        hyperparams = GatHyperParams()
        gh = GraphHelper(logger)

    except Exception as e:
        logger.exception("Failed to initialize the ModelTester instance. Aborting...")
        raise typer.Exit(code=1) from e

    for file_path in [f for f in os.listdir(hyperparams.dataset.train) if f.endswith('.gexf')]:
        g,  label = gh.load_transaction_graph_from_gexf( os.path.join(hyperparams.dataset.train, file_path))
        logger.info(f"LABEL: {label}",  )

def _get_logger(level="INFO") -> logging.Logger:
    """
    Configures the logging, and returns the logger instance

    ### Args:
    - level (str, optional): Logging level. Defaults to 'INFO'.

    ### Returns:
        logging.Logger: Instance of the logger
    """

    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    absl.logging.set_verbosity(absl.logging.ERROR)  # Disabling the TensorFlow warnings

    return logging.getLogger("rich")


if __name__ == "__main__":
    app()
