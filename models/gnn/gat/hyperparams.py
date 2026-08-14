from __future__ import annotations
import json
from pathlib import Path


class GatHyperParams:
    """
    Class for getting all triplet_network's params
    """

    def __init__(self, path: Path = Path("./models/gnn/gat/hyperparams.json")) -> None:
        """Init class"""

        with open(path, "r") as json_file:
            hyperparams_dict = json.loads(json_file.read())

            self.meta = (
                Meta.default()
                if "meta" not in hyperparams_dict
                else Meta(hyperparams_dict["meta"])
            )
            self.dataset = (
                DatasetParams.default()
                if "dataset" not in hyperparams_dict
                else DatasetParams(hyperparams_dict["dataset"])
            )
            self.training = (
                TrainingParams.default()
                if "training" not in hyperparams_dict
                else TrainingParams(hyperparams_dict["training"])
            )
            self.testing = (
                TestingParams.default()
                if "testing" not in hyperparams_dict
                else TestingParams(hyperparams_dict["testing"])
            )


class Meta:
    """
    Class containing triplet_network's meta data.
    """

    def __init__(self, json_dictionary: dict) -> None:
        self.version = json_dictionary.get("version", "v1").replace(" ", "")
        self.save_path = json_dictionary.get("save_path", "./../../assets/models/").replace(
            " ", ""
        )

    @staticmethod
    def default() -> Meta:
        return Meta({"version": "v1"})


class DatasetParams:
    """
    Class containing parameters of the dataset.
    """

    def __init__(self, json_dictionary: dict) -> None:
        self.version = json_dictionary.get("version", "v1").replace(" ", "")
        self.test = json_dictionary.get(
            "test", "./../../assets/test/"
        ).replace(" ", "")
        self.train = json_dictionary.get(
            "train", "./../../assets/train/"
        ).replace(" ", "")
        self.valid = json_dictionary.get(
            "valid", "./../../assets/valid/"
        ).replace(" ", "")
        self.raw_dataset = json_dictionary.get("raw_dataset", "./../../assets/graphs/").replace(" ", "")
        self.dataset_fraction = json_dictionary.get("dataset_fraction", 1.0)
        self.start_block = json_dictionary.get("start_block", 0)
        self.end_block = json_dictionary.get("end_block", 0)
        self.step = json_dictionary.get("step", 0)
        self.txs_per_block = json_dictionary.get("txs_per_block", 0)
        self.raw_dataset_csv = json_dictionary.get("raw_dataset_csv", "./../../assets/train/BitcoinHeistData.csv").replace(" ", "")

    @staticmethod
    def default() -> DatasetParams:
        return DatasetParams({"version": "v1"})


class TrainingParams:
    """
    Class containing parameters of the training.
    """

    def __init__(self, json_dictionary: dict) -> None:
        """
        Initializes the TrainingParams instance.

        ### Args:
        - json_dictionary (dict): Dictionary corresponding to the "training" key in the hyperparameters json file
        """

        self.learning_rate = json_dictionary.get("learning_rate", 1e-4)
        self.batch_size = json_dictionary.get("batch_size", 32)
        self.epochs_number = json_dictionary.get("epochs_number", 10)
        self.validation_split = json_dictionary.get("validation_split", 0.2)
        self.verbose = json_dictionary.get("verbose", 1)
        self.clipnorm = json_dictionary.get("clipnorm", 1.0)
        self.seed = json_dictionary.get("seed", 42)
        self.class_weighted_loss = json_dictionary.get("class_weighted_loss", True)

    @staticmethod
    def default() -> TrainingParams:
        """
        Returns the default training parameters
        """
        return TrainingParams(
            {
                "learning_rate": 0.00001,
                "batch_size": 32,
                "epochs_number": 4,
                "validation_split": 0.2,
                "verbose": 1,
            }
        )


class TestingParams:
    """
    Class containing parameters of the training.
    """

    def __init__(self, json_dictionary: dict) -> None:
        """
        Initializes the TrainingParams instance.

        ### Args:
        - json_dictionary (dict): Dictionary corresponding to the "testing" key in the hyperparameters json file
        """

        self.model_path = json_dictionary.get(
            "model_path", "./../../assets/model.h5"
        ).replace(" ", "")
        self.batch_size = json_dictionary.get("batch_size", 32)
        self.verbose = json_dictionary.get("verbose", 1)
        self.show_plots = json_dictionary.get("show_plots", False)

    @staticmethod
    def default() -> TrainingParams:
        """
        Returns the default training parameters
        """
        return TestingParams(
            {
                "batch_size": 32,
                "verbose": 1,
            }
        )
