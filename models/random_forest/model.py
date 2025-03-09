from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
import pandas as pd


class RandomFores:
    def __init__(self):
        self._model = tfdf.keras.RandomForestModel()

    def trine(self, dataset_path: Path ):
        train_df = pd.read_csv(dataset_path)
        test_df = pd.read_csv(dataset_path)

        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")
        test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="my_label")

        self._model.fit(train_ds)
        self._model.compile(loss="binary_crossentropy", optimizer="adam",  metrics=["accuracy"])
        self._model.evaluate(test_ds)

        self._model.save()

    def save(self,  path='./models/random_forest/weight/'):
        self._model.save()

    def summary(self):
        self._model.summary()

    def show_logs(self):

        logs = self._model.make_inspector().training_logs()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
        plt.xlabel("Number of trees")
        plt.ylabel("Accuracy (out-of-bag)")

        plt.subplot(1, 2, 2)
        plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
        plt.xlabel("Number of trees")
        plt.ylabel("Logloss (out-of-bag)")

        plt.show()

