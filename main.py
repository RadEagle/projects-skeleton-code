import os

import constants
from data.StartingDataset import StartingDataset
from data.TestingDataset import TestingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

import torch


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS_TOY, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", hyperparameters["epochs"])
    print("Batch size:", hyperparameters["batch_size"])

    # Initalize dataset and model. Then train the model!
    # size is somehow 313
    train_dataset = StartingDataset()
    val_dataset = TestingDataset()
    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device=device
    )


if __name__ == "__main__":
    main()
