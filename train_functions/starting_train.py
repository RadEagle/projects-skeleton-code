import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# extra imports
from torch.utils.tensorboard import SummaryWriter


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            model.train()
            images, labels = batch
            #print("Data present")
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #print("Results calculated")
            loss = loss_fn(outputs, labels)
            #print("loss calculated")
            loss.backward()       # Compute gradients
            #print("back propagation")
            optimizer.step()      # Update all the weights with the gradients you just calculated
            #print("step")
            optimizer.zero_grad() # Clear gradients before next iteration
            #print("zero_grad")

            # Periodically evaluate our model + log to Tensorboard
            if ((step % n_eval == 0) and (step != 0)):
                #print("Enters if statement")
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                accuracy = compute_accuracy(outputs, labels)
                ###Log results to Tensorboard??!!!?
                # https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
                writer = SummaryWriter()
                writer.add_scalar("Accuracy", accuracy)
                writer.flush()
                writer.close()
                #print("Closed")


                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                model.eval()
                evaluate(val_loader, model, loss_fn, device)
                model.train()
                print("end of if")

            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    print("Start evaluate")
    writer = SummaryWriter()
    print(len(val_loader))
    with torch.no_grad():
        for batch in tqdm(val_loader):
            #print("start batch")
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            #accuracy = compute_accuracy(outputs, labels)

            # writer.add_scalar("Loss and Accuracy", loss, accuracy)
            #print("end batch")

    writer.flush()
    writer.close()

