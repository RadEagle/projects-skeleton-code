import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gc

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
    torch.cuda.empty_cache()
    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders,   

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), lr = 0.0005, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    step = 0


    ##This line for local tensorboard
    writer = SummaryWriter()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader, position=0, leave=True):
            # Backpropagation and gradient descent
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Clear gradients before next iteration

            outputs = model(images)
            labels = torch.tensor(labels, dtype=torch.long)

            loss = loss_fn(outputs, labels)
            outputs = torch.argmax(outputs, dim=1)
            outputs = torch.tensor(outputs, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.float)

            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated
 

            # Periodically evaluate our model + log to Tensorboard
            # after every 100 steps this goes inside number of batches 
            if ((step % n_eval == 0) and (step != 0)):
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                accuracy = compute_accuracy(outputs, labels)

                #Log results to Tensorboard

                writer.add_scalar("Accuracy", accuracy)
                writer.add_scalar("Loss", loss)

                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                model.eval()

                
                evaluate(val_loader, model, loss_fn, device, writer)

                model.train()

            step += 1
        
    evaluate(val_loader, model, loss_fn, device, writer)



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


def evaluate(val_loader, model, loss_fn, device, writer):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    print(len(val_loader))
    with torch.no_grad():
        for batch in tqdm(val_loader, position=0, leave=False):

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            outputs = torch.argmax(outputs, dim=1)
            outputs = torch.tensor(outputs, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.float)
            accuracy = compute_accuracy(outputs, labels)

            writer.add_scalar("Eval Accuracy", accuracy)
            writer.add_scalar("Eval Loss", loss)


