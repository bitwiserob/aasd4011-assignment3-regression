import torch
from torch import nn
import numpy as np

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    """
    learning_rate = (np.pi*np.e)/10000  # May need to be adjusted
    num_epochs = 1000     # Increased number of epochs




    input_features = X.shape[1]  # Number of features in X
    output_features = y.shape[1]
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Using Mean Squared Error for regression

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X,y,model,loss_fn,optimizer)
        if False:
            break
        previous_loss = loss.item()


        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model, loss