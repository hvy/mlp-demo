# Graph plotting
import matplotlib.pyplot as plt

# Chainer modules
import chainer.functions as F
from chainer import FunctionSet, Variable, optimizers

# NumPy is used by Chainer
import numpy as np
from numpy import genfromtxt

# Helper module to read data from CSV
from dataparser import DataParser

def target_function(x):
    """
    The function that we want to approximate using a simple Multi-Layered Perceptron, MLP
    y = 2 * x + 8
    """
    x_double = np.multiply(2, x)
    return np.add(x_double, 8)

# Read the training and test data from CSV files
csv_parser = DataParser()
x_train, y_train = csv_parser.parse("data/linear_training.csv", delimiter=",")
x_test, y_test = csv_parser.parse("data/linear_test.csv", delimiter=",")

# Network parameters
n_units = 10

# Training parameters
n_epochs = 70
# batchsize = np.size(x_train)
batchsize = np.size(x_train)

# The size of the training data
datasize = np.size(x_train)

# Define the linear network model with 1 input unit and 1 output unit
model = FunctionSet(
    l1 = F.Linear(1, n_units),
    l2 = F.Linear(n_units, 1)
)

def forward(x_data, y_data, train=True):
    """
    Define the forward algorithm, a sigmoid activation function and a mean squared error (loss)
    """
    # Convert the NumPy data into Chainer Variables
    x = Variable(x_data)
    t = Variable(y_data)

    # Compute the output of the hidden layer with cuDNN (NVIDIA GPU library for DNNs) disabled
    h1 = F.sigmoid(model.l1(x), use_cudnn=False)
    # Replace the line above with the following line to use dropout
    #h1 = F.dropout(F.sigmoid(model.l1(x)), train=True)

    # Compute the output of the network
    y = model.l2(h1)

    # Return the loss so that it can be plotted later on
    return F.mean_squared_error(y, t)

# Setup the model with SGD
optimizer = optimizers.SGD()
# Alternative optimizer, Adam
#optimizer = optimizers.Adam()
optimizer.setup(model)

# Data used for later plotting
train_loss = []
test_loss = []

# Finally, start the training
for epoch in range(n_epochs):
    print "Epoch ", epoch

    sum_loss = 0

    # Randomize the batch selection
    indices = np.random.permutation(datasize)

    for i in range(0, datasize, batchsize):
        x_batch = x_train[indices[i : i + batchsize]]
        y_batch = y_train[indices[i : i + batchsize]]

        # Reset the gradients
        optimizer.zero_grads()

        # Compute the forward algorithm
        loss = forward(x_batch, y_batch, train=True)

        # Compute the backward propagation
        loss.backward()

        # Update the Chainer parameters, using the current gradients
        optimizer.update()

        # Register the loss so that it can be plotted later on
        sum_loss += float(loss.data) * batchsize

    # Compute the mean loss for this epoch
    epoch_mean_loss = sum_loss / datasize

    # Save the epoch mean loss so that it can be plotted later on
    train_loss.append(epoch_mean_loss)

    # Run the test data so that the delta loss can be plotted
    for i in range(0, datasize, batchsize):
        x_batch = x_test[indices[i : i + batchsize]]
        y_batch = y_test[indices[i : i + batchsize]]

        # Compute the forward algorithm
        loss = forward(x_batch, y_batch, train=False)

        # Register the loss so that it can be plotted later on
        sum_loss += float(loss.data) * batchsize

    # Compute the mean loss for this epoch
    epoch_mean_loss = sum_loss / datasize

    # Save the epoch mean loss so that it can be plotted later on
    test_loss.append(epoch_mean_loss)

plt.plot(train_loss)
plt.plot(test_loss)
plt.ylabel("Mean squared error")
plt.xlabel("Epochs")
plt.show()

