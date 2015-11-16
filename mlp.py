import numpy as np
from numpy import genfromtxt
import chainer
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import matplotlib.pyplot as plt
from dataparser import DataParser

def target_function(x):
    """
    The function that we want to approximate using a simple Multi-Layered Perceptron, MLP
    y = 2 * x + 8
    """
    x_double = np.multiply(2, x)
    return np.add(x_double, 8)

# Read the training data from a CSV
csv_parser = DataParser()
x_train, y_train = csv_parser.parse("data/linear.csv", delimiter=",")

# Generate the test data from the target function
x_test = np.arange(-4, 5.1, 0.1, dtype=np.float32)
y_test = target_function(x_train)
x_test = x_train.reshape(len(x_train), 1)
y_test = y_train.reshape(len(y_train), 1)

# Network parameters
n_units = 10

# Training parameters
n_epochs = 100
batchsize = np.size(x_train)

# The size of the training data
datasize = np.size(x_train)

# Define the linear network model with 1 input unit and 1 output unit
model = chainer.FunctionSet(
    l1 = F.Linear(1, n_units),
    l2 = F.Linear(n_units, 1)
)

def forward(x_data, y_data):
    """
    Define the forward algorithm, a sigmoid activation function and a mean squared error (loss)
    """
    # Convert the NumPy data into Chainer Variables
    x = chainer.Variable(x_data)
    t = chainer.Variable(y_data)

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
train_acc = []
test_loss = []
test_acc = []

l1_W = []
l2_W = []

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
        loss = forward(x_batch, y_batch)

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

    print "Epoch mean loss = {}".format(epoch_mean_loss)

plt.plot(train_loss)
plt.ylabel("Mean squared error")
plt.xlabel("Epochs")
plt.show()

