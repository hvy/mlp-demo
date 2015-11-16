import numpy as np

class DataParser:
    def __init__(self):
        """
        Empty constructor.
        """

    def parse(self, filename, delimiter=","):
        """
        Read the given file, e.g. a CSV file, and return a training data set. It assumes that the
        x data is given in column 0 and that the y data is given in column 1.
        """

        try:
            # Try to open the file
            genfile = np.genfromtxt(filename, delimiter=delimiter, dtype=np.float32)

            # Extract the data and shape it into a proper format
            x_train = genfile[:,0]
            y_train = genfile[:,1]
            x_train = x_train.reshape(len(x_train), 1)
            y_train = y_train.reshape(len(y_train), 1)

            return x_train, y_train

        except IOError as e:
            raise e

