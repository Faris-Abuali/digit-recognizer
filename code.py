import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('data/train.csv')

# print(data.head())

data = np.array(data)
m, n = data.shape
# print(data.shape)
np.random.shuffle(data) # shuffle before splitting into dev and training sets

# ------------------------------------------------------------
data_dev = data[0:1000].T
# Now after transposing the matrix, each column is an example (digit from 0 to 9)
# and we have 784 rows (28px*28px) gives us 784px must be stored for each example
# print(data_dev.shape)

Y_dev = data_dev[0] # The output (the actual digit) is in the first column of the training set
# print(Y_dev) #list of all the digits (outputs)
# print(len(Y_dev)) #1000
X_dev = data_dev[1:n] #we skipped the 0th row because it is the output (the labels)
X_dev = X_dev / 255.
# print(X_dev)
# ------------------------------------------------------------

data_train = data[1000:m].T  # from 1000 to 42000
Y_train = data_train[0] # The output (the actual digit) is in the first column of the training set
X_train = data_train[1:n] # we skipped the 0th row because it is the output (the labels)
X_train = X_train / 255.


_,m_train = X_train.shape
# print(X_train.shape) # (784, 41000)
# print(m_train) # 41000


"""Our NN will have a simple two-layer architecture:
- Input layer (A^0) will have 784 units corresponding to the 784 pixels in each 28x28 input image. 
- A hidden layer (A^1) will have 10 units with ReLU activation, 
- and finally our output layer (A^2) will have 10 units corresponding to the ten digit classes with softmax activation."""

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
# Remember: np.random.rand(10, 784) will return a list of 10 lists, each list of 748 columns
# and the values will be in range (0, 1). That's why we made -0.5 to make the range from (-0.5, 0.5)
# which is a common range for initial values of the weights and the biases (thresholds)

def ReLU(Z):
    return np.maximum(Z, 0)
# This is element-wise, which means it will take every element of the matrix Z and compare it with 0. If the element
# element > 0 then return the element. Else, return 0
# And tha's how the ReLU (Rectified Linear Activation function) works.

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
# Z is a matrix of 10xm.  Each column is an ecample.
# sum(np.exp(z)) returns the sum for each column (the sum of the exponent of each element in that column)
# then we divide each element in the column by that sum.

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0
# If the z > 0, then the derivative of the linear function is 1. Else if the z<=0 then the derivative of the constant is 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # Y.size: the number of examples we have (The size of the training set)
    # Y.max() will be 9, so Y.max() + 1 will return 10
    # So, one_hot_Y is a list of m lists, each list of 10 columns
    one_hot_Y[np.arange(Y.size), Y] = 1
    # np.arange(Y.size) returns a list from [0, 1, 2, ..., Y.size()]
    one_hot_Y = one_hot_Y.T
    return one_hot_Y #  returns a list of 10 rows, each column is an example


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y # This is the error
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    # Now for the hidden layer:
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
# -------------------------------------------------------------------
def get_predictions(A2):
    # remember: A2 is a matrix of 10xm
    # 10 rows because we have 10 classifications 0,1,2,3,4,5,6,7,8,and 9
    # m is the number of examples in the training or testing set. (e.g. m = 41000 images)
    return np.argmax(A2, 0) # here we decide which digit this example/image represents based on the higher probability which was calculated using the softmax activation function. 

def get_accuracy(predictions, Y):
    # Y is the list of the desired (expected) outputs: e.g. [9 2 1 ... 5 8 1]
    print(f"prediction: {predictions}, Desired Output: {Y}")
    # predictions is the list that our model has outputted: e.g. [9 2 2 ... 6 9 1]
    return np.sum(predictions == Y) / Y.size
    # np.sum(predictions == Y) returns the number of true predictions


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Accuracy: {round(accuracy, 4)}") # prints the accuracy on the training set after every 10 iterations
            print("-----------------------------------------")
    return W1, b1, W2, b2
# -------------------------------------------------------------------


# ---------------------- To visualize what is happening: ----------------------
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# We will use matplotlib to convert the data to image:
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    # WTF is X_train[:, index, None]?????
    # This means take only a column whose index = index, so the 
    # taken column will be an example 
    # (a sample of input that represents a digit written in hand)
    # It will be a vector of length 784 of numbers from 0 to 255
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# -------------------------------------------------------------------
# Now let's run the code:
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500) 

# Let's look at a couple of examples:
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
test_prediction(4, W1, b1, W2, b2)
test_prediction(5, W1, b1, W2, b2)
test_prediction(6, W1, b1, W2, b2)
test_prediction(7, W1, b1, W2, b2)
test_prediction(8, W1, b1, W2, b2)
test_prediction(101, W1, b1, W2, b2)
test_prediction(106, W1, b1, W2, b2)


# Finally, let's find the accuracy on the dev set:
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Accuracy on the testing set: {accuracy}")
