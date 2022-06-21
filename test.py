import numpy as np

# Y = [9, 2, 4, 4]
# predictions = [9, 1, 4, 4]

# Y = np.array(Y)
# predictions = np.array(predictions)

# print(np.sum(Y == predictions))


# l = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9],
# ])

# print(l)
# print("------------")
# print(l[:, 1, None])

# # X_train[:, index, None]



def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


W1 = np.random.rand(10, 784) - 0.5  # 10x784
X0 = np.random.rand(784, 3)         # 784x3
Z1 = W1.dot(X0)                     # 10x3
print(Z1)

b1 = np.random.rand(10, 1) - 0.5    # 10x1
print("b1")
print(b1)

F = Z1 + b1
print(F)