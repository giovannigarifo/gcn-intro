# -------------------------------
# | Toy example of a simple GCN |
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 

# ----------------------------------
# | Graph structure representation |
# ----------------------------------
# Graphs are fully represented by their "adjancency matrix A":
#   A(i,j) contains the edge weight for the directed edge that goes from node "i" to node "j", or zero if they're
#   not connected, at least in the direction "from i to "j". 
#   A consequence of this is that for a non-directed graph, A is symetric.
# 
# From the adjacency matrix, it's possible to obtain the "in-degree matrix D":
#   D(i,i) is equal to the number of inward edges of the node "i". This matrix is diagonal.
#
# Because GCNs performs the convolution between the features of the node and the features of the node neighbors by
# performing matrix multiplications, we need to add self-loops to the adjacency matrix, by simply adding the 
# identity matrix I to A.
#
# From A it's possible to obtain the "transition matrix P":
#   P(i,j) is equal to the probability of a random walk from i to j. To obtain P, just perform the matrix product between the
#   inverse of D and A.

print("\nAdjacency Matrix A (with self loops added):")
A = np.matrix(
    [[0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float)
A = A + np.identity(A.shape[0], dtype=float) # add self loops
print(A)

print("\n In-degree matrix D:")
D = np.diag(np.array(np.sum(A, axis=0))[0])
print(D) 

print("\nTransition matrix P:")
P = np.linalg.inv(D) * A    # use linalg.inv() because D**-1 raise a division by zero error
print(P)

# plot the graph
Graph = nx.from_numpy_matrix(np.array(P), create_using=nx.MultiDiGraph())
nx.draw(Graph, with_labels=True)
plt.show()


# --------------------------------
# | Node features representation |
# --------------------------------
# The features of each node are represented by the matrix X where the i-th row contains the features of 
# the i-th node of the graph, for this toy example the features are self generated. In a real word scenario the
# features will be a vector representation of each node characteristics.

print("\nFeature Matrix X:")
X = np.matrix([
        [i, -i] for i in range(A.shape[0])
    ], 
    dtype=float)
print(X)


# -------------------
# | Network weights |
# -------------------
# The weight matrix has a number of rows equal to the the number of features of the "input feature matrix X", and a
# number of columns equal to the number of features that the "output feature matrix H" will have.
# In this case, we use two values for each weight, so H will be shaped as a "Nx2" matrix, where N is the number of nodes.
# we'll initialize the weights with random values for this example. 

print("\nWeight matrix W:")
W = np.matrix([
            [1, -1], #first weight
            [-1, 1]  #second weight
        ])
print(W)


# -----------------------
# | Activation function |
# -----------------------
# no better activation function then a ReLU for a toy example!

def ReLU (m):
  return np.maximum(m, 0)


# ----------------------------
# | Forward pass 1-layer GCN |
# ----------------------------
# now we're ready to write down our GCN composed of one hidden layer:
#
#               |   Convolutional Layer   |
# Input Layer   |  Pooling     Filtering  |   Output Layer
#     X        ->   A * X  ->  (A*X) * W   ->      H

print("\n---\nPerforming forward pass of GCN:\n---\n")

conv = P*X
print("\nResult of P*X:")
print(conv)
print("As can be seen, the result is equal to the sum of the features of each node with the", 
      "features of its neighbors! This operation is similar to the pooling in a standard CNN.")

conv_w = conv*W
print("\nResult of weight multiplication, P*X*W:")
print(conv_w)
print("The multiplication with the weights is similar to the filtering operation in a standard CNN.")

print("\nApplying activation function, output feature matrix H:")
H = ReLU(conv_w)
print(H)
