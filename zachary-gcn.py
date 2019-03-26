# --------------------------------------
# | GCN over Zachary Karate Club graph |
# --------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 

zkc = nx.karate_club_graph() # graph representation
nx.draw(zkc, with_labels=True)
plt.show()

order = sorted(list(zkc.nodes()))
A = nx.to_numpy_matrix(zkc, nodelist=order)

A = A + np.eye(zkc.number_of_nodes()) # add self loops
D = np.matrix(np.diag(np.array(np.sum(A, axis=0))[0])) # in-degree matrix

# now get the transition matrix P
P = np.linalg.inv(D) * A

# weights taken randomly from normal distribution
W_1 = np.random.normal(
    loc=0, # mean of the distribution
    scale=1, # standard deviation
    size=(zkc.number_of_nodes(), 4) # shape of output matrix
    )
W_2 = np.random.normal(
    loc=0, 
    scale=1,
    size=(W_1.shape[1], 2) 
    )

# activation function
def ReLU (m):
  return np.maximum(m, 0)

# single hidden layer of the GCN
def hidden_layer(P, H_in, W):
    H_out = P*H_in*W # convolution
    return ReLU(H_out) #activation

# -------------------------------------------------
# | apply GCN forward pass with two hidden layers |
# -------------------------------------------------
#
# At each convolutional layer "l", the network convolves the features of each node with the neighbours features
# of the nodes that are one step further (l-th level neighbours).

# input features, for this example each node is represented as a one categorical variable, node i-th
# as a 1 as i-th feature.
input_features = np.eye(zkc.number_of_nodes()) 

# - first convolutional layer -  
# convolving each node features with it's immediate neighbor features,
# neighbours of the i-th node are the one reachable from i-th node with a directed outgoing edge.
H_1 = hidden_layer(P, input_features, W_1)

# - second convolutional layer - 
# convolving each node features (that now are equal to the sum of the node features
# and the features of each node immediate neighbors) with it's immediate neighbor features, that are also now
# represented by the sum of the neighbour neighbours features.
#
# This means that now each node has as features the sum of his features, his first-step neighbours features, and 
# it's second step neighbours features (the immediate neighbours of his neighours). 
H_2 = hidden_layer(P, H_1, W_2)

output_features = H_2

# dictionary of features of each node, <node>:<array-of-features>
feature_representations = {
    node:np.array(output_features)[node] for node in zkc.nodes() # using dictionary comprehension
    }

for node in feature_representations:
    print(node, ":", feature_representations[node])

# plot the features representation of each node
x_axis = np.array(output_features[:,0])
y_axis = np.array(output_features[:,1])

plt.scatter(x_axis, y_axis)
plt.show()