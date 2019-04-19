# ---------------------------------------------------------------
# | GCN over Zachary Karate Club graph using Deep Graph Library |
# ---------------------------------------------------------------
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# | Define the Graph |
# --------------------
#
# In DGL the graph class stores nodes, edges and features associated with nodes. They're always directed graphs.

graph = dgl.DGLGraph() # create an empty graph

# add nodes, for ZKC we have 34 nodes
num_nodes = 34;
graph.add_nodes(num_nodes);

# add edges between nodes, graph is undirected so we had the same edge in both directions.
edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)]

# convert the list of tuples to two lists of source nodes and dest nodes
src, dst = list(zip(*edge_list))
graph.add_edges(src, dst)
graph.add_edges(dst, src)
print('graph has %d nodes and %d edges.\n' % (graph.number_of_nodes(), graph.number_of_edges()))

# Add features to the graph, as one hot encoded vector, ndata is an abbreviation for DGLGraph.nodes[:].data
graph.ndata['features'] = torch.eye(num_nodes)
print("Features tensor:\n", graph.nodes[:].data)

# ------------------
# | Define the GCN |
# ------------------
#
# In DGL the pooling and convolution operations are interpreted as:
#   pooling     -> message + reduce functions, actually performs A*X
#   filtering   -> linear transformation of A*X with the W tensor, performing (A*X)*W

def gcn_message(edges):
        '''
        Computes a batch of messages called 'msg' using the src node's feature "h".
        
        edges.src['h'] is a <number-of-edges>*<number-of-node-feature> tensor. A row
        contains the features of the src node of the corresponding edge.
        
        Parameters
        ----------
                edges : batch of edges of the graph

        Return
        ------
                a dictionary with key "msg" and as value a "tensor" that contains the
                **features of all the source nodes** of the edges contained in the batch.
        '''
        return {'msg' : edges.src['h']}


def gcn_reduce(nodes):
        '''
        Computes the new features 'h' for each node, by summing the features received as messages in the node's
        mailbox. The msg contains the features of the neighbour nodes, they're summed and returned as
        the new features of each node.

        The function may be called multiple times for multiple batch of nodes.

        Parameters
        ----------
                nodes : batch of nodes of the graph

        Return
        ------
                a dictionary that has as key "h" and as value a tensor with the new node(s) features,
                the number of rows of the returned tensor depends on the number of nodes received
                as argument (may be a batch or a single node)
        '''
        return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}


class GCNLayer(nn.Module):
        def __init__(self, in_features, out_features):
                super(GCNLayer, self).__init__()
                self.linear = nn.Linear(in_features, out_features)

        def forward(self, graph, inputs):
                '''
                Perform a forward pass through the a GCN layer

                Parameters
                        graph : the DGL graph
                        inputs : the features of the graph's nodes
                '''

                graph.ndata['h'] = inputs # set the node's features

                # step 1 : perform pooling among neighbourhood of each node
                graph.send(graph.edges(), gcn_message) # compute messages, they're added to node's mailboxes
                graph.recv(graph.nodes(), gcn_reduce) # reduction, sum all messages in the node's mailboxes
                h = graph.ndata.pop('h'); # we obtain the result of A*X

                # step 2: linear transformation with weights, H*W
                return self.linear(h) 


class simple_GCN(nn.Module):
        '''
        GCN architecture
        '''
        def __init__(self, in_features, hidden_size, num_classes):
                super(simple_GCN, self).__init__()
                self.layer1 = GCNLayer(in_features, hidden_size)
                self.layer2 = GCNLayer(hidden_size, num_classes)

        def forward(self, graph, inputs):
                h = self.layer1(graph, inputs)
                h = torch.relu(h)
                h = self.layer2(graph, h)
                return h

# ----------------
# | Tran the GCN |
# ----------------
net = simple_GCN(num_nodes, 5, 2) # num_features, num_hidden_features, num_classes (= output features used for classification)

# data initialization
inputs = graph.ndata['features']
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_raw_outputs = []

for epoch in range(50):
        raw_output = net(graph, inputs)

        all_raw_outputs.append(raw_output.detach())

        # compute probabilities of output classes for each node
        logp = F.log_softmax(raw_output, 1) # apply log(softmax) to raw output of NN, more performance compared to softmax

        # compute loss
        loss = F.nll_loss(logp[labeled_nodes], labels) # compute loss (negative log likelihood) only for labeled nodes

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))