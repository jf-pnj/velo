import torch as th
import torch.nn.functional as F
import torch.nn as nn

class NeuralNetwork(nn.Module):
    '''
    This class initialises a Neural Network based on the input arguments. After
    training, new data can be fed into the network to provide predictions.
    '''
    def __init__(self, input_dim, layer_dims, dropout_value):
        '''
        This constructor creates the Neural Network based on the specifications
        of the input arguments.
        
        input_dim: the number of features of the data
        layer_dims: a list of integers with the number of neurons per
        consecutive layer_dims
        dropout_value: the value of dropout to be used.
        '''
        super(NeuralNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, layer_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_value))
        layers.append(nn.Linear(layer_dims[0], layer_dims[1]))
        layers.append(nn.ReLU())
        if len(layer_dims) - 2 > 0:
            for i in range(len(layer_dims)-2):
                layers.append(nn.Dropout(dropout_value))
                layers.append(nn.Linear(layer_dims[i+1], layer_dims[i+2]))
                layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_value))
        layers.append(nn.Linear(layer_dims[-1], 1))
        self.seq = nn.Sequential(*layers)

    def forward(self, inputs):
        '''
        This function feeds the data through the created Neural Network
        '''
        return self.seq(inputs)