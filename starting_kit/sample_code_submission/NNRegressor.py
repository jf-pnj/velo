import torch as th
import torch.nn.functional as F
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout_value):
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
        return self.seq(inputs)