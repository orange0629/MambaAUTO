import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, dropout:float = 0.1):
        '''
        MLP is used to serve as patch embedder or linear layers.
        '''
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
