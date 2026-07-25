import torch
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # input_size: número de features de entrada
        # hidden_size: número de neurônios na camada oculta
        # num_classes: número de neurônios na última camada (equivale ao número de classes)

        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)   # recebe o número de features / retorna o tamanho da camada oculta atual
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  # recebe o tamanho da camada oculta anterior / retorna o número de classes

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        y_pred = self.l2(x)