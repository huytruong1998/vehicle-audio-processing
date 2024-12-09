import torch
from torch import nn, optim

def loss_and_optimizer(model:nn.Module, learning_rate:float=0.001):
    """
    :param model:nn.Module
    :param learning_rate:float = 0.001
    :return: loss_function:nn.Module, optimizer:nn.Module

    Return Cross Entropy Loss function and Adam optimizer
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return [loss_function, optimizer]
