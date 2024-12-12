import torch
from torch import nn, optim

def torch_loss_and_optimizer(model:nn.Module, learning_rate:float=0.001):
    """
    :param model:nn.Module
    :param learning_rate:float = 0.001
    :return: loss_function:nn.Module, optimizer:nn.Module

    Return Cross Entropy Loss function and Adam optimizer
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return [loss_function, optimizer]

def get_total_false(label):
    false_count = 0

    for index in range(len(label)):

        if label[index] == 0:
            false_count += 1
    return false_count

def get_total_true(label):
    true_count = 0

    for index in range(len(label)):

        if label[index] == 0:
            true_count += 1
    return true_count

def custom_loss(data_label, pred_label):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for index in range(len(data_label)):
        if pred_label[index] == data_label[index]:
            true_positive += 1
        elif pred_label[index] == 1 & data_label[index] == 0:
            false_positive += 1
        elif pred_label[index] == 0 & data_label[index] == 1:
            false_negative += 1
        else:
            true_negative += 1
        
    precision = true_positive / (true_positive + true_negative)
    recall =true_positive / (true_positive + false_positive)
    
    return precision, recall
            