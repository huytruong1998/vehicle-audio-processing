import torch
from torch.utils.data import DataLoader
import os

from feature_extract import SoundDataSet
from cnn import CNNNetwork
import loss_optimizer 

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

def getFilePaths(path):
    return [path +'/'+ file for file in os.listdir(path)]

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    loss = None
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")
    return loss


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    losses= []

    for i in range(epochs):
        print(f"Epoch {i+1}")
        losses.append(train_single_epoch(model, data_loader, loss_fn, optimiser, device))
        print("---------------------------")
    print("Finished training")

    return losses

if __name__ == "__main__":
    audio_paths = []
    audio_labels = []
    categories = {
        0: "bus",
        1: "tram"
    }
    for key, value in categories.items():
        paths = getFilePaths('dataset/' + value)
        audio_paths += paths
        audio_labels += [key]*len(paths)
   

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    audio_data = SoundDataSet(file_paths=audio_paths, labels=audio_labels, device=device)
    #audio_data.__plot_spectrogram__(26)
    #audio_data.__plot_mfcc__(26)
    #audio_data.__play_audio__(26)

    model = CNNNetwork()
    loss_func, optimizer = loss_optimizer.torch_loss_and_optimizer(model=model, lr= LEARNING_RATE)
    train_data_loader = create_data_loader(audio_data, BATCH_SIZE)

    loss = train(model, train_data_loader,loss_func, optimizer, device, EPOCHS)

    if False:
        torch.save(model.state_dict(), "vehicle_audio_processing_model.pth")
        print("Trained feed forward net saved at vehicle_audio_processing_model.pth")

