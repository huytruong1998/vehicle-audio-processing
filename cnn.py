import torch
from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    def __init__(self, time_frames, n_mels=128):  # Accepts time_frames and n_mels
        super().__init__()
        self.n_mels = n_mels
        self.time_frames = time_frames

        # Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces dimensions by half
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Calculate the dimensions of the feature map after convolutional layers
        final_height = self._calc_final_dim(self.n_mels, num_pooling_layers=4)
        final_width = self._calc_final_dim(self.time_frames, num_pooling_layers=4)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=128 * final_height * final_width, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5)  # Regularization
        )
        self.linear2 = nn.Linear(in_features=128, out_features=1)

        # Output Layer
        self.output = nn.Sigmoid()

    def _calc_final_dim(self, input_dim, num_pooling_layers):
        """
        Calculate the final dimension after convolution and pooling layers.
        Each pooling layer halves the input dimension.
        """
        for _ in range(num_pooling_layers):
            input_dim = input_dim // 2
        return input_dim

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)  # Flatten feature maps into a 1D vector
        x = self.linear1(x)  # Dense hidden layer
        logits = self.linear2(x)  # Final linear layer
        output = self.output(logits)  # Apply sigmoid for binary classification
        return output

model=CNNNetwork().cuda()
summary(model,(1,128,430))