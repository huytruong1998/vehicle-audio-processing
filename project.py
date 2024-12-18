import numpy as np
import librosa.display, os
import time
from torch import nn, optim
import torch
from torchsummary import summary
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms,datasets


SPECTOGRAM_SHOW = False
MFCC_SHOW = False
MEL_SPECTOGRAM = True
CREATE_IMAGES = True
SAVE_MODEL = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

##Power Spectogram Functions
def create_power_spectrogram(audio_file, image_file):
    """
    Creates Power spectogram based on audio file and 
    Saves it to image file
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
def create_power_spec_pngs_from_wavs(input_path, output_path):
    """
    Creates Power Spectogram Images folder based on input path
    and Saves them in output file
    """
    #Checks if the folder exits
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_power_spectrogram(input_file, output_file)

##Mel Spectogram Functions
def resample_audio(signal, sr, sampled_sr= 44100):
    if sr != sampled_sr:
        signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=sampled_sr)
        return signal_resampled, sampled_sr
    
    return signal, sr


def process_audio_duration(signal, sr, target_duration=5.0):
    """
    Adjust the Audio Data duration. If it is longer, reduces it. 
    If it is shorter, pads it with zeros.
    """
    # Make sure all audio are 5s
    target_samples = int(target_duration * sr)
    current_samples = len(signal)
    
    if current_samples > target_samples:
        signal = signal[:target_samples]
    
    # If the audio is shorter, pad with zeros (silence) to the target duration
    elif current_samples < target_samples:
        pad_length = target_samples - current_samples
        signal = np.pad(signal, (0, pad_length), mode='constant')
    
    return signal, sr


def melspec_normalization(log_ms):
    """
    Mel Spectogram Normalization between [0, 1]
    """
    return (log_ms - log_ms.min()) / (log_ms.max() - log_ms.min())
    

def load_audio(audio_file):
    """
    Transforms Audio file to Data and SR, and Returns both of them
    """
    data, sr = librosa.load(audio_file)
    data, sr = resample_audio(data, sr)
    # TO DO: should we add this?  reduce accuracy but looks better to compare spectrogram
    data, sr = process_audio_duration(data,sr)
    data = librosa.util.normalize(data)
    return data, sr
    

def create_mel_spectrogram(audio_file):
    """
    Creates Mel Spectogram from Audio File and Returns it
    """
    y, sr = load_audio(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    log_ms = melspec_normalization(log_ms)
    return log_ms, sr


def create_mel_spectrogram_image(audio_file, image_file):
    """
    Creates Mel Spectogram and Saves it
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    log_ms, sr = create_mel_spectrogram(audio_file)

    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)

    
def create_mel_spect_pngs_from_wavs(input_path:str, output_path:str):
    """
    Middle Function for Mel Spectpogram creation from input path to output path
    """

    #checking if the folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    #creating Spectgorams from input folder to output folder
    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_mel_spectrogram_image(input_file, output_file)

    print(f'Spectograms Created based on file {input_path}')


##SHOW Functions
def create_spectrogram(audio_file):
    """
    Creates Power Spectogram and returns it with SR
    """
    y, sr = load_audio(audio_file)
    n_fft = 1024
    win_length = n_fft
    hop_length = n_fft//2
    librosa_spectrogram = librosa.stft(y=y,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    librosa_power_spectrogram = librosa.amplitude_to_db(librosa_spectrogram, ref=np.max)
    return librosa_power_spectrogram, sr


def create_mfcc(audio_file):
    """
    Creates MFCC and returns it
    """
    y, sr = load_audio(audio_file)
    n_fft = 1024
    hop_length =n_fft//2 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return mfccs


def load_images_from_path(path):
    #transforms.ToTensor() automatically converts and normalizes pixel values to [0.0, 1.0]
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    full_dataset = datasets.ImageFolder(
        root=path,
        transform=transformation
    )

    return full_dataset


def show_images(images):
    """
    Shows some of the Spectogram Images
    """
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)


# Plot some spectrogram and compare
if SPECTOGRAM_SHOW:
    librosa_power_spectrogram1,sr1 = create_spectrogram('./dataset/bus/657981__petrijuhani__bussi2.wav')
    librosa_power_spectrogram2,sr2 = create_spectrogram('./dataset/tram/709545__masa_ite__tram_hervannan_kampus_b_2.wav')

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa_power_spectrogram1, sr=sr1, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Bus')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa_power_spectrogram2, sr=sr2, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Tram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


#Plot some mfccs and compare
if MFCC_SHOW:
    mfccs1 = create_mfcc('./dataset/bus/657981__petrijuhani__bussi2.wav')
    mfccs2 = create_mfcc('./dataset/tram/709545__masa_ite__tram_hervannan_kampus_b_2.wav')

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(mfccs1, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f')
    plt.title("MFCCs Bus")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")

    plt.subplot(1, 2, 2)
    plt.imshow(mfccs2, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f')
    plt.title("MFCCs Bus")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")

    plt.tight_layout()
    plt.show()


#Creating Spectograms
if CREATE_IMAGES:
    if MEL_SPECTOGRAM:
        create_mel_spect_pngs_from_wavs('./dataset/bus', './Spectogram_images/bus')
        create_mel_spect_pngs_from_wavs('./dataset/tram', './Spectogram_images/tram')
    else:
        create_power_spec_pngs_from_wavs('./dataset/bus', './Spectogram_images/bus')
        create_power_spec_pngs_from_wavs('./dataset/tram', './Spectogram_images/tram')

full_dataset = load_images_from_path('./Spectogram_images')
##Spliting the data to use 70% training and 30% testing
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
# define a loader for the testing data we can iterate through in 32-image batches
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)



print(f'Training set size: {len(train_loader.dataset)}')
print(f'Testing set size: {len(test_loader.dataset)}')


#Creating the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), #(224+2×1−3)/1 +1 = (32, 224,224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces dimensions by half (32, 112,112)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (112+2×1−3)/1 +1 = (64, 112,112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces dimensions by half (64, 56,56)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # (56+2×1−3)/1 +1 = (128, 56,56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces dimensions by half (128, 28,28)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # (28+2×1−3)/1 +1 = (128, 28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces dimensions by half (128, 14,14)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128*14*14, out_features=1024)
        self.dropout = nn.Dropout(p=0.2)
        # dropout: Reduces overfitting, Improves generalization.
        self.fc2 = nn.Linear(in_features=1024, out_features=1)
        self.output = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)  # Flatten feature maps into a 1D vector
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x) 
        x = self.output(x)  # Apply Sigmoid for binary classification
        
        return x
model=CNNModel().to(device)
summary(model,(3,224,224))

#Create Training single epoch function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total_samples = 0
    print("Epoch:", epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        # Converted to float32 with shape [batch_size, 1]
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

        predicted = (output >= 0.5).float()
        correct += torch.sum(target == predicted).item()
        total_samples += target.size(0)
        
        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
    # return accuracy for training
    accuracy = 100. * correct / total_samples
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss,accuracy

#Create Testing single epoch function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            
            output = model(data)
            
            test_loss += loss_criteria(output, target).item()
            
            predicted = (output >= 0.5).float()
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        accuracy))
    
    # return average loss for the epoch
    return avg_loss,accuracy

# Use an "Adam" optimizer to adjust weights
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Specify the loss criteria (use BCE here for Binary Classification)
loss_criteria = nn.BCELoss()

# Track metrics in these arrays
epoch_nums = []
epoch_loss = []
epoch_validation_loss = []

epoch_training_accuracy = []
epoch_test_accuracy = []

epochs = 10
print('Training on', device)
for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)
        epoch_nums.append(epoch)
        epoch_loss.append(train_loss)
        epoch_validation_loss.append(test_loss)
        epoch_training_accuracy.append(train_accuracy)
        epoch_test_accuracy.append(test_accuracy)

# Plot Loss
plt.figure()
plt.plot(epoch_nums, epoch_loss)
plt.plot(epoch_nums, epoch_validation_loss)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation Loss'], loc='upper right')
plt.show()


# Plot Accuracy
plt.figure()
plt.plot(epoch_nums, epoch_training_accuracy)
plt.plot(epoch_nums, epoch_test_accuracy)
plt.title('Training and Validation Acuuracy')
plt.xlabel('Epoch')
plt.ylabel('Acuuracy')
plt.legend(['Training Acuuracy', 'Validation Acuuracy'], loc='upper right')
plt.show()