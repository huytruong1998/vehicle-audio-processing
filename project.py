import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


SPECTOGRAM_SHOW = False
MFCC_SHOW = False
MEL_SPECTOGRAM = True
CREATE_IMAGES = True
SAVE_MODEL = True


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
        signal = np.pad(y, (0, pad_length), mode='constant')
    
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
    data, sr = resample_audio(y, sr)
    # TO DO: should we add this?  reduce accuracy but looks better to compare spectrogram
    #y, sr = process_audio_duration(y,sr)
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


def load_images_from_path(path, label):
    """
    Converts Images to NumPy arrays, creates Labels based on their positon and Returns
    """
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))
        
    return images, labels


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


#Creating images and labels
x = []
y = []

images, labels = load_images_from_path('./Spectogram_images/bus', 0)
x += images
y += labels
show_images(images)

images, labels = load_images_from_path('./Spectogram_images/tram', 1)
x += images
y += labels
show_images(images)


#Spliting the data to Train and Test data
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded =  tf.keras.utils.to_categorical(y_train)
y_test_encoded =  tf.keras.utils.to_categorical(y_test)


#Creating the model
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


#Training the model
start_time = time.time()
hist = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), batch_size=10, epochs=10)
end_time = time.time()

total_time = end_time - start_time
print(f"Total Training Time: {total_time:.2f} seconds")


#Training and Validation history
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)
loss = hist.history['loss']

plt.plot(epochs, loss, '--', label='Loss')
plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
plt.plot()


#Saving the model
if SAVE_MODEL:
    model_name = 'vehicle-audio-processing-model'
    model_path = model_name + '.keras'
    model.save(model_path)
    print(f'Model is Saved to {model_path}')


#Creating Test Spectogram images
if CREATE_IMAGES:
    if MEL_SPECTOGRAM:
        create_mel_spect_pngs_from_wavs('./test_audio', './Spectogram_images/test')
    else:
        create_power_spec_pngs_from_wavs('./test_audio', './Spectogram_images/test')
        
#You can change the name of the test audio with another test data name
test_audio_name = '664055__juusooo__bus-2'
test_audio_path = './Spectogram_images/test/' + test_audio_name + '.png'
x = tf.keras.utils.load_img(test_audio_path, target_size=(224, 224))

plt.xticks([])
plt.yticks([])
plt.imshow(x)

#Testing 
x = tf.keras.utils.img_to_array(x)
x = np.expand_dims(x, axis=0)

predictions = model.predict(x)
class_labels = ['bus', 'tram']

for i, label in enumerate(class_labels):
    print(f'{label}: {predictions[0][i]}')