import librosa
import os
from torch.utils.data import Dataset
import torch
import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

def getFilePaths(path):
    return [path +'/'+ file for file in os.listdir(path)]



class SoundDataSet(Dataset):
    def __init__(self, file_paths, labels,categories, sr,n_fft,hop_length, n_mfcc, device, duration=5):
        self.file_paths = file_paths
        self.labels = labels
        

        self.sampled_sr = sr
        self.n_fft = n_fft
        self.duration = duration
        self.hop_length = hop_length
        self.n_mfcc= n_mfcc
        self.device = device

        self.c2i={}
        self.i2c={}
        self.categories = categories
        self.images= []
        for i, category in enumerate(self.categories):
            self.c2i[category]=i
            self.i2c[i]=category
        for index in range(len(file_paths)):
            log_ms = self.__get_melspectrogram_db__(index)
            image = self.__melspec_to_image__(log_ms)
            self.images.append(image)


         
    def __len__(self):
        return len(self.file_paths)

    def __get_item__(self, index):
        return self.images[index], self.labels[index]
    
    def __load_audio__(self, index):
        audio_sample_path = self.file_paths[index]
        signal, sr = librosa.load(audio_sample_path, sr=None)
        # Get the first 5s (duration) of the audio
        signal = self.__get_audio_duration__(signal, sr)
        signal, sr = self.__resample_audio__(signal, sr)
        return signal, sr

    def __get_audio_duration__(self, signal,sr):
        if signal.shape[0]<self.duration*sr:
            signal=np.pad(signal,int(np.ceil((self.duration*sr-signal.shape[0])/2)),mode='reflect')
        else:
            signal=signal[:self.duration*sr]
        return signal

    def __resample_audio__(self, signal, sr):
        if sr != self.sampled_sr:
            signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=self.sampled_sr)
            return signal_resampled, self.sampled_sr
        return signal, sr

    def __get_melspectrogram_db__(self, index):
        signal, sr = self.__load_audio__(index)
        ms = librosa.feature.melspectrogram(y=signal, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        return log_ms
    
    def __melspec_to_image__(self,log_ms):
        melspec_norm = self.__melspec_normalization__(log_ms)
        melspec_min, melspec_max = melspec_norm.min(), melspec_norm.max()
        melspec_scaled = 255 * (melspec_norm - melspec_min) / (melspec_max - melspec_min)
        melspec_scaled = melspec_scaled.astype(np.uint8)
        return melspec_scaled
    
    def __melspec_normalization__(self,log_ms):
        mean = log_ms.mean()
        std = log_ms.std()
        melspec_norm = (log_ms - mean) / (std)
        return melspec_norm

    def __plot_mfcc__(self, index):
        signal, sr = self.__load_audio__(index)
        hop_size = self.hop_length//2
        mfccs_librosa = librosa.feature.mfcc(y=signal, sr=sr, n_fft=self.n_fft, hop_length=hop_size)
        plt.figure()
        plt.imshow(mfccs_librosa, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f')
        plt.title("MFCCs")
        plt.xlabel("Time Frames")
        plt.ylabel("MFCC Coefficients")
        plt.tight_layout()
        plt.show()
        

    def __plot_spectrogram__(self, index):
        signal, sr = self.__load_audio__(index)
        nfft = self.n_fft
        win_size = nfft
        hop_size = nfft//2
        librosa_spectrogram = librosa.stft(signal,n_fft=nfft, hop_length=hop_size, win_length=win_size)
        librosa_power_spectrogram = librosa.amplitude_to_db(librosa_spectrogram, ref=np.max)

        plt.figure()
        librosa.display.specshow(librosa_power_spectrogram, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

    


if __name__ == "__main__":
    audio_paths = []
    audio_labels = []
    categories = ['bus', 'tram']
    for i, category in enumerate(categories):
        paths = getFilePaths('dataset/' + category)
        audio_paths += paths
        audio_labels += [i]*len(paths)
   

    SAMPLE_RATE = 44100
    n_fft=2048
    hop_length = 512
    n_mfcc = 13

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    audio_data = SoundDataSet(file_paths=audio_paths, labels=audio_labels,categories=categories, sr=SAMPLE_RATE,n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc, device=device)
    audio_data.__plot_spectrogram__(40)
    audio_data.__plot_mfcc__(40)
