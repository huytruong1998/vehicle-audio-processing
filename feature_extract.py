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
    def __init__(self, file_paths, labels, sr,n_fft, duration=5):
         self.file_paths = file_paths
         self.labels = labels
         self.sampled_sr = sr
         self.n_fft = n_fft
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, index):
        audio_sample_path = self.file_paths[index]
        label = self.labels[index]
        signal, sr = librosa.load(audio_sample_path, sr=None)
        signal_resampled, sr_resampled = self._resample_audio_(signal, sr)
        return signal_resampled, sr_resampled, label
        # sd.play(signal_resampled, sr_resampled)
        # sd.wait()

    def __normalized_mfcc_item__(self, index):
        signal, sr, label = self.__getitem__(index)
        win_size= self.n_fft
        hop_size = win_size//2
        mfccs_librosa = librosa.feature.mfcc(y=signal, sr=sr, n_fft=self.n_fft, hop_length=hop_size)
        mfcc_mean = np.mean(mfccs_librosa)
        return mfccs_librosa, mfcc_mean

    def __plot_mfcc__(self, index):
        mfccs, _ = self.__normalized_mfcc_item__(index)
        plt.figure()
        plt.imshow(mfccs, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f')
        plt.title("MFCCs")
        plt.xlabel("Time Frames")
        plt.ylabel("MFCC Coefficients")
        plt.tight_layout()
        plt.show()
        

    def __plot_spectrogram__(self, index):
        signal, sr, label = self.__getitem__(index)
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

    def _get_audio_label(self,index):
        return self.labels[index]

    def _resample_audio_(self, signal, sr):
        if sr != self.sampled_sr:
            signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=self.sampled_sr)
            return signal_resampled, self.sampled_sr
        return signal, sr


if __name__ == "__main__":
    audio_bus = getFilePaths('dataset/bus')
    bus_labels = np.zeros(len(audio_bus))

    audio_tram = getFilePaths('dataset/tram')
    tram_labels = np.ones(len(audio_tram))

    all_audio = audio_bus + audio_tram
    all_labels = np.concatenate((bus_labels, tram_labels))

    SAMPLE_RATE = 44100

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    audio_data = SoundDataSet(file_paths=all_audio, labels=all_labels, sr=SAMPLE_RATE,n_fft=512)
    audio_data.__plot_spectrogram__(10)
