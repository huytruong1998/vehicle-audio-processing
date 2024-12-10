import librosa
from torch.utils.data import Dataset
import torch
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import torchaudio


class SoundDataSet(Dataset):
    def __init__(self, file_paths, labels, device, sr=44100,n_fft=1024,hop_length=512, n_mfcc=13,n_mels=128, duration=5):
        self.file_paths = file_paths
        self.labels = labels
        
        self.sampled_sr = sr
        self.n_fft = n_fft
        self.duration = duration
        self.hop_length = hop_length
        self.n_mfcc= n_mfcc
        self.device = device
        self.n_mels = n_mels

        self.melspectrogram_dbs= []
        for index in range(len(file_paths)):
            mel_spec_db = self.__get_melspectrogram_db__(index)
            mel_spec_db = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
            self.melspectrogram_dbs.append(mel_spec_db)

         
    def __len__(self):
        return len(self.file_paths)

    def __get_item__(self, index):
        return self.melspectrogram_dbs[index], self.labels[index]
    
    def __load_audio__(self, index):
        audio_sample_path = self.file_paths[index]
        signal, sr = librosa.load(audio_sample_path, sr=None)
        y, x = torchaudio.load(audio_sample_path)
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
        ms = librosa.feature.melspectrogram(y=signal, sr=sr,fmax=sr// 2,n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(ms, ref=np.max)
        return self.__melspec_normalization__(mel_spec_db)

    def __melspec_normalization__(self,mel_spec_db):
        # Normalize to [0, 1]
        return (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

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

    def __play_audio__(self, index):
        signal, sr = self.__load_audio__(index)
        plt.figure()
        plt.plot(signal)
        plt.show()
        sd.play(signal)
        sd.wait()
        


