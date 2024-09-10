import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS


class SpectogramDataset(Dataset):
    def __init__(self, root, url, download=False):
        self.dataset = SPEECHCOMMANDS(root=root, url=url, download=download)

    def __getitem__(self, n):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[n]

        # compute mel spectrogram
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate,  # sample_rate of original waveform
            n_fft=64,  # number of frequency bins
            win_length=400,
            hop_length=160,
            n_mels=64,
            window_fn=torch.hamming_window,
            power=2.0,
            normalized=True,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=True,
            norm="slaney",
            mel_scale="htk",
        )(waveform)

        return spectrogram, label

    def __len__(self):
        return len(self.dataset)
