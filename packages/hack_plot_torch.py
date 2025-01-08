''' This module defines the SpectrogramConverter class for converting iq array
to spectrogram in CUDA Tensor.'''

import numpy as np
import torch
import math
from scipy.signal import get_window


class SpectrogramConverter:
    '''Convert IQ data to spectrogram'''
    TFA_WINDOW_SIZE = 120
    TFA_OVERLAP = 0.3
    SIGMA = 5

    def __init__(self, name='SpectrogramConverter', device = 'cpu'):
        self.__bandwidth = None
        self.__sample_rate = None
        self.__time_duration = None
        self.__window = None
        self.__start_spectro_f = None
        self.__stop_spectro_f = None
        self.__device = device

    def __set_args(self, bandwidth, sample_rate, time_duration):
        if (self.__bandwidth == bandwidth and self.__sample_rate == sample_rate
                and self.__time_duration == time_duration):
            return

        self.__bandwidth = bandwidth
        self.__sample_rate = sample_rate
        self.__time_duration = time_duration

        self.__window = get_window(('kaiser', self.SIGMA),
                                   self.TFA_WINDOW_SIZE)
        self.__start_spectro_f = math.ceil(
            0.5 * (1 - bandwidth / sample_rate) * (self.TFA_WINDOW_SIZE))
        self.__stop_spectro_f = self.TFA_WINDOW_SIZE - self.__start_spectro_f

    def convert(self, bandwidth: float, sample_rate: float,
                time_duration: float, iq_arr: np.ndarray) -> torch.Tensor:
        self.__set_args(bandwidth, sample_rate, time_duration)
        # Convert numpy array to tensor and move to GPU
        iq_tensor = torch.from_numpy(iq_arr).to(self.__device)
        # Compute spectrogram
        spectro = self.__to_spectrogram(iq_tensor)
        # Convert to dBm
        spectro_dbm = self.__spec_to_dbm(spectro)
        return spectro_dbm

    def __to_spectrogram(self, iq_tensor: torch.Tensor) -> torch.Tensor:
        torch_window = torch.tensor(self.__window,
                                    device=iq_tensor.device,
                                    dtype=torch.float32)
        spectro_complex = torch.stft(iq_tensor,
                                     n_fft=self.TFA_WINDOW_SIZE,
                                     hop_length=int(self.TFA_WINDOW_SIZE *
                                                    (1 - self.TFA_OVERLAP)),
                                     window=torch_window,
                                     return_complex=True)
        spectro = torch.abs(spectro_complex)
        spectro = torch.fft.fftshift(spectro, dim=0)  # pylint: disable=not-callable
        # spectro = spectro[self.__start_spectro_f:self.__stop_spectro_f, :]
        if self.__device != 'cpu':
            spectro = spectro.cpu().numpy()
        return spectro

    def __spec_to_dbm(self,
                      spectro: torch.Tensor,
                      down_sample=1) -> torch.Tensor:
        # Ensure spectrogram values are positive for the log operation
        spectro_clipped = torch.clamp(spectro, min=1e-10)
        spectro_dbm = 10 * torch.log10(
            spectro_clipped[::down_sample, ::down_sample]).t()
        return spectro_dbm
