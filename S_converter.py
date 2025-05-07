''' This module defines the SpectrogramConverter class for converting iq array
to spectrogram in CUDA Tensor.'''

import numpy as np
import torch
import math
from scipy.signal import get_window

class SpectrogramConverter:
    '''Convert IQ data to spectrogram'''
    TFA_WINDOW_SIZE = 1500
    TFA_OVERLAP = 0.3
    SIGMA = 5

    def __init__(self, name='SpectrogramConverter'):
        self.__bandwidth = None
        self.__sample_rate = None
        self.__time_duration = None
        self.__window = None
        self.__start_spectro_f = None
        self.__stop_spectro_f = None

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
        iq_tensor = torch.from_numpy(iq_arr).to("cuda")
        # Compute spectrogram
        spectro = self.__to_spectrogram(iq_tensor)
        # Convert to dBm
        spectro_dbm = self.__spec_to_dbm(spectro)
        # Sqrt and convert to dbm
        return spectro_dbm

    def __to_spectrogram(self, iq_tensor: torch.Tensor) -> torch.Tensor:
        '''
        ref: https://signalhound.com/sigdownloads/SDK/online_docs/sm_api/index.html?srsltid=AfmBOoq62W0ikYbQ3BEGJHg1ZnHPBVdTp-6HNlihA80KUry6gdsIJEw5
        Sample Power (dBm) = 10.0 * log10(re*re + im*im);
        '''
        torch_window = torch.tensor(self.__window,
                                    device=iq_tensor.device,
                                    dtype=torch.float32)
        spectro_complex = torch.stft(iq_tensor,
                                     n_fft=self.TFA_WINDOW_SIZE,
                                     hop_length=int(self.TFA_WINDOW_SIZE *
                                                    (1 - self.TFA_OVERLAP)),
                                     window=torch_window,
                                     return_complex=True)
        spectro = (torch.abs(spectro_complex)/self.TFA_WINDOW_SIZE)**2
        spectro = torch.fft.fftshift(spectro, dim=0)  # pylint: disable=not-callable
        spectro = spectro[self.__start_spectro_f:self.__stop_spectro_f, :]
        return spectro

    def __spec_to_dbm(self,
                      spectro: torch.Tensor,
                      down_sample=1) -> torch.Tensor:
        spectro_dbm = 10 * torch.log10(
            spectro[::down_sample, ::down_sample]).t()
        return spectro_dbm


class SpectrogramConverterMod:
    """
    Converts IQ data to spectrogram with parameters optimized for a 15.36e6 sample rate.
    """
    TFA_WINDOW_SIZE = 92    # Reduced window size compared to the original converter
    TFA_OVERLAP = 0.3        # 50% overlap for smoother time resolution
    SIGMA = 5                # Kaiser window parameter

    def __init__(self, name='SpectrogramConverterMod'):
        self.__bandwidth = None
        self.__sample_rate = None
        self.__time_duration = None
        self.__window = None
        self.__start_spectro_f = None
        self.__stop_spectro_f = None

    def __set_args(self, bandwidth: float, sample_rate: float, time_duration: float):
        """
        Sets parameters for the spectrogram conversion and computes the window and the frequency slice.
        """
        # If the parameters have not changed, do not reset
        if (self.__bandwidth == bandwidth and 
            self.__sample_rate == sample_rate and 
            self.__time_duration == time_duration):
            return

        self.__bandwidth = bandwidth
        self.__sample_rate = sample_rate
        self.__time_duration = time_duration

        # Create a Kaiser window with the given SIGMA parameter.
        self.__window = get_window(('kaiser', self.SIGMA), self.TFA_WINDOW_SIZE)


    def convert(self, bandwidth: float, sample_rate: float, 
                time_duration: float, iq_arr: np.ndarray) -> torch.Tensor:

        self.__set_args(bandwidth, sample_rate, time_duration)
        # Transfer the IQ data to a Torch tensor on GPU.
        iq_tensor = torch.from_numpy(iq_arr).to("cuda")
        # Compute the spectrogram.
        spectro = self.__to_spectrogram(iq_tensor)
        # Convert the magnitude to dBm.
        spectro_dbm = self.__spec_to_dbm(spectro)
        return spectro_dbm

    def __to_spectrogram(self, iq_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs STFT on the IQ tensor and extracts the desired frequency bins.
        """
        torch_window = torch.tensor(self.__window, device=iq_tensor.device, dtype=torch.float32)
        # Calculate hop length based on overlap.
        hop_length = int(self.TFA_WINDOW_SIZE * (1 - self.TFA_OVERLAP))
        spectro_complex = torch.stft(iq_tensor,
                                     n_fft=self.TFA_WINDOW_SIZE,
                                     hop_length=hop_length,
                                     window=torch_window,
                                     return_complex=True)
        # Normalize the STFT magnitude.
        spectro = (torch.abs(spectro_complex) / self.TFA_WINDOW_SIZE) ** 2
        # Shift zero frequency to center.
        spectro = torch.fft.fftshift(spectro, dim=0)
        return spectro

    def __spec_to_dbm(self, spectro: torch.Tensor, down_sample: int = 1) -> torch.Tensor:
        """
        Converts spectrogram magnitude to dBm.
        """
        spectro_dbm = 10 * torch.log10(spectro[::down_sample, ::down_sample]).t()
        return spectro_dbm