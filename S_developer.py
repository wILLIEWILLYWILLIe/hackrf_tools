#!/usr/bin/env python3
"""
This module processes IQ data files based on JSON metadata, creates spectrograms
using a SpectrogramConverter, applies frequency shifting and filtering for desired
channels, and saves the images to disk. Optionally, it supports interactive
selection of files.
"""

import os
import json
import numpy as np
import torch
import cv2
from scipy.signal import firwin, lfilter, resample_poly
import matplotlib.pyplot as plt

from S_converter import SpectrogramConverter, SpectrogramConverterMod

# --------------------
# Module-Level Constants
# --------------------
REPLAY_DIR = '/home/iwave/Desktop/0110_2_ant129_in1'  # Adjust the path as needed
PLOT = 0        # Set to 1 to save normal spectrogram images
PICK = 0        # Set to 1 to enable interactive picking

TOTAL_CHANNEL_2G = [2.4145e9, 2.4295e9, 2.4445e9, 2.4595e9]
TOTAL_CHANNEL_5G = [5.7565e9, 5.7765e9, 5.7965e9, 5.8165e9]
DID_SAMPLE_RATE = 15.36e6  # Desired channel bandwidth (Hz) for filtering
DESIRED_FINAL_LENGTH = 5376000 

NEED_PROCESS_IQ_FILES = [
    '1736500600.8764782_19228083.npy', 
    '1736500600.8764782_22336138.npy', 
    '1736500602.1502016_20182652.npy', 
    '1736500602.1502016_19228083.npy', 
    '1736500602.1502016_22336138.npy', 
    '1736500603.4315228_20182652.npy', 
]

# --------------------
# Helper Functions
# --------------------
def pick_did_image(spec_image: np.ndarray, npy_filename: str = '') -> bool:
    """
    Displays the spectrogram image and asks the user whether to select the associated npy file.
    Returns True if selected (input "y" or "Y"), False otherwise.
    """
    cv2.imshow("Spectrogram", spec_image)
    print(f"Displaying {npy_filename}. Press any key on the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    user_input = input("Save this npy filename? (y/Y to save, any other key to skip): ")
    if user_input.lower() == "y":
        print("Selected.")
        return True
    else:
        print("Skipped.")
        return False

# --------------------
# IQProcessor Class
# --------------------
class IQProcessor:
    """
    Encapsulates methods to process IQ data:
      - Converting raw IQ to spectrogram using a SpectrogramConverter.
      - Shifting and filtering IQ data to isolate a specific band.
      - Plotting the filtered spectrogram and saving it.
    """
    def __init__(self, directory: str, converter: SpectrogramConverter):
        """
        :param directory: Base directory for input/output files.
        :param converter: An instance of SpectrogramConverter.
        """
        self.replay_dir = directory
        self.converter = converter
        self.converter_mod = SpectrogramConverterMod()

    def get_spectrogram(self, iq_arr: np.ndarray,
                        bandwidth: float,
                        sample_rate: float,
                        time_duration: float,
                        capture_center: float) -> torch.Tensor:
        """
        Converts IQ data to a spectrogram.
        Stores necessary parameters for filtering (sample rate, bandwidth, etc.).
        :param iq_arr: Raw IQ data (numpy array of complex numbers)
        :param bandwidth: Bandwidth parameter (Hz)
        :param sample_rate: Sample rate (Hz)
        :param time_duration: Duration of captured signal (seconds)
        :param capture_center: Center frequency of capture (Hz)
        :return: Spectrogram tensor (in dBm)
        """
        self.__sample_rate = sample_rate
        self.__bandwidth = bandwidth
        self.__time_duration = time_duration
        self.__capture_center = capture_center
        return self.converter.convert(bandwidth, sample_rate, time_duration, iq_arr)
    


    def shift_iq(self, iq_arr: np.ndarray,
                target_center: float) -> np.ndarray:
        """
        Shifts the IQ data so that the target_center frequency is moved to baseband,
        :param iq_arr: Raw IQ data (numpy complex array).
        :param target_center: Desired frequency to be shifted to baseband (Hz).
        :return: Filtered IQ data (numpy complex array).
        """
        # Compute the frequency offset in Hz.
        offset_freq = target_center - self.__capture_center
        # Generate a time axis.
        t = np.arange(len(iq_arr)) / self.__sample_rate
        # Shift the spectrum by mixing with a complex exponential.
        iq_shifted = iq_arr * np.exp(-1j * 2 * np.pi * offset_freq * t)
        return iq_shifted.astype(np.complex64)
    
    def filter_iq(self, iq_arr: np.ndarray,
                desired_bw: float = DID_SAMPLE_RATE,
                numtaps: int = 101) -> np.ndarray:
        """
        Applies a lowpass FIR filter to isolate a band of width desired_bw.
        :param iq_arr: Raw IQ data (numpy complex array).
        :param desired_bw: Bandwidth to isolate (Hz).
        :param numtaps: Number of taps for FIR filter.
        :return: Filtered IQ data (numpy complex array).
        """
        # Design a lowpass FIR filter with cutoff = desired_bw/2.
        nyq_rate = self.__sample_rate / 2.0
        norm_cutoff = (desired_bw / 2.0) / nyq_rate
        taps = firwin(numtaps, norm_cutoff, window="hamming").astype(np.float32)
        # Filter the shifted signal.
        iq_filtered = lfilter(taps, 1.0, iq_arr)
        print(f"Filtered IQ data : {iq_filtered.shape}, type : {iq_filtered.dtype}, samples: {iq_filtered[:2]}")
        # print(f'    -Maximum IQ value : {np.max(iq_filtered)}, Minimum IQ value : {np.min(iq_filtered)}')
        return iq_filtered.astype(np.complex64)



    def plot_filtered_spectrogram(self,
                                  iq_arr: np.ndarray,
                                  target_center: float,
                                  bandwidth: float,
                                  sample_rate: float,
                                  time_duration: float,
                                  timestamp: float,
                                  sn: str):
        """
        Filters the IQ data for the given target frequency, converts it into a spectrogram,
        downsamples and normalizes the image, then saves it with the target center frequency in the filename.
        :param iq_arr: Raw IQ data.
        :param target_center: The target center frequency (Hz) to shift/filter.
        :param bandwidth: Bandwidth (Hz) for spectrogram conversion.
        :param sample_rate: Sample rate (Hz).
        :param time_duration: Duration of the signal (s).
        :param timestamp: Timestamp from metadata.
        :param sn: Serial number identifier.
        """
        print(f"{'-'*20} Processing {target_center/1e6:.2f} MHz {'-'*20}")
        print(f'Original IQ data : {iq_arr.shape}, type : {iq_arr.dtype}, samples: {iq_arr[:2]}')
        # print(f'    -Maximum IQ value : {np.max(iq_arr)}, Minimum IQ value : {np.min(iq_arr)}')

        # Shift the IQ data to baseband and filter it.  
        shifted_iq = self.shift_iq(iq_arr, target_center)
        # Filter the shifted IQ data to isolate the desired bandwidth.
        filtered_iq = self.filter_iq(shifted_iq, desired_bw=DID_SAMPLE_RATE)
        spectro_dbm = self.converter.convert(bandwidth, sample_rate, time_duration, filtered_iq)
        spectro_np = spectro_dbm.cpu().numpy()

        # Downsample and normalize the spectrogram image.
        DOWN_SAMPLE_FACTOR = 5 
        spec_norm = cv2.normalize(spectro_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        spec_norm = spec_norm.astype(np.uint8)
        spec_norm = spec_norm[::DOWN_SAMPLE_FACTOR, ::DOWN_SAMPLE_FACTOR]

        # Create an output directory and filename.
        out_dir = os.path.join(self.replay_dir, '_spec_images')
        out_dir = '_spec_images'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_filename = f"{timestamp}_{sn}_filtered_{int(target_center/1e6):d}MHz.png"
        cv2.imwrite(os.path.join(out_dir, out_filename), spec_norm)
        print(f"Saved filtered spectrogram image as {out_filename}")

        self.save_downsampled_iq(filtered_iq, timestamp, sn, target_center)
        print(f"{'-'*60}")
    
    def save_downsampled_iq(self,
                            iq_arr: np.ndarray,
                            timestamp: float,
                            sn: str,
                            target_center: float):
        """
        Downsamples the filtered IQ data to DID_SAMPLE_RATE (15.36 M) and duplicates its length to
        DESIRED_FINAL_LENGTH samples, then saves it as a binary file with the .iq extension.
        :param iq_arr: Filtered IQ data (numpy complex array).
        :param timestamp: Timestamp from metadata.
        :param sn: Serial number.
        :param target_center: Target center frequency used for filtering.
        """
        new_iq = self.downsample_and_duplicate(iq_arr)
        ################################################################
        sample_rate = 15.36e6
        bandwidth = 15.36e6  # For a full-band capture at this rate
        time_duration = 0.35  # seconds

        # Create an instance of the modified converter.
        spectro_dbm = self.converter_mod.convert(bandwidth, sample_rate, time_duration, new_iq)

        # Convert the spectrogram tensor to a CPU numpy array.
        spectro_np = spectro_dbm.cpu().numpy()

        # Plot using matplotlib.
        plt.figure(figsize=(5, 25))
        plt.imshow(spectro_np, aspect='auto', 
                #    origin='lower'
                   )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency bins")
        plt.title("Spec at 15.36e6 sample rate")
        plt.colorbar(label="Intensity (dBm)")
        plt.tight_layout()
        plt.show()
        ################################################################

        out_dir = '_filtered_iq'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Build a filename that shows the target center frequency in MHz.
        out_filename = f"{timestamp}_{sn}_filtered_{int(target_center/1e6)}MHz.iq"
        out_path = os.path.join(out_dir, out_filename)
        # Save the IQ data as binary. This writes the raw bytes of the numpy array.
        new_iq.tofile(out_path)
        print(f"Saved downsampled IQ data as {out_filename}")

        down_iq = np.fromfile(out_path, dtype=np.complex64)
        print("Downsampled IQ sample:", down_iq[:10])

    def downsample_and_duplicate(self, iq_arr: np.ndarray) -> np.ndarray:
        """
        Downsamples the given IQ data from the processor's sample rate to DID_SAMPLE_RATE,
        then duplicates (tiles) the signal until its length equals DESIRED_FINAL_LENGTH.
            :param iq_arr: Filtered IQ data (numpy complex array).
            :return: Downsampled and length-adjusted IQ data.
        """
        # Calculate decimation ratio.
        # Using resample_poly to perform polyphase resampling.
        # We want to change the rate from self.__sample_rate to DID_SAMPLE_RATE.
        decim_factor = self.__sample_rate / DID_SAMPLE_RATE
        # Round the factor to nearest integer for the polyphase filter.
        down_factor = int(round(decim_factor))
        new_iq = resample_poly(iq_arr, up=1, down=down_factor,window=1)
        new_iq = new_iq.astype(np.complex64)
        new_iq = np.around(new_iq*32768)
        print(f"Downsample {len(iq_arr)} -> {len(new_iq)} down_factor {down_factor}")
        print(f'    -Maximum IQ value : {np.max(new_iq)}, Minimum IQ value : {np.min(new_iq)}')
        # Duplicate (tile) until we have at least DESIRED_FINAL_LENGTH samples.
        L = new_iq.shape[0]
        # dulpicate the first 50 percentage and added before new iq
        new_iq = np.concatenate((new_iq[:int(L/2)], new_iq))
        new_iq = np.concatenate((new_iq[:int(3*L/4)], new_iq))
        new_iq = np.concatenate((new_iq[:int(3*L/4)], new_iq))
        new_iq = np.concatenate((new_iq[:int(3*L/4)], new_iq))
        new_iq = np.concatenate((new_iq[:int(3*L/4)], new_iq))
        if L < DESIRED_FINAL_LENGTH:
            repeats = int(np.ceil(DESIRED_FINAL_LENGTH / L))
            new_iq = np.tile(new_iq, repeats)
        new_iq = new_iq[:DESIRED_FINAL_LENGTH]
        print(f"Duplicated IQ length: {new_iq.shape[0]} samples, dtype: {new_iq.dtype}, samples: {new_iq[:2]}")
        return new_iq

# --------------------
# Main Processing Function
# --------------------
def process_all_entries():
    """
    Reads JSON metadata files from REPLAY_DIR, processes each IQ file accordingly,
    generates a normal spectrogram image, and for selected files (in NEED_PROCESS_IQ_FILES)
    applies additional filtering and saves filtered spectrogram images.
    """
    converter = SpectrogramConverter()
    processor = IQProcessor(REPLAY_DIR, converter)
    selected_files = []

    # Find all JSON files with the expected suffix.
    json_files = sorted([f for f in os.listdir(REPLAY_DIR) if f.endswith('_data.json')])
    print(f"Found {len(json_files)} JSON files.")

    all_entries = []
    for jf in json_files:
        json_path = os.path.join(REPLAY_DIR, jf)
        with open(json_path, 'r') as fp:
            info = json.load(fp)
            all_entries.append(info)

    # Sort JSON entries by timestamp.
    all_entries.sort(key=lambda x: x["timestamp"])
    print(f"Found {len(all_entries)} entries. Starting processing...")

    # Process each JSON entry.
    for entry in all_entries:
        timestamp = entry["timestamp"]
        serial_nums = entry["serial_numbers"]
        sa_info = entry.get("sa", {})
        # Optional: iq_info can be used if needed.
        # print(f"Processing timestamp: {timestamp}, serial numbers: {serial_nums}")

        for sn in serial_nums:
            npy_filename = f"{timestamp}_{sn}.npy"
            npy_path = os.path.join(REPLAY_DIR, npy_filename)
            if not os.path.exists(npy_path):
                print(f"Warning: {npy_path} does not exist, skipping.")
                continue

            iq_arr = np.load(npy_path)
            if sn not in sa_info:
                print(f"Warning: No settings in sa_info for serial number {sn}, skipping.")
                continue

            sample_rate   = sa_info[sn]["sample_rate"]
            bandwidth     = sa_info[sn]["bandwidth"]
            time_duration = sa_info[sn]["time_duration"]
            capture_center = sa_info[sn].get("center_freq", 5.79e9)

            # Generate and save the normal (unfiltered) spectrogram.
            spectro_dbm = processor.get_spectrogram(iq_arr, bandwidth, sample_rate, time_duration, capture_center)
            spectro_np = spectro_dbm.cpu().numpy()
            if PLOT:
                DOWN_SAMPLE_FACTOR = 5
                spec_norm = cv2.normalize(spectro_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                spec_norm = spec_norm.astype(np.uint8)
                spec_norm = spec_norm[::DOWN_SAMPLE_FACTOR, ::DOWN_SAMPLE_FACTOR]
                
                out_dir = os.path.join(REPLAY_DIR, '_spec_images')
                out_dir = '_spec_images'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_filename = f"{timestamp}_{sn}.png"
                cv2.imwrite(os.path.join(out_dir, out_filename), spec_norm)
                print(f"Saved normal spectrogram image as {out_filename}")

            # If the file is in the list for additional processing, apply filtering.
            if npy_filename in NEED_PROCESS_IQ_FILES:
                print(f"{'='*20} Processing {npy_filename} {'='*20}")
                # Choose target centers based on capture_center.
                if capture_center > 5e9:
                    target_centers = TOTAL_CHANNEL_5G
                else:
                    target_centers = TOTAL_CHANNEL_2G

                # For each target center, plot and save the filtered spectrogram.
                for target_center in target_centers:
                    processor.plot_filtered_spectrogram(iq_arr, target_center, bandwidth,
                                                        sample_rate, time_duration, timestamp, sn)

    # Optionally, if PICK is enabled, interactively select files.
    if PICK and selected_files:
        txt_filename = os.path.join(REPLAY_DIR, "selected_files.txt")
        with open(txt_filename, 'w') as f:
            for fname in selected_files:
                f.write(fname + "\n")
        print(f"Selected filenames saved to {txt_filename}")
        print("Selected files:", selected_files)
    else:
        print("No files were selected.")

# --------------------
# Main Entry Point
# --------------------
def main():
    process_all_entries()

if __name__ == "__main__":
    main()
