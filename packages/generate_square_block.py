#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.signal import butter, sosfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.

    :param lowcut: Low cutoff frequency in Hz
    :param highcut: High cutoff frequency in Hz
    :param fs: Sampling rate in Hz
    :param order: Order of the filter
    :return: Second-order sections for the filter
    """
    # Ensure lowcut is not negative and greater than 0
    if lowcut <= 0:
        raise ValueError("Low cutoff frequency must be greater than 0 Hz.")
    # Ensure highcut does not exceed Nyquist
    highcut = min(highcut, fs / 2 - 1)
    if highcut <= lowcut:
        raise ValueError("High cutoff frequency must be greater than low cutoff frequency.")
    sos = butter(order, [lowcut, highcut], analog=False, btype='band', output='sos', fs=fs)
    return sos

def generate_square_block(filename, center_freq, bandwidth, block_duration, time_gap, sample_rate, total_duration, amplitude=0.5):
    """
    Generate a square block signal with specified bandwidth and center frequency,
    interleaved with silent gaps.

    :param filename: Output filename (e.g., 'square_block_iq.bin')
    :param center_freq: Center frequency of the block in Hz (baseband)
    :param bandwidth: Bandwidth of the block in Hz
    :param block_duration: Duration of each block in seconds
    :param time_gap: Duration of silence between blocks in seconds
    :param sample_rate: Sample rate in samples per second
    :param total_duration: Total duration of the signal in seconds
    :param amplitude: Amplitude of the signal (0 < amplitude <= 1)
    """
    # Calculate number of blocks
    block_total_time = block_duration + time_gap
    num_blocks = int(np.floor(total_duration / block_total_time))
    print(f"Number of Blocks: {num_blocks}")
    
    # Calculate low and high cutoff frequencies
    lowcut = center_freq - (bandwidth / 2)
    highcut = center_freq + (bandwidth / 2)
    
    # Adjust lowcut and highcut to be within valid range
    lowcut = max(lowcut, 0.0)
    highcut = min(highcut, sample_rate / 2 - 1)
    
    print(f"Bandpass Filter: {lowcut} Hz to {highcut} Hz")
    
    # Validate cutoff frequencies
    if lowcut <= 0 or highcut <= lowcut:
        raise ValueError("Invalid bandpass filter settings. Ensure 0 < lowcut < highcut < Nyquist frequency.")
    
    # Design bandpass filter
    sos = butter_bandpass(lowcut, highcut, sample_rate, order=6)
    
    # Initialize IQ signal
    iq_signal = np.array([], dtype=np.complex64)
    
    for i in range(num_blocks):
        # Generate white noise
        num_samples = int(block_duration * sample_rate)
        noise = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        noise *= amplitude
        
        # Apply bandpass filter
        filtered_noise = sosfilt(sos, noise)
        
        # Append to IQ signal
        iq_signal = np.concatenate((iq_signal, filtered_noise))
        print(f"Block {i+1}: Active for {block_duration} seconds")
        
        # Append silence (zeros)
        num_gap_samples = int(time_gap * sample_rate)
        silence = np.zeros(num_gap_samples, dtype=np.complex64)
        iq_signal = np.concatenate((iq_signal, silence))
        print(f"Block {i+1}: Silent for {time_gap} seconds")
    
    # Calculate remaining time
    remaining_time = total_duration - num_blocks * block_total_time
    if remaining_time > 0:
        # Generate final block
        num_samples = int(remaining_time * sample_rate)
        noise = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        noise *= amplitude
        filtered_noise = sosfilt(sos, noise)
        iq_signal = np.concatenate((iq_signal, filtered_noise))
        print(f"Final Block: Active for {remaining_time} seconds")
    
    # Normalize to 8-bit unsigned integers (0-255)
    i_samples = ((iq_signal.real + 1.0) * 127.5).astype(np.uint8)
    q_samples = ((iq_signal.imag + 1.0) * 127.5).astype(np.uint8)
    
    # Interleave I and Q
    interleaved = np.empty((i_samples.size + q_samples.size,), dtype=np.uint8)
    interleaved[0::2] = i_samples
    interleaved[1::2] = q_samples
    
    # Save to file
    interleaved.tofile(filename)
    print(f"Square block signal saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate a square block IQ data file.")
    parser.add_argument('-c', '--center-freq', type=float, required=True, help='Center frequency in Hz (baseband)')
    parser.add_argument('-b', '--bandwidth', type=float, required=True, help='Bandwidth of the block in Hz')
    parser.add_argument('-bd', '--block-duration', type=float, required=True, help='Duration of each block in seconds')
    parser.add_argument('-tg', '--time-gap', type=float, required=True, help='Duration of silence between blocks in seconds')
    parser.add_argument('-s', '--sample-rate', type=float, required=True, help='Sample rate in Hz')
    parser.add_argument('-d', '--duration', type=float, required=True, help='Total duration in seconds')
    parser.add_argument('-a', '--amplitude', type=float, default=0.5, help='Amplitude (0 < amplitude <= 1)')
    parser.add_argument('-o', '--output', type=str, default='square_block_iq.bin', help='Output IQ file name')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0 < args.amplitude <= 1):
        raise ValueError("Amplitude must be greater than 0 and less than or equal to 1.")
    if args.bandwidth <= 0:
        raise ValueError("Bandwidth must be a positive number.")
    if args.block_duration <= 0:
        raise ValueError("Block duration must be a positive number.")
    if args.time_gap < 0:
        raise ValueError("Time gap cannot be negative.")
    if args.sample_rate <= 0:
        raise ValueError("Sample rate must be a positive number.")
    if args.duration <= 0:
        raise ValueError("Total duration must be a positive number.")
    if args.bandwidth > args.sample_rate / 2:
        raise ValueError("Bandwidth cannot exceed Nyquist frequency (sample_rate / 2).")
    
    generate_square_block(
        filename=args.output,
        center_freq=args.center_freq,
        bandwidth=args.bandwidth,
        block_duration=args.block_duration,
        time_gap=args.time_gap,
        sample_rate=args.sample_rate,
        total_duration=args.duration,
        amplitude=args.amplitude
    )

if __name__ == "__main__":
    main()
