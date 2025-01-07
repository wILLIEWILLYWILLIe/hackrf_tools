#!/usr/bin/env python3

import argparse
import numpy as np

def generate_chirp(filename, start_freq, end_freq, sample_rate, duration, amplitude=0.5):
    """
    Generate a chirp signal and save as interleaved I/Q samples.

    :param filename: Output filename (e.g., 'chirp_iq.bin')
    :param start_freq: Starting frequency of the chirp in Hz
    :param end_freq: Ending frequency of the chirp in Hz
    :param sample_rate: Sample rate in samples per second
    :param duration: Duration in seconds
    :param amplitude: Amplitude of the chirp (0 < amplitude <= 1)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Linear chirp: frequency increases from start_freq to end_freq over time
    k = (end_freq - start_freq) / duration  # Chirp rate
    waveform = amplitude * np.exp(1j * (2 * np.pi * start_freq * t + np.pi * k * t**2))
    
    # Normalize to 8-bit unsigned integers (0-255)
    # HackRF expects samples in uint8 format with I and Q interleaved
    i_samples = ((waveform.real + 1.0) * 127.5).astype(np.uint8)
    q_samples = ((waveform.imag + 1.0) * 127.5).astype(np.uint8)
    
    # Interleave I and Q
    interleaved = np.empty((i_samples.size + q_samples.size,), dtype=np.uint8)
    interleaved[0::2] = i_samples
    interleaved[1::2] = q_samples
    
    # Save to file
    interleaved.tofile(filename)
    print(f"Chirp signal saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate a chirp IQ data file.")
    parser.add_argument('--start-freq', type=float, required=True, help='Chirp start frequency in Hz')
    parser.add_argument('--end-freq', type=float, required=True, help='Chirp end frequency in Hz')
    parser.add_argument('-s', '--sample-rate', type=float, required=True, help='Sample rate in Hz')
    parser.add_argument('-d', '--duration', type=float, required=True, help='Duration in seconds')
    parser.add_argument('-a', '--amplitude', type=float, default=0.5, help='Amplitude (0 < amplitude <= 1)')
    parser.add_argument('-o', '--output', type=str, default='chirp_iq.bin', help='Output IQ file name')

    args = parser.parse_args()

    if not (0 < args.amplitude <= 1):
        raise ValueError("Amplitude must be greater than 0 and less than or equal to 1.")
    if args.start_freq >= args.end_freq:
        raise ValueError("Start frequency must be less than end frequency.")

    generate_chirp(args.output, args.start_freq, args.end_freq, args.sample_rate, args.duration, args.amplitude)

if __name__ == "__main__":
    main()
