#!/usr/bin/env python3

import argparse
import numpy as np

def generate_tone(filename, frequency, sample_rate, duration, amplitude=0.5):
    """
    Generate a sine wave tone and save as interleaved I/Q samples.

    :param filename: Output filename (e.g., 'tone_iq.bin')
    :param frequency: Tone frequency in Hz
    :param sample_rate: Sample rate in samples per second
    :param duration: Duration in seconds
    :param amplitude: Amplitude of the sine wave (0 < amplitude <= 1)
    """
    t = np.arange(int(sample_rate * duration)) / sample_rate
    waveform = amplitude * np.exp(1j * 2 * np.pi * frequency * t)
    
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
    print(f"Tone saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate a tone IQ data file.")
    parser.add_argument('-f', '--frequency', type=float, required=True, help='Tone frequency in Hz')
    parser.add_argument('-s', '--sample-rate', type=float, required=True, help='Sample rate in Hz')
    parser.add_argument('-d', '--duration', type=float, required=True, help='Duration in seconds')
    parser.add_argument('-a', '--amplitude', type=float, default=0.5, help='Amplitude (0 < amplitude <= 1)')
    parser.add_argument('-o', '--output', type=str, default='tone_iq.bin', help='Output IQ file name')

    args = parser.parse_args()

    if not (0 < args.amplitude <= 1):
        raise ValueError("Amplitude must be greater than 0 and less than or equal to 1.")

    generate_tone(args.output, args.frequency, args.sample_rate, args.duration, args.amplitude)

if __name__ == "__main__":
    main()
