#!/usr/bin/env python3

import argparse
import numpy as np

def generate_hopping(filename, start_freq, end_freq, hop_step, hop_duration, sample_rate, total_duration, amplitude=0.5):
    """
    Generate a frequency-hopping signal and save as interleaved I/Q samples.

    :param filename: Output filename (e.g., 'hopping_iq.bin')
    :param start_freq: Starting frequency of the hop in Hz
    :param end_freq: Ending frequency of the hop in Hz
    :param hop_step: Frequency step between hops in Hz
    :param hop_duration: Duration of each hop in seconds
    :param sample_rate: Sample rate in samples per second
    :param total_duration: Total duration of the signal in seconds
    :param amplitude: Amplitude of the signal (0 < amplitude <= 1)
    """
    # Generate list of hop frequencies
    if hop_step <= 0:
        raise ValueError("Hop step must be a positive number.")
    if start_freq > end_freq:
        raise ValueError("Start frequency must be less than or equal to end frequency.")

    hop_frequencies = np.arange(start_freq, end_freq + hop_step, hop_step)
    num_hops = len(hop_frequencies)
    print(f"Hop Frequencies (Hz): {hop_frequencies}")
    print(f"Number of Hops: {num_hops}")

    # Calculate number of samples per hop
    samples_per_hop = int(sample_rate * hop_duration)
    print(f"Samples per Hop: {samples_per_hop}")

    # Calculate total number of samples
    total_samples = int(sample_rate * total_duration)
    print(f"Total Samples: {total_samples}")

    # Initialize IQ array
    iq_signal = np.array([], dtype=np.complex64)

    # Generate hopping signal
    for i in range(int(np.ceil(total_samples / samples_per_hop))):
        freq = hop_frequencies[i % num_hops]
        t = np.arange(samples_per_hop) / sample_rate
        waveform = amplitude * np.exp(1j * 2 * np.pi * freq * t)
        iq_signal = np.concatenate((iq_signal, waveform))
        print(f"Hop {i+1}: Frequency = {freq} Hz")

    # Trim to total_samples
    iq_signal = iq_signal[:total_samples]

    # Normalize to 8-bit unsigned integers (0-255)
    i_samples = ((iq_signal.real + 1.0) * 127.5).astype(np.uint8)
    q_samples = ((iq_signal.imag + 1.0) * 127.5).astype(np.uint8)

    # Interleave I and Q
    interleaved = np.empty((i_samples.size + q_samples.size,), dtype=np.uint8)
    interleaved[0::2] = i_samples
    interleaved[1::2] = q_samples

    # Save to file
    interleaved.tofile(filename)
    print(f"Frequency-hopping signal saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate a frequency-hopping IQ data file.")
    parser.add_argument('--start-freq', type=float, required=True, help='Hopping start frequency in Hz')
    parser.add_argument('--end-freq', type=float, required=True, help='Hopping end frequency in Hz')
    parser.add_argument('--hop-step', type=float, required=True, help='Frequency step between hops in Hz')
    parser.add_argument('--hop-duration', type=float, required=True, help='Duration of each hop in seconds')
    parser.add_argument('-s', '--sample-rate', type=float, required=True, help='Sample rate in Hz')
    parser.add_argument('-d', '--duration', type=float, required=True, help='Total duration in seconds')
    parser.add_argument('-a', '--amplitude', type=float, default=0.5, help='Amplitude (0 < amplitude <= 1)')
    parser.add_argument('-o', '--output', type=str, default='hopping_iq.bin', help='Output IQ file name')

    args = parser.parse_args()

    if not (0 < args.amplitude <= 1):
        raise ValueError("Amplitude must be greater than 0 and less than or equal to 1.")
    if args.hop_duration <= 0:
        raise ValueError("Hop duration must be a positive number.")
    if args.duration <= 0:
        raise ValueError("Total duration must be a positive number.")
    if args.sample_rate <= 0:
        raise ValueError("Sample rate must be a positive number.")

    generate_hopping(
        filename=args.output,
        start_freq=args.start_freq,
        end_freq=args.end_freq,
        hop_step=args.hop_step,
        hop_duration=args.hop_duration,
        sample_rate=args.sample_rate,
        total_duration=args.duration,
        amplitude=args.amplitude
    )

if __name__ == "__main__":
    main()
