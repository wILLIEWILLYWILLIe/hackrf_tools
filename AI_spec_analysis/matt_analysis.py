import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# from spec_plot import SpectrogramConverter

try:
    # Only import HackRF if needed
    from pyhackrf2 import HackRF
except ImportError:
    HackRF = None

def analyze_modulation(iq_signal):
    """
    Analyze the modulation type of an IQ signal using the constellation diagram.

    Parameters:
        iq_signal (numpy.ndarray): Complex-valued array of IQ samples.

    Returns:
        str: The detected modulation type ('BPSK', 'QPSK', or 'Unknown').
    """
    # Normalize the IQ signal (remove DC component)
    iq_signal -= np.mean(iq_signal)

    # Find the unique points in the constellation 
    tolerance = 1.5
    unique_points = []

    for point in iq_signal:
        found = False
        for unique_point in unique_points:
            if np.abs(point - unique_point) < tolerance:
                found = True
                break
        if not found:
            unique_points.append(point)

    # Number of unique points determines modulation type
    num_unique_points = len(unique_points)

    if num_unique_points == 2:
        return "BPSK"
    elif num_unique_points == 4:
        return "QPSK"
    elif num_unique_points == 16:
        return "16QAM"
    else:
        return "Unknown"

def generate_16qam_signal(num_samples, snr_db=10):
    """
    Generate a random 16-QAM signal with the specified number of samples and SNR.

    Parameters:
        num_samples (int): Number of IQ samples to generate.
        snr_db (float): Signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: Complex-valued 16-QAM signal.
    """
    # Define 16-QAM constellation points (4x4 grid)
    real_parts = np.array([-3, -1, 1, 3])  # I-axis amplitude levels
    imag_parts = np.array([-3, -1, 1, 3])  # Q-axis amplitude levels

    # Generate random indices for selecting constellation points
    real_indices = np.random.choice(len(real_parts), size=num_samples)
    imag_indices = np.random.choice(len(imag_parts), size=num_samples)

    # Create 16-QAM symbols
    data = real_parts[real_indices] + 1j * imag_parts[imag_indices]

    # Compute signal power
    signal_power = np.mean(np.abs(data)**2)

    # Compute noise power
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    # Add Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    qam_signal = data + noise

    return qam_signal

def generate_bpsk_signal(num_samples, snr_db=10):
    """
    Generate a random BPSK signal with the specified number of samples and SNR.

    Parameters:
        num_samples (int): Number of IQ samples to generate.
        snr_db (float): Signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: Complex-valued BPSK signal.
    """
    # Define BPSK symbols (+1 and -1)
    bpsk_symbols = np.array([1, -1])

    # Randomly select BPSK symbols
    data = np.random.choice(bpsk_symbols, size=num_samples)

    # Compute signal power
    signal_power = np.mean(np.abs(data)**2)

    # Compute noise power
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    # Add Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    bpsk_signal = data + noise

    return bpsk_signal

def plot_constellation(iq_signal, title):
    """
    Plot the constellation diagram of an IQ signal.

    Parameters:
        iq_signal (numpy.ndarray): Complex-valued array of IQ samples.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(iq_signal.real, iq_signal.imag, 'o', markersize=2, alpha=0.6)
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.title(title)
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_time_domain_signal(iq_signal, title):
    """
    Plot the time-domain representation of an IQ signal.

    Parameters:
        iq_signal (numpy.ndarray): Complex-valued array of IQ samples.
        title (str): Title for the plot.
    """
    time = np.arange(len(iq_signal))
    plt.figure(figsize=(12, 6))
    plt.plot(time, iq_signal.real, label='In-Phase')
    plt.plot(time, iq_signal.imag, label='Quadrature')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_qpsk_signal(num_samples, snr_db=10):
    """
    Generate a random QPSK signal with the specified number of samples and SNR.

    Parameters:
        num_samples (int): Number of IQ samples to generate.
        snr_db (float): Signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: Complex-valued QPSK signal.
    """
    # Define QPSK constellation points
    qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])

    # Randomly select constellation points
    data = np.random.choice(qpsk_symbols, size=num_samples)

    # Compute signal power
    signal_power = np.mean(np.abs(data)**2)

    # Compute noise power
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    # Add Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    qpsk_signal = data + noise

    return qpsk_signal

def generate_ofdm_signal(num_symbols, num_subcarriers, cp_length, snr_db=20):
    """
    Generate an OFDM signal with the specified parameters.

    Parameters:
        num_symbols (int): Number of OFDM symbols.
        num_subcarriers (int): Number of subcarriers.
        cp_length (int): Length of the cyclic prefix.
        snr_db (float): Signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: Time-domain OFDM signal.
    """
    # Step 1: Generate random QPSK symbols for each subcarrier
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j])  # QPSK
    symbols = np.random.choice(constellation, size=(num_symbols, num_subcarriers))
    
    # Step 2: Perform IFFT to convert frequency-domain to time-domain
    time_domain_symbols = np.fft.ifft(symbols, axis=1)
    
    # Step 3: Add cyclic prefix
    cp = time_domain_symbols[:, -cp_length:]  # Take last cp_length samples as cyclic prefix
    ofdm_symbols_with_cp = np.hstack((cp, time_domain_symbols))
    
    # Step 4: Flatten the symbols to create a continuous signal
    ofdm_signal = ofdm_symbols_with_cp.flatten()
    
    # Step 5: Add Gaussian noise
    signal_power = np.mean(np.abs(ofdm_signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(ofdm_signal)) + 1j * np.random.randn(len(ofdm_signal)))
    ofdm_signal_noisy = ofdm_signal + noise
    
    return ofdm_signal_noisy

def plot_ofdm_signal(ofdm_signal, num_subcarriers, cp_length):
    """
    Plot the OFDM signal in both time and frequency domains.

    Parameters:
        ofdm_signal (numpy.ndarray): The OFDM signal.
        num_subcarriers (int): Number of subcarriers.
        cp_length (int): Length of the cyclic prefix.
    """
    # Time-domain representation
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(np.real(ofdm_signal), label="Real Part")
    plt.plot(np.imag(ofdm_signal), label="Imaginary Part", linestyle="--")
    plt.title("Time-Domain OFDM Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Frequency-domain representation (FFT)
    freq_domain = np.fft.fft(ofdm_signal[cp_length:cp_length+num_subcarriers])  # Remove CP for plotting
    plt.subplot(2, 1, 2)
    plt.stem(np.abs(freq_domain), use_line_collection=True)
    plt.title("Frequency-Domain OFDM Signal")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()

def load_iq_data(filename):
    """Load IQ samples from a .npy file."""
    iq_data = np.load(filename)
    return iq_data.flatten()  # Flatten in case of multiple segments
def normalize_iq(iq_data):
    """Normalize IQ data to unit circle."""
    iq_data -= np.mean(iq_data)  # Remove DC component
    # iq_data /= np.abs(iq_data).max()  # Normalize magnitude
    iq_data /= np.max(np.abs(iq_data))
    return iq_data

def main():
    TEST_WITH_FAKE_DATA = False
    TEST_QPSK_SIGNAL = False
    TEST_BPSK_SIGNAL = False
    TEST_16QAM_SIGNAL = False
    TEST_OFDM_SIGNAL = False

    # Scanning parameters
    sample_rate = 20e6  # Sampling rate 
    start_freq = 5780e6  # Starting frequency 
    end_freq = 5800e6  # Ending frequency 
    step_freq = sample_rate  # Bandwidth per scan step 
    sample_time_interval = 5e-3  # Time per scan step 
    samples_to_read = int(sample_time_interval * sample_rate)  # Samples per step
    segments = int((end_freq - start_freq) // step_freq)  # Number of segments

    # Gain settings
    LNA_GAIN = 8  # LNA gain (range: 0 ~ 40)
    VGA_GAIN = 8  # VGA gain (range: 0 ~ 62)

    # Save scan results
    IQ_data = []
    # filename = "SG_Signal/4_LTE_20MHz_64QAM_cable_5780MHz.npy"
    # iq_data = load_iq_data(filename)
    

    if TEST_QPSK_SIGNAL:
        # Generate a QPSK signal for testing
        num_samples = 10000  # Length of the signal
        snr_db = 20          # SNR
        qpsk_signal = generate_qpsk_signal(num_samples, snr_db)

        # Analyze and plot the QPSK signal
        modulation_type = analyze_modulation(qpsk_signal)
        print(f"Detected modulation type: {modulation_type}")

        plot_constellation(qpsk_signal, f"Generated QPSK Signal ({modulation_type})")
        plot_time_domain_signal(qpsk_signal, "Time-Domain Representation of QPSK Signal")
        return
    
    if TEST_BPSK_SIGNAL:
        # Generate a BPSK signal for testing
        num_samples = 10000  # Length of the signal
        snr_db = 20          # SNR
        bpsk_signal = generate_bpsk_signal(num_samples, snr_db)

        # Analyze and plot the BPSK signal
        modulation_type = analyze_modulation(bpsk_signal)
        print(f"Detected modulation type: {modulation_type}")

        plot_constellation(bpsk_signal, f"Generated BPSK Signal ({modulation_type})")
        plot_time_domain_signal(bpsk_signal, "Time-Domain Representation of BPSK Signal")

        return

    if TEST_16QAM_SIGNAL:
        # Generate a 16-QAM signal for testing
        num_samples = 10000  # Length of the signal
        snr_db = 20          # SNR
        qam_signal = generate_16qam_signal(num_samples, snr_db)

        # Analyze and plot the 16-QAM signal
        modulation_type = analyze_modulation(qam_signal)
        print(f"Detected modulation type: {modulation_type}")

        plot_constellation(qam_signal, f"Generated 16-QAM Signal ({modulation_type})")
        plot_time_domain_signal(qam_signal, "Time-Domain Representation of 16-QAM Signal")

        return
    
    if TEST_OFDM_SIGNAL:
        # Generate a OFDM signal for testing
        num_symbols = 10        # Number of OFDM symbols
        num_subcarriers = 64    # Number of subcarriers
        cp_length = 16          # Length of cyclic prefix
        snr_db = 20             # SNR in dB        
        ofdm_signal = generate_ofdm_signal(num_symbols, num_subcarriers, cp_length, snr_db)
        
        plot_ofdm_signal(ofdm_signal, num_subcarriers, cp_length)

        # Analyze and plot the OFDM signal
        modulation_type = analyze_modulation(ofdm_signal)
        print(f"Detected modulation type: {modulation_type}")

        plot_constellation(ofdm_signal, f"Generated OFDM Signal ({modulation_type})")
        plot_time_domain_signal(ofdm_signal, "Time-Domain Representation of OFDM Signal")

        return

    if not TEST_WITH_FAKE_DATA and HackRF is None:
        print("pyhackrf2 not installed, and we are not using fake data. Exiting.")
        return

    if TEST_WITH_FAKE_DATA:
        # Use random data to simulate scanning
        print(f"[FAKE MODE] Generating random IQ data for {segments} segments...")
        for i in range(segments):
            current_freq = start_freq + i * step_freq
            print(f"Fake scanning at {current_freq / 1e6:.1f} MHz...")
            samples = (np.random.randn(samples_to_read) + 
                       1j * np.random.randn(samples_to_read)).astype(np.complex64)
            IQ_data.append(samples)
    else:
        # Use HackRF for scanning
        hackrf = HackRF()
        print(f"Scanning from {start_freq / 1e6} MHz to {end_freq / 1e6} MHz "
              f"in {step_freq / 1e6} MHz steps...")

        try:
            for current_seg in range(segments):
                current_freq = start_freq + current_seg * step_freq
                hackrf.sample_rate = sample_rate
                hackrf.center_freq = current_freq

                # Set gain
                hackrf.lna_gain = LNA_GAIN
                hackrf.vga_gain = VGA_GAIN

                print(f"Scanning at {current_freq / 1e6:.1f} MHz with LNA Gain: {LNA_GAIN} dB and VGA Gain: {VGA_GAIN} dB...")

                samples = hackrf.read_samples(samples_to_read)
                samples = np.ascontiguousarray(samples, dtype=np.complex64)
                IQ_data.append(samples)
                print(len(IQ_data), samples.shape)

        finally:
            hackrf.close()
            print("HackRF closed.")

    # ==================================================
    # 2. Convert all collected IQ data to Spectrograms
    # ==================================================
    # spec_conv = SpectrogramConverter()

    spectrogram_list = []
    freq_list = []

    for i, raw_iq in enumerate(IQ_data):
        current_freq = start_freq + i * step_freq
        print(f"Converting segment {i+1}/{segments} at {current_freq / 1e6:.1f} MHz")

        # Remove DC LO leakage (zero mean)
        raw_iq -= np.mean(raw_iq)  # Zero-mean adjustment for IQ signal

        iq_arr = np.ascontiguousarray(raw_iq, dtype=np.complex64)

        # spectro_dbm = spec_conv.convert(
        #     bandwidth=step_freq,
        #     sample_rate=sample_rate,
        #     time_duration=sample_time_interval,
        #     iq_arr=iq_arr
        # )
        # spectro_np = spectro_dbm.cpu().numpy()
        # spectrogram_list.append(spectro_np)
        # print(spectro_dbm.shape)
        # freq_list.append(current_freq)
        
        # Determine modulation type
        modulation_type = analyze_modulation(iq_arr)
        print(f"Detected modulation type at {current_freq / 1e6:.1f} MHz: {modulation_type}")

        # Plot constellation diagram
        plot_constellation(iq_arr, f"Constellation Diagram at {current_freq / 1e6:.1f} MHz ({modulation_type})")
        plot_time_domain_signal(iq_arr, "Time-Domain Representation of QPSK Signal")
    # Combine all spectrograms
    final_spectrogram = np.concatenate(spectrogram_list, axis=1)
    print(final_spectrogram.shape)

    T, F = final_spectrogram.shape

    # Frequency axis updated to start_freq to end_freq
    freq_axis_mhz = np.linspace(start_freq / 1e6, end_freq / 1e6, F)

    # Time axis updated range to [0, sample_time_interval]
    time_axis_s = np.linspace(0, sample_time_interval, T)

    # Set plot range
    x_min, x_max = freq_axis_mhz[0], freq_axis_mhz[-1]
    y_min, y_max = time_axis_s[0], time_axis_s[-1]

    # Plot
    plt.figure(figsize=(6, 20))
    plt.title("Concatenated Spectrogram of All Segments")
    plt.imshow(final_spectrogram,
               origin='lower',       # Ensure low frequency and early time at the bottom
               aspect='auto',        # Keep aspect ratio
               cmap='viridis',
               extent=(x_min, x_max, y_min, y_max))
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")

    plt.show()
    plt.savefig("output_spectrogram.png")
    print("Plot saved to output_spectrogram.png")

if __name__ == "__main__":
    main()