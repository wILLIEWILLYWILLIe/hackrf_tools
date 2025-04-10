import numpy as np
import matplotlib.pyplot as plt

def generate_modulated_signal(mod_type, num_symbols=1000, seed=None):
    """
    Generate a random complex baseband signal for the given modulation type.
    Options: 'BPSK', 'QPSK', '16QAM', '64QAM'.

    Args:
        mod_type (str): Modulation type (BPSK, QPSK, 16QAM, 64QAM).
        num_symbols (int): Number of symbols to generate.
        seed (int or None): Optional random seed for repeatability.

    Returns:
        iq_data (np.ndarray): Complex array of shape (num_symbols,) 
                              representing the modulated IQ samples.
    """
    if seed is not None:
        np.random.seed(seed)

    # Number of bits per symbol for each modulation
    if mod_type == 'BPSK':
        M = 2
    elif mod_type == 'QPSK':
        M = 4
    elif mod_type == '16QAM':
        M = 16
    elif mod_type == '64QAM':
        M = 64
    else:
        raise ValueError(f"Unsupported mod_type: {mod_type}")

    k = int(np.log2(M))  # bits per symbol

    # Generate random bits
    bits = np.random.randint(0, 2, size=num_symbols * k)

    # Symbol mapping
    if mod_type == 'BPSK':
        # BPSK: 0 -> -1, 1 -> +1
        # We'll interpret each bit as one symbol
        symbols = 2*bits - 1  # values in [-1, +1]
        iq_data = symbols.astype(np.complex64)

    elif mod_type == 'QPSK':
        # QPSK: 2 bits per symbol => map 00->(1+1j), 01->(1-1j), 11->(-1-1j), 10->(-1+1j)
        bits_reshaped = bits.reshape(-1, 2)
        mapping = {
            (0,0):  1+1j,
            (0,1):  1-1j,
            (1,1): -1-1j,
            (1,0): -1+1j
        }
        iq_list = []
        for b0, b1 in bits_reshaped:
            iq_list.append(mapping[(b0, b1)])
        iq_data = np.array(iq_list, dtype=np.complex64) / np.sqrt(2.0)  # normalized

    elif mod_type in ['16QAM', '64QAM']:
        # For rectangular QAM: 
        #   16QAM => real,imag ∈ {±1, ±3}
        #   64QAM => real,imag ∈ {±1, ±3, ±5, ±7} 
        # We'll do Gray-coded mapping by splitting bits into half for real and half for imag.

        # Define constellation axis levels
        if mod_type == '16QAM':
            levels = np.array([-3, -1, 1, 3])
        else:  # 64QAM
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])

        bits_reshaped = bits.reshape(-1, k)
        # half the bits define the 'I' index, half define the 'Q' index
        half_k = k // 2
        # Real part index from the first half_k bits, imag part index from second half_k
        # Convert each group of bits to a decimal index
        real_index = np.array([int("".join(str(x) for x in row[:half_k]), 2) 
                               for row in bits_reshaped])
        imag_index = np.array([int("".join(str(x) for x in row[half_k:]), 2) 
                               for row in bits_reshaped])
        
        re = levels[real_index]
        im = levels[imag_index]

        # Combine real and imaginary
        iq_data = (re + 1j*im).astype(np.complex64)

        # Normalize average power to ~ 1 (optional step)
        # "Rect" QAM has average power ~ ( (levels^2).mean() * 2 ) for large sets
        # For 16QAM, average symbol energy ~ 10, for 64QAM, ~ 42.
        # We'll do exact normalization based on the sample set here:
        avg_power = np.mean(np.abs(iq_data)**2)
        iq_data /= np.sqrt(avg_power)  # so average power = 1

    else:
        raise ValueError(f"Unknown mod_type: {mod_type}")

    return iq_data

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

def plot_Constellation_Angle(iq_data):
    """Plot a 2x2 subplot with enhanced constellation and additional visualizations."""
    
    # Compute phase angles and magnitudes
    angles = np.angle(iq_data, deg=True)  # Get phase angles in degrees
    angles = (angles + 360) % 360  # Ensure all angles are in [0, 360]
    magnitudes = np.abs(iq_data)  # Compute signal magnitude

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1️⃣ Standard Constellation Plot
    axes[0, 0].scatter(iq_data.real, iq_data.imag, s=5, alpha=0.5)
    axes[0, 0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0, 0].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0, 0].set_xlim(-1.5, 1.5)
    axes[0, 0].set_ylim(-1.5, 1.5)
    axes[0, 0].set_title("IQ Constellation Diagram")
    axes[0, 0].set_xlabel("In-phase (I)")
    axes[0, 0].set_ylabel("Quadrature (Q)")
    axes[0, 0].grid(True)

    # 2️⃣ Density Heatmap Constellation
    histogram = axes[0, 1].hist2d(iq_data.real, iq_data.imag, bins=100, cmap='viridis',
                                  range=[[-1.5, 1.5], [-1.5, 1.5]])
    fig.colorbar(histogram[3], ax=axes[0, 1], label='Density')
    axes[0, 1].set_title("Constellation Density Heatmap")
    axes[0, 1].set_xlabel("In-phase (I)")
    axes[0, 1].set_ylabel("Quadrature (Q)")
    axes[0, 1].grid(True)

    # Add QPSK ideal constellation points
    ideal_angles = np.array([45, 135, 225, 315])
    ideal_points = np.exp(1j * ideal_angles * np.pi / 180)
    axes[0, 1].scatter(ideal_points.real, ideal_points.imag, 
                        color='red', marker='x', s=100, label='Ideal QPSK')
    axes[0, 1].legend()

    # 3️⃣ Phase Angle Histogram
    axes[1, 0].hist(angles, bins=360, range=(0, 360), color='blue', alpha=0.7)
    axes[1, 0].set_title("Phase Angle Distribution")
    axes[1, 0].set_xlabel("Phase Angle (degrees)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True)

    # # 4️⃣ Magnitude (Amplitude) Histogram
    # axes[1, 1].hist(magnitudes, bins=100, color='green', alpha=0.7)
    # axes[1, 1].set_title("Magnitude (Amplitude) Distribution")
    # axes[1, 1].set_xlabel("Magnitude")
    # axes[1, 1].set_ylabel("Count")
    # axes[1, 1].grid(True)
    # 4️⃣ Amplitude vs. Time
    sample_indices = np.arange(len(iq_data))
    axes[1, 1].plot(sample_indices, magnitudes, color='green', alpha=0.7)
    axes[1, 1].set_title("Amplitude vs. Time")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].grid(True)

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()

def analyze_modulation(iq_signal, 
                      phase_tolerance=10, 
                      amplitude_variation_threshold=0.15,
                      min_cluster_percentage=0.05):
    """
    Enhanced modulation analysis function with improved BPSK detection.
    """
    # Normalize the signal first
    iq_signal = normalize_iq(iq_signal)
    
    # Compute phase angles and magnitudes
    angles = np.angle(iq_signal, deg=True)
    angles = (angles + 360) % 360
    magnitudes = np.abs(iq_signal)
    
    # Compute amplitude statistics
    amp_mean = np.mean(magnitudes)
    amp_std = np.std(magnitudes)
    relative_variation = amp_std / amp_mean
    
    # First, check for PSK modulations using phase distribution
    def count_points_near_angles(test_angles, tolerance=15):
        count = 0
        for angle in angles:
            if any(abs((angle - ref + 180) % 360 - 180) < tolerance for ref in test_angles):
                count += 1
        return count / len(angles)
    
    # Test for BPSK
    bpsk_angles = [0, 180]
    bpsk_ratio = count_points_near_angles(bpsk_angles)
    
    # Test for QPSK
    qpsk_angles = [45, 135, 225, 315]
    qpsk_ratio = count_points_near_angles(qpsk_angles)
    
    # If amplitude variation is low, likely PSK
    if relative_variation < amplitude_variation_threshold:
        if bpsk_ratio > 0.8:  # Strict threshold for BPSK
            return "BPSK"
        elif qpsk_ratio > 0.7:  # Slightly more lenient for QPSK
            return "QPSK"
    
    # If not PSK, analyze for QAM
    # Use 2D histogram to find constellation points
    hist, xedges, yedges = np.histogram2d(
        iq_signal.real, 
        iq_signal.imag,
        bins=int(np.sqrt(len(iq_signal)/50)),  # Adaptive bin size
        range=[[-3, 3], [-3, 3]]
    )
    
    # Count significant constellation points
    significant_points = np.sum(hist > len(iq_signal)/(hist.size*2))
    
    if significant_points <= 4:
        # Double-check QPSK
        if qpsk_ratio > 0.6:
            return "QPSK"
    elif significant_points <= 16:
        return "16QAM"
    else:
        return "64QAM"
    
    return "Unknown"

def analyze_qpsk_quality(iq_data):
    """Analyze QPSK signal quality metrics."""
    angles = np.angle(iq_data, deg=True)
    angles = (angles + 360) % 360
    
    # Define ideal QPSK angles
    ideal_angles = np.array([45, 135, 225, 315])
    
    # Find nearest ideal angle for each sample
    angle_errors = []
    for angle in angles:
        error = min(abs(angle - ideal) for ideal in ideal_angles)
        angle_errors.append(error)
    
    evm = np.mean(angle_errors)
    phase_stddev = np.std(angle_errors)
    
    return {
        'EVM (degrees)': evm,
        'Phase StdDev': phase_stddev,
        'Max Error': max(angle_errors)
    }

def main():
    # ------------------------------
    # EXAMPLE: Use synthetic signals
    # ------------------------------
    # for mod in ['BPSK', 'QPSK', '16QAM', '64QAM']:
    for mod in []:

        print(f"\n=== Testing {mod} ===")

        iq_data = generate_modulated_signal(mod_type=mod, num_symbols=100000, seed=359)
        iq_data = normalize_iq(iq_data)
        plot_Constellation_Angle(iq_data)
        detected_mod = analyze_modulation(iq_data)
        print(f"Detected Modulation: {detected_mod}")

    # ------------------------------
    # Or load from file if you prefer
    # ------------------------------
    filename = "SG_Signal/xxx.npy"
    filename = "HackRF_SG/"
    iq_data = load_iq_data(filename)
    iq_data = normalize_iq(iq_data)
    plot_Constellation_Angle(iq_data)
    print("Detected Modulation from file:", analyze_modulation(iq_data))

def analy_from_array(iq_data):
    print(iq_data.shape)
    iq_data = normalize_iq(iq_data)
    plot_Constellation_Angle(iq_data)
    print("Detected Modulation from file:", analyze_modulation(iq_data))


if __name__ == "__main__":
    main()
