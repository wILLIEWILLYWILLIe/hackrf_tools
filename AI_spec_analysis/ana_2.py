import numpy as np
import matplotlib.pyplot as plt


def load_iq_data(filename):
    """
    Load IQ samples from a .npy file.
    """
    iq_data = np.load(filename)
    # Flatten in case of multiple segments
    return iq_data.flatten()


def preprocess_iq(iq_data):
    """
    Preprocess IQ data:
    1) Remove DC offset.
    2) Normalize average power to 1.
    3) Remove global phase offset.
    
    Returns:
        iq_data_processed (np.ndarray): The preprocessed complex IQ data.
    """
    # 1) Remove DC offset
    iq_data = iq_data - np.mean(iq_data)
    
    # 2) Normalize average power
    power = np.mean(np.abs(iq_data)**2)
    if power > 0:
        iq_data = iq_data / np.sqrt(power)
    
    # 3) Remove global phase offset (rotate so that average IQ angle ~ 0)
    avg_angle = np.angle(np.mean(iq_data))
    iq_data = iq_data * np.exp(-1j * avg_angle)
    
    return iq_data


def plot_enhanced_constellation(iq_data):
    """
    Plot a 2x2 subplot with enhanced constellation and additional visualizations.
    """
    # Compute phase angles and magnitudes
    angles = np.angle(iq_data, deg=True)  # phase in degrees
    angles = (angles + 360) % 360
    magnitudes = np.abs(iq_data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Standard Constellation Plot
    axes[0, 0].scatter(iq_data.real, iq_data.imag, s=5, alpha=0.5)
    axes[0, 0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0, 0].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0, 0].set_xlim(-2, 2)
    axes[0, 0].set_ylim(-2, 2)
    axes[0, 0].set_title("IQ Constellation Diagram")
    axes[0, 0].set_xlabel("In-phase (I)")
    axes[0, 0].set_ylabel("Quadrature (Q)")
    axes[0, 0].grid(True)

    # 2) Density Heatmap
    hist_data = axes[0, 1].hist2d(iq_data.real, iq_data.imag, 
                                  bins=100, cmap='viridis',
                                  range=[[-2, 2], [-2, 2]])
    plt.colorbar(hist_data[3], ax=axes[0, 1], label='Density')
    axes[0, 1].set_title("Constellation Density Heatmap")
    axes[0, 1].set_xlabel("In-phase (I)")
    axes[0, 1].set_ylabel("Quadrature (Q)")
    axes[0, 1].grid(True)

    # 3) Phase Angle Histogram
    axes[1, 0].hist(angles, bins=360, range=(0, 360), color='blue', alpha=0.7)
    axes[1, 0].set_title("Phase Angle Distribution")
    axes[1, 0].set_xlabel("Phase Angle (degrees)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True)

    # 4) Magnitude Histogram
    axes[1, 1].hist(magnitudes, bins=100, color='green', alpha=0.7)
    axes[1, 1].set_title("Magnitude Distribution")
    axes[1, 1].set_xlabel("Magnitude")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def get_reference_constellation(mod_type):
    """
    Return the ideal (normalized) constellation points (complex128) for a given modulation type:
    BPSK, QPSK, 16QAM, 64QAM, etc.
    
    Note: These are often scaled so that average power = 1. 
    However, we will search over a scale factor later, 
    so here we just define the base 'shape.'
    """
    if mod_type == 'BPSK':
        # ±1
        return np.array([-1+0j, 1+0j], dtype=np.complex64)
    
    elif mod_type == 'QPSK':
        # 4 points at ±1/√2 ± j/√2
        return np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    elif mod_type == '16QAM':
        # 4x4 grid: real,imag ∈ {±1, ±3} (average power 10 if unscaled)
        re = np.array([-3, -1, 1, 3])
        q = np.array([-3, -1, 1, 3])
        points = []
        for r in re:
            for i in q:
                points.append(r + 1j*i)
        return np.array(points)  # unscaled base
    
    elif mod_type == '64QAM':
        # 8x8 grid: real,imag ∈ {±1, ±3, ±5, ±7}
        re = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        q = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        points = []
        for r in re:
            for i in q:
                points.append(r + 1j*i)
        return np.array(points)
    
    else:
        return None


def average_min_distance(iq_data, ref_points):
    """
    Compute the average minimum distance from each sample in iq_data
    to the nearest point in ref_points (which should already be scaled).
    """
    # For efficient nearest-point search, you could use a KD-Tree in scipy,
    # but for simplicity we just do a broadcasted distance check.
    # shape: (len(iq_data), len(ref_points))
    dists = np.abs(iq_data[:, None] - ref_points[None, :])
    # For each sample, find the minimum distance to any ref point
    min_dists = np.min(dists, axis=1)
    return np.mean(min_dists)


def find_best_scale(iq_data, ref_points, scale_min=0.1, scale_max=2.0, scale_step=0.01):
    """
    Try different scale factors to fit ref_points to iq_data.
    Return the (best_scale, best_dist) that yields the smallest average_min_distance.
    """
    scales = np.arange(scale_min, scale_max + scale_step, scale_step)
    best_scale = None
    best_dist = float('inf')
    
    for s in scales:
        # Scale the reference constellation
        scaled_ref = s * ref_points
        dist = average_min_distance(iq_data, scaled_ref)
        if dist < best_dist:
            best_dist = dist
            best_scale = s
    return best_scale, best_dist


def classify_modulation(iq_data, candidates=('BPSK', 'QPSK', '16QAM', '64QAM'), 
                        unknown_threshold=0.5):
    """
    Classify the modulation by comparing the I/Q data to each candidate's
    ideal reference constellation (after searching for the best scale factor).
    
    unknown_threshold : float
        If the best average distance is still higher than this threshold,
        we label the result "Unknown." Tweak for your data/noise levels.
    
    Returns:
        detected_mod (str): One of the candidate mod types or 'Unknown'.
    """
    best_mod = 'Unknown'
    best_dist = float('inf')
    
    for mod_type in candidates:
        ref_points = get_reference_constellation(mod_type)
        if ref_points is None:
            continue
        
        # Find best scale
        scale, dist = find_best_scale(iq_data, ref_points)
        
        # Keep track of whichever is smallest
        if dist < best_dist:
            best_dist = dist
            best_mod = mod_type
    
    # Finally, check if the best_dist is reasonably small
    if best_dist > unknown_threshold:
        return "Unknown"
    else:
        return best_mod


def main():
    filename = "QPSK_20M_5790MHz.npy"  # example filename
    # filename = "64qam_5780MHz.npy"
    
    # 1) Load and preprocess the signal
    iq_data = load_iq_data(filename)
    iq_data = preprocess_iq(iq_data)
    
    # 2) Plot the constellation and distributions
    plot_enhanced_constellation(iq_data)
    
    # 3) Classify the modulation
    detected_mod = classify_modulation(iq_data)
    print(f"Detected Modulation: {detected_mod}")


if __name__ == "__main__":
    main()
