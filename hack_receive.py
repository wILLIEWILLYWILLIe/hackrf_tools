import numpy as np
import matplotlib.pyplot as plt
import torch
from packages.hack_plot_cv import SpectrogramConverter_CV
from packages.hack_plot_torch import SpectrogramConverter

try:
    # Only import HackRF if needed
    from pyhackrf2 import HackRF
except ImportError:
    HackRF = None

def main():
    TEST_WITH_FAKE_DATA = False
    USE_GPU = False
    USE_TORCH = True

    device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')
    if USE_TORCH:
        spec_conv = SpectrogramConverter(device)
    else:
        spec_conv = SpectrogramConverter_CV()

    # 掃描參數設定
    sample_rate = 20e6  # 取樣率 (20 MHz)
    start_freq = 2400e6  # 掃描起始頻率 (2400 MHz)
    end_freq = 2500e6  # 掃描結束頻率 (2500 MHz)
    step_freq = sample_rate  # 每次掃描的頻寬 (20 MHz)
    sample_time_interval = 15.36e-3  # 單次掃描時間 
    samples_to_read = int(sample_time_interval * sample_rate)  # 每次讀取的樣本數
    segments = int((end_freq - start_freq) // step_freq)  # 分段數

    # 增益參數設定
    LNA_GAIN = 20  # LNA 增益（範圍：0 ~ 40）
    VGA_GAIN = 10  # VGA 增益（範圍：0 ~ 62）

    # 儲存掃描結果
    IQ_data = []

    if not TEST_WITH_FAKE_DATA and HackRF is None:
        print("pyhackrf2 not installed, and we are not using fake data. Exiting.")
        return

    if TEST_WITH_FAKE_DATA:
        # 使用隨機數據模擬掃描
        print(f"[FAKE MODE] Generating random IQ data for {segments} segments...")
        for i in range(segments):
            current_freq = start_freq + i * step_freq
            print(f"Fake scanning at {current_freq / 1e6:.1f} MHz...")
            samples = (np.random.randn(samples_to_read) + 
                       1j * np.random.randn(samples_to_read)).astype(np.complex64)
            IQ_data.append(samples)
    else:
        # 使用 HackRF 進行掃描
        hackrf = HackRF()
        print(f"Scanning from {start_freq / 1e6} MHz to {end_freq / 1e6} MHz "
              f"in {step_freq / 1e6} MHz steps...")

        try:
            for current_seg in range(segments):
                current_freq = start_freq + current_seg * step_freq

                hackrf.sample_rate = sample_rate
                hackrf.center_freq = current_freq

                # 設定增益
                hackrf.lna_gain = LNA_GAIN
                hackrf.vga_gain = VGA_GAIN

                print(f"Scanning at {current_freq / 1e6:.1f} MHz with LNA Gain: {LNA_GAIN} dB and VGA Gain: {VGA_GAIN} dB...")

                samples = hackrf.read_samples(samples_to_read)
                samples = np.asarray(samples, dtype=np.complex64)

                IQ_data.append(samples)

        finally:
            hackrf.close()
            print("HackRF closed.")

    # ==================================================
    # 2. Convert all collected IQ data to Spectrograms
    # ==================================================

    spectrogram_list = []
    freq_list = []

    for i, raw_iq in enumerate(IQ_data):
        current_freq = start_freq + i * step_freq
        print(f"Converting segment {i+1}/{segments} at {current_freq / 1e6:.1f} MHz")

        # 去除 DC LO leakage (零均值化)
        raw_iq -= np.mean(raw_iq)  # 將 IQ 信號的均值調整為零
        iq_arr = raw_iq
        # iq_arr = np.ascontiguousarray(raw_iq, dtype=np.complex64)

        spectro_dbm = spec_conv.convert(
            bandwidth=step_freq,
            sample_rate=sample_rate,
            time_duration=sample_time_interval,
            iq_arr=iq_arr
        )
        spectrogram_list.append(spectro_dbm)
        freq_list.append(current_freq)

    # 合併所有頻譜圖
    final_spectrogram = np.concatenate(spectrogram_list, axis=1)
    print(f'final shape {final_spectrogram.shape}, total IQ length {segments * IQ_data[0].shape[0]}')

    T, F = final_spectrogram.shape

    # 頻率軸更新為 start_freq 到 end_freq
    freq_axis_mhz = np.linspace(start_freq / 1e6, end_freq / 1e6, F)

    # 時間軸更新範圍為 [0, sample_time_interval]
    time_axis_s = np.linspace(0, sample_time_interval, T)

    # 設定繪圖範圍
    x_min, x_max = freq_axis_mhz[0], freq_axis_mhz[-1]
    y_min, y_max = time_axis_s[0], time_axis_s[-1]

    # 畫圖
    plt.figure(figsize=(6, 18))
    plt.title("Concatenated Spectrogram of All Segments")
    plt.imshow(final_spectrogram,
               origin='lower',       # 確保低頻和早期時間在下方
               aspect='auto',        # 保持比例不失真
               cmap='gray',
               extent=(x_min, x_max, y_min, y_max))
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")

    # plt.show()
    plt.savefig("output_spectrogram.png")
    print("Plot saved to output_spectrogram.png")

if __name__ == "__main__":
    main()