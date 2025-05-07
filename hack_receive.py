import numpy as np
import matplotlib.pyplot as plt
import torch

from packages.hack_plot_cv import SpectrogramConverter_CV
from packages.hack_plot_torch import SpectrogramConverter

try:
    from pyhackrf2 import HackRF
except ImportError:
    HackRF = None

try:
    from sa.sm200b import SM200b
    from sa.sa_setting import SA_TRIGGER_TYPE
except:
    SM200B = None

# ================
# PARAMETERS INIT
# ================

USING_DEVICE= [
                'FAKE',
                'HACKRF',
                'SM200B'
                ]
USE_GPU     = True
USE_TORCH   = True
SAVE_NPY    = True
SAVE_NPY_NAME = "xxx"

# Scanning parameters (Hz)
START_FREQ = 5780e6  # scanning start frequency
END_FREQ   = 5800e6  # scanning end frequency
CENTER_FREQ = 5.79e9  # For 200B
TIME_INTERVAL = 15.36e-3*5 # 單次掃描時間 

# Device-specific parameters
HACK_RF_PARAM = {
    'SAMPLE_RATE': 20e6,  # 20 MHz sample rate
    'LNA_GAIN': 5,       # LNA gain (0 ~ 40 dB)
    'VGA_GAIN': 5        # VGA gain (0 ~ 62 dB)
}

SM200B_PARAM = {
    'SAMPLE_RATE': 250e6,  # 250 MHz sample rate
    'REF_LEVEL': 0 ,        # Reference level in dB
    'SERIAL_NUMBER' : 22336138
}

class IQ_Analysis:
    def __init__(self, device_type='HACKRF', use_fake=False):
        self.USING_DEVICE = device_type.upper()

        # Setup spectrogram converter.
        torch_device = torch.device("cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu")
        if USE_TORCH:
            self.spec_conv = SpectrogramConverter(torch_device)
        else:
            self.spec_conv = SpectrogramConverter_CV()

    # @staticmethod
    def hackrf_receive(self):
        IQ_data = []
        hackrf  = HackRF()
        step_freq = HACK_RF_PARAM['SAMPLE_RATE']  

        print(f"Scanning from {START_FREQ/ 1e6} MHz to {END_FREQ / 1e6} MHz "
              f"in {step_freq / 1e6} MHz steps...")
        
        samples_to_read = int(TIME_INTERVAL * HACK_RF_PARAM['SAMPLE_RATE'])  # 每次讀取的樣本數
        segments = int((END_FREQ - START_FREQ) // step_freq)  # 分段數

        try:
            for current_seg in range(segments):
                current_freq = START_FREQ + current_seg * step_freq

                hackrf.sample_rate = HACK_RF_PARAM['SAMPLE_RATE']
                hackrf.center_freq = current_freq

                # 設定增益
                hackrf.lna_gain = HACK_RF_PARAM['LNA_GAIN']
                hackrf.vga_gain = HACK_RF_PARAM['VGA_GAIN']

                print(f"Scanning at {current_freq / 1e6:.1f} MHz with LNA Gain: {HACK_RF_PARAM['LNA_GAIN']} dB and VGA Gain: {HACK_RF_PARAM['VGA_GAIN']} dB...")

                samples = hackrf.read_samples(samples_to_read)
                # samples = np.asarray(samples, dtype=np.complex64).copy()
                samples = np.array(samples, dtype=np.complex64)

                IQ_data.append(samples)

        finally:
            hackrf.close()
            print("HackRF closed.")

        return IQ_data
    
    # @staticmethod
    def sm200b_receive(self):
        IQ_data = []
        device = SM200b(SM200B_PARAM['SERIAL_NUMBER'])

        device.setCenterFreq(CENTER_FREQ)
        device.setRefLevel(SM200B_PARAM['REF_LEVEL'])
        device.setTimeDuration(TIME_INTERVAL)
        device.setTrigger(SA_TRIGGER_TYPE.IMMEDIATE, 0)

        iq_point_num = int(SM200B_PARAM['SAMPLE_RATE']*TIME_INTERVAL)
        iq_arr = np.ndarray(shape=(iq_point_num,), dtype=np.complex64)

        device.startCaptureIq()
        device.waitCaptureIq()
        device.readIq(iq_arr)

        device.close()

        IQ_data.append(iq_arr)

        return IQ_data
    
    def receive_iq(self):
        if self.USING_DEVICE == 'FAKE':
            IQ_data = []
            samples_to_read = int(TIME_INTERVAL * HACK_RF_PARAM['SAMPLE_RATE'])
            segments = int((END_FREQ - START_FREQ) // HACK_RF_PARAM['SAMPLE_RATE'])  # 分段數

            print(f"[FAKE MODE] Generating random IQ data for {segments} segments...")
            for seg in range(segments):
                current_freq = START_FREQ + seg * END_FREQ
                print(f"Fake scanning at {current_freq/1e6:.1f} MHz...")
                samples = (np.random.randn(samples_to_read) +
                           1j * np.random.randn(samples_to_read)).astype(np.complex64)
                IQ_data.append(samples)

        elif self.USING_DEVICE == 'HACKRF' : 
            if HackRF is None:
                print("pyhackrf2 not installed. Exiting.")
                return None
            IQ_data = self.hackrf_receive()
        
        elif self.USING_DEVICE == 'SM200B':
            if SM200b is None:
                print("SM200b module not installed. Exiting.")
                return None
            IQ_data = self.sm200b_receive()
        else:
            print("Unsupported device type. Exiting.")
            return None
        
        return IQ_data
    
    def save_iq_data(self, IQ_data):
        if SAVE_NPY:
            for i, raw_iq in enumerate(IQ_data):
                fc_mhz = int((START_FREQ + i * HACK_RF_PARAM['SAMPLE_RATE']) / 1e6)
                filename = f"HackRF_SG/{SAVE_NPY_NAME}.npy"
                np.save(filename, raw_iq)
                print(f"Saved IQ data to {filename}")
    
    def convert_to_spectrogram(self, IQ_data):
        spectrogram_list = []
        if len(IQ_data) > 0:
            for i, raw_iq in enumerate(IQ_data):
                current_freq = START_FREQ + i * HACK_RF_PARAM['SAMPLE_RATE']
                segments = int((END_FREQ - START_FREQ) // HACK_RF_PARAM['SAMPLE_RATE'])

                print(f"Converting segment {i+1}/{segments} at {current_freq/1e6:.1f} MHz")
                # Remove DC offset (LO leakage).
                raw_iq = raw_iq - np.mean(raw_iq)
                spectro_dbm = self.spec_conv.convert(
                    bandwidth=self.step_freq,
                    sample_rate=self.sample_rate,
                    time_duration=self.time_interval,
                    iq_arr=raw_iq
                )
                spectrogram_list.append(spectro_dbm)
            final_spectrogram = np.concatenate(spectrogram_list, axis=1)
            print(f"Final spectrogram shape: {final_spectrogram.shape}")
            return final_spectrogram
        else:
            pass

    def plot_spectrogram(self, final_spectrogram):
        T, F = final_spectrogram.shape
        freq_axis_mhz = np.linspace(self.start_freq / 1e6, self.end_freq / 1e6, F)
        time_axis_s = np.linspace(0, self.time_interval, T)
        plt.figure(figsize=(6, 18))
        plt.title("Concatenated Spectrogram of All Segments")
        plt.imshow(final_spectrogram,
                   origin='lower',
                   aspect='auto',
                   cmap='gray',
                   extent=(freq_axis_mhz[0], freq_axis_mhz[-1], time_axis_s[0], time_axis_s[-1]))
        plt.colorbar(label="Power (dB)")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (s)")
        plt.savefig("output_spectrogram.png")
        print("Plot saved to output_spectrogram.png")
        plt.close()

    def run(self):
        IQ_data = self.receive_iq()
        if IQ_data is None:
            print("No IQ data acquired. Exiting.")
            return

        self.save_iq_data(IQ_data)
        if self.USING_DEVICE == 'HACKRF':
            # final_spectrogram = self.convert_to_spectrogram(IQ_data)
            # self.plot_spectrogram(final_spectrogram)
            pass
        return IQ_data


# =======================
# Main Entry Point
# =======================
def main():
    # Choose device type: 'FAKE', 'HACKRF', or 'SM200B'
    # analysis = IQ_Analysis(device_type="HACKRF", use_fake=False)
    analysis = IQ_Analysis(device_type="SM200B", use_fake=False)
    IQ_data = analysis.run()
    print(len(IQ_data))



if __name__ == "__main__":
    main()