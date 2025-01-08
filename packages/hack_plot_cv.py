from pathlib import Path
import json
# from dsp_algorithm import findTotalpairs, find_group_pairs2, get_img_from_spectro, show_total_pairs2
import numpy as np
import glob
import math
import cv2
import os
from scipy.signal import spectrogram, get_window
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import freeze_support
from PIL import Image

def get_img_from_spectro(spectro, plot_q = 5, dpi = 100):

	r'''
	Convert spectrogram to image array.

	plot_q : downsample_rate.
	'''

	spectro = 10*np.log10(spectro[::plot_q,::plot_q])

	im = Image.fromarray((spectro-spectro.min())/(spectro.max()-spectro.min())*255)

	return np.array(im).astype(np.uint8).T.copy()


class SpectrogramConverter_CV():
    def __init__(self, dir='', select: bool=False):
        self.__replay_dir = dir
        self.__select = select
        self.__select_path = self.__replay_dir+"\\selected"
        if (self.__select):
            Path(self.__select_path).mkdir(parents=True, exist_ok=True)
            print(f"Select mode is enabled, selected data will be saved at {self.__select_path}.")
            print("Press ENTER to save or any other keys to discard.")

        self.__bandwidth = None
        self.__sample_rate = None
        self.__time_duration = None

    def set_params(self, bandwidth, sample_rate, time_duration):
        """ Set parameters for TFA and signal processing. """
        self.__bandwidth = bandwidth
        self.__sample_rate = sample_rate
        self.__time_duration = time_duration
        self.initialize_tfa_settings()

    def initialize_tfa_settings(self):
        """ Initialize settings related to Time-Frequency Analysis. """
        self.TFA_WINDOW_SIZE = 1500
        self.TFA_ALPHA = 10
        self.TFA_OVERLAP = 0.3
        self.SIGMA = 5
        self.window = get_window(('kaiser', self.SIGMA), self.TFA_WINDOW_SIZE)
        self.START_SPECTRO_F = math.ceil(0.5*(1-self.__bandwidth/self.__sample_rate)*(self.TFA_WINDOW_SIZE))
        self.STOP_SPECTRO_F = self.TFA_WINDOW_SIZE - self.START_SPECTRO_F
        self.START_SPECTRO_T = 0
        self.STOP_SPECTRO_T = math.floor((int(self.__time_duration*self.__sample_rate * 1e-3) - self.TFA_WINDOW_SIZE) /
                                         (self.TFA_WINDOW_SIZE*(1-self.TFA_OVERLAP)))+1
        self.SPECTRO_SHAPE = (self.STOP_SPECTRO_F-self.START_SPECTRO_F, self.STOP_SPECTRO_T)
        self.DOWN_SAMPLING_RATE = 5

        # self.findTotalpairs = findTotalpairs(self.SPECTRO_SHAPE, self.__bandwidth, int(
        #     self.__time_duration*self.__sample_rate * 1e-3), self.__sample_rate, self.DOWN_SAMPLING_RATE)
        # self.find_group_pairs2 = find_group_pairs2(self.SPECTRO_SHAPE, self.__bandwidth, int(
        #     self.__time_duration*self.__sample_rate * 1e-3), self.__sample_rate)

    def toSpectrogram(self, iq):
        f, t, spectro = spectrogram(iq,
                                    fs=self.__sample_rate,
                                    window=self.window,
                                    noverlap=self.TFA_OVERLAP*self.TFA_WINDOW_SIZE,
                                    return_onesided=False)
        spectro = abs(np.fft.fftshift(spectro[1:], axes=0))
        spectro = spectro[self.START_SPECTRO_F:self.STOP_SPECTRO_F, :]
        return spectro

    def normalizeImg(self, img):
        img_min = np.min(img)
        img_max = np.max(img)
        img = (img-img_min)/(img_max-img_min)

    def process_file(self, file_info):
        replay_dir, file_name = file_info
        with open(os.path.join(replay_dir, file_name), 'r') as f:
            data = json.load(f)

        for serial_number in data['serial_numbers']:
            self.set_params(bandwidth=data["sa"][serial_number]["bandwidth"],
                            sample_rate=data["sa"][serial_number]["sample_rate"],
                            time_duration=data["sa"][serial_number]["time_duration"])
            iq = np.load(f"{replay_dir}/{data['timestamp']}_{serial_number}.npy")
            spectro = self.toSpectrogram(iq)
            img = get_img_from_spectro(spectro=spectro, plot_q=4)
            self.normalizeImg(img)
            # rec_img = self.find_signal(spectro)
            # cv2.imwrite(f"{replay_dir}/{data['timestamp']}_{serial_number}_rec.png", rec_img)

            if (self.__select):
                cv2.imshow(f"{self.__select_path}/{data['timestamp']}_{serial_number}.npy", img)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if (key == 13):  # Enter
                    print("save", f"{self.__select_path}/{data['timestamp']}_{serial_number}.npy")
                    cv2.imwrite(f"{self.__select_path}/{data['timestamp']}_{serial_number}.png", img)
                    np.save(f"{self.__select_path}/{data['timestamp']}_{serial_number}.npy", iq)
                    with open(f"{self.__select_path}/{data['timestamp']}_{serial_number}_data.json", 'w') as selected_f:
                        json.dump(data, selected_f)

            else:
                cv2.imwrite(f"{replay_dir}/{data['timestamp']}_{serial_number}.png", img)

    def convert(self, bandwidth,sample_rate,time_duration, iq_arr):
        self.set_params(bandwidth,sample_rate, time_duration)
        spectro = self.toSpectrogram(iq_arr)
        img = get_img_from_spectro(spectro=spectro, plot_q=1)
        self.normalizeImg(img)
        # rec_img = self.find_signal(spectro)
        # cv2.imwrite(f"{replay_dir}/{data['timestamp']}_{serial_number}_rec.png", rec_img)

        # cv2.imwrite(f"spectrogram.png", img)
        return img

    # def find_signal(self, raw_spectro):
    #     findTotalpairs_Result = self.findTotalpairs.run(raw_spectro.T)
    #     if len(findTotalpairs_Result) == 2:
    #         _total_pairs, rough_NL = findTotalpairs_Result
    #     else:
    #         _total_pairs, POI_tot_pairs, rough_NL = findTotalpairs_Result
    #     plot_name = ''
        # rec_img = show_total_pairs2(raw_spectro, _total_pairs, '_total_pairs', block_ind=0, print_info=0, plot_q=4, NL=rough_NL, print_range=['500', '800'], put_text=plot_name)
        # rec_img2 = show_total_pairs2(raw_spectro, POI_tot_pairs, 'POI_total_pairs', block_ind=0, print_info=0, plot_q=4, NL=rough_NL, print_range=['500', '800'], put_text=plot_name)
        # return rec_img

    def read_dir(self):
        file_names = glob.glob(f"{self.__replay_dir}/*_data.json")
        file_info = [(self.__replay_dir, os.path.basename(file_name)) for file_name in file_names]
        if (self.__select):
            process_num = 1
        else:
            process_num = os.cpu_count()
        with Pool(process_num) as pool:
            for _ in tqdm(pool.imap(self.process_file, file_info), total=len(file_info), desc="Processing files"):
                pass


def main():
    # parser = argparse.ArgumentParser(description='Plot replay spectrogram.')
    # parser.add_argument('--replay_dir', default='C:\\Users\\Frank Chen\\Desktop\\5_20M_new', type=str, help='Replay Directory.')
    # parser.add_argument('--select', default=False, type=bool, help='Enable selecting useful IQ by pressing ENTER or DELETE. Images shows one-by-one.')
    # args = parser.parse_args()

    replay_dir = input("Data directory: ")
    select = True if input("Select mode? (y/n): ") == 'y' else False

    plot_replay = replay_helper(dir=replay_dir, select=select)
    plot_replay.read_dir()


if __name__ == "__main__":
    freeze_support()
    main()
    os.system("pause")
