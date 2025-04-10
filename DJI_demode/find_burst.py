import numpy as np
import scipy.fft as fft
import subprocess
# from paho.mqtt.client import Client
import paho.mqtt.client as mqtt
from framepg import framepg
from mathematic_func import *
import matplotlib.pyplot as plt
import os, sys
import threading
import queue
import scipy.signal as signal

# import torch
# torch.set_printoptions(profile="full")
# np.set_printoptions(threshold=np.inf)

enable_equalizer = 1
correlationscore = 0.1
# slice_1000 = 0


cwd = os.getcwd()
bitdata = os.path.join(cwd,"bits")
remove_turbo_cwd = os.path.join(cwd,"cpp","remove_turbo")
words = ["sudo", remove_turbo_cwd, bitdata]
command = " ".join(words)
# bitdata= "/home/iwave/PIN/DJI_demode/bits"
# command = "sudo /home/iwave/PIN/DJI_demode/cpp/remove_turbo /home/iwave/PIN/DJI_demode/bits"

def on_connect(client, userdata, flags, rc):        #for client
  print("Connected with result code %s."%str(rc))


def freq_offset(burst,precal_func_config,padding,k):
    fft_size=1024
    interp_rate=1
    offset_burst_len = np.sum(precal_func_config["cyclic_prefix_length_schedule"][0:3]) + (fft_size * 3)+ padding
    zc_samples = burst[offset_burst_len:offset_burst_len + 1024]
    fft_bins = np.abs(np.fft.fftshift(np.fft.fft(zc_samples))**2)
    # plt.plot(fft_bins)
    # plt.show()
    bin_count = 15
    fft_bins[0:int((fft_size * interp_rate / 2) - bin_count)] = 100000000
    fft_bins[int((fft_size * interp_rate / 2) + bin_count - 1):] = 100000000
    center_offset = np.argmin(fft_bins)
    # print(center_offset)
    if k == 512:
        center_offset = k
    # print(center_offset)
    integer_offset = ((1024 / 2) - center_offset ) * 15e3
    radians = 2 * np.pi * (integer_offset) / (15.36e6 )
    burst = burst * np.exp(1j * radians * np.arange(len(burst)))
    return burst

def cfo_freq_offset(burst, start_timestamp, precal_func_config, offset_radians,fs,mul,i,qupload, hackrf_num, worker_idx, processcut1000_idx):
    FFT_SIZE=1024
    offset_hz = offset_radians * fs / (2 * np.pi) + mul * 20 -500
    offset_radians1 = offset_hz / fs * (2 * np.pi)
    # print(offset_radians1,"           ",mul)
    # print(offset_hz,"           ",mul)
    burst = np.multiply(burst,np.exp(1j*(-offset_radians1)*np.arange(1,len(burst)+1)))
    #####OFDM#####
    freq_domain = np.zeros((len(precal_func_config["cyclic_prefix_length_schedule"]),FFT_SIZE),dtype=complex)
    time_domain = np.zeros((len(precal_func_config["cyclic_prefix_length_schedule"]),FFT_SIZE),dtype=complex)
    
    sample_offset = 0
    for idx in range(len(precal_func_config["cyclic_prefix_length_schedule"])):
        symbol = burst[sample_offset:sample_offset + FFT_SIZE + precal_func_config["cyclic_prefix_length_schedule"][idx]]
        symbol1 = symbol[precal_func_config["cyclic_prefix_length_schedule"][idx]:len(symbol)]
        time_domain[idx,:] = symbol1
        freq_domain[idx,:] = np.fft.fftshift(np.fft.fft(time_domain[idx,:]))
        sample_offset = sample_offset + FFT_SIZE + precal_func_config["cyclic_prefix_length_schedule"][idx]
    
    gold_seq4 = np.fft.fftshift(np.fft.fft(np.reshape(zcsequence(FFT_SIZE,600),np.shape(freq_domain[3,:]))))        #index = 4, root = 600
    # gold_seq6 = np.fft.fftshift(np.fft.fft(np.reshape(zcsequence(fft_size,147),np.shape(freq_domain[5,:]))))      #index = 6, root = 147
    channel1 = gold_seq4 / freq_domain[3,:]
    channel2 = gold_seq4 / freq_domain[5,:]
    channel1 = channel1[precal_func_config["data_carrier_indices"]]
    channel2 = channel2[precal_func_config["data_carrier_indices"]]
    channel = channel1
    bits = np.zeros((9,1200))
    for idx in range(0,len(bits)):
        data_carriers = freq_domain[idx,precal_func_config["data_carrier_indices"]]
        if enable_equalizer == 1:
            data_carriers = np.multiply(data_carriers,channel)
        bits[idx,:] = quantize_qpsk(data_carriers,idx)
    bits = np.concatenate((bits[1], bits[2], bits[4], bits[6], bits[7], bits[8]))
    bits = np.asarray(bits,dtype = 'int')
    bits = bits ^ precal_func_config["second_scrambler"]
    bits = np.asarray(bits, dtype = 'byte')
    timestamp = int(start_timestamp.timestamp())
    tmpbitdata = os.path.join(cwd,"tmp_bits",f"bits{mul}_{processcut1000_idx%200}_{timestamp%5}_{hackrf_num[-14:]}")
    f = open(tmpbitdata, "wb")
    f.write(bits)
    f.close()
    if not qupload.qsize() > 0:
        words = ["sudo", remove_turbo_cwd, tmpbitdata]
        command = " ".join(words)
        try:
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, check=True, text=True)
            returned_text = result.stdout
            # returned_text = returned_text.decode("utf-8")
            returned_text = returned_text.strip()
            returned_text = returned_text.lstrip("0")
            if returned_text[0:7] != '[ERROR]':
                if not qupload.qsize()>0:
                    qupload.put(returned_text)
            # # 这里可以使用 returned_text，而不在终端上显示输出
            # print(returned_text)
        except subprocess.CalledProcessError as e:
            returned_text="CRC"
            # print(f"Command failed with return code {e.returncode}")
            # print(e.stderr.decode("utf-8"))

def center_freq_offset(burst,start_timestamp, precal_func_config,fs,i,qupload, hackrf_num, worker_idx, processcut1000_idx):
    FFT_SIZE = int(fs/15e3)
    num_ofdm_symbols = len(precal_func_config["cyclic_prefix_length_schedule"])
    full_burst_len = np.sum(precal_func_config["cyclic_prefix_length_schedule"]) + (FFT_SIZE * num_ofdm_symbols)
    num_tests = len(burst) - full_burst_len
    padding=50
    k=512-0+i
    burst = freq_offset(burst,precal_func_config,padding,k)
    scores_cp_sto = np.zeros(num_tests)
    for idx in range(num_tests):
        offset = idx
        scores1 = np.zeros(num_ofdm_symbols)
        for cp_idx in range(num_ofdm_symbols):
            cp_len = precal_func_config["cyclic_prefix_length_schedule"][cp_idx]
            window = burst[offset:offset + FFT_SIZE + cp_len]
            left = window[0:cp_len]
            right = window[len(window) - cp_len:len(window)]
            #Correlate the two windows
            scores1[cp_idx] = np.abs(np.correlate(left, right))
            # Move the sample pointer forward by the full symbol size
            offset = offset + cp_len + FFT_SIZE
        scores_cp_sto[idx] = np.sum(scores1[1:len(window)]) / (len(scores1) - 1)
    testburst=burst
    true_start_index=np.argmax(scores_cp_sto)
    # print(true_start_index)
    burst=testburst[true_start_index:len(testburst)]
    
    cfo_est_symbol = burst[precal_func_config["zc_seq_offset"]-precal_func_config["short_cp_len"]-1 : precal_func_config["zc_seq_offset"]+FFT_SIZE-1]
    cyclic_prefix = cfo_est_symbol[0:precal_func_config["short_cp_len"]]
    symbol_tail = cfo_est_symbol[len(cfo_est_symbol) - precal_func_config["short_cp_len"] : len(cfo_est_symbol)]
    offset_radians = np.angle(np.dot(np.conjugate(cyclic_prefix),symbol_tail)) / FFT_SIZE
    testburst=burst
    # offset_radians1=offset_radians
    process_list_cut=[]
    for mul in range(50):
        process_list_cut.append(threading.Thread(target=cfo_freq_offset, args=(burst,start_timestamp, precal_func_config, offset_radians,
                                                                               fs,mul,i,qupload, hackrf_num, worker_idx, processcut1000_idx)))
    for mul in range(50):
        process_list_cut[mul].start()
    for mul in range(50): 
        process_list_cut[mul].join()

def solve_burst(burst,start_timestamp, precal_func_config, hackrf_num, worker_idx, processcut1000_idx, fs = 15.36e6):
    timestamp = start_timestamp.timestamp()
    qupload = queue.Queue()
    center_freq_offset(burst,start_timestamp, precal_func_config, fs, 0, qupload, hackrf_num, worker_idx, processcut1000_idx)
    if qupload.empty():
        center_freq_offset(burst,start_timestamp, precal_func_config, fs, 1, qupload, hackrf_num, worker_idx, processcut1000_idx)

    if qupload.empty():
        print("="*20, "burst get CRC", "="*20)
    
    while not qupload.empty():
        returned_text = qupload.get()
        # qupload.queue.clear()
        print(returned_text)
        upload_data = framepg(returned_text,timestamp)
        print("upload_data=",upload_data)
        print(f"solved by hackrf {hackrf_num}")
        #////////////////////////////////////////
        if upload_data != []:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            ip = "140.112.45.233"
            # ip = "140.112.19.87"
            port = 1883
            client.on_connect = on_connect
            client.connect(ip, port, 60)
            client.loop_start()
            for data_to_upload in upload_data:
                print(data_to_upload)
                client.publish("/iShield-Demod/1", data_to_upload, qos=0)
        #////////////////////////////////////////




def find_burst_sol4(processcut1000_idx , samples, start_timestamp, precal_func_config, numprocesscut, hackrf_num, worker_idx, fs = 15.36e6, td = 0.35):
    FFT_SIZE = int(fs/15e3)
    # if slice_1000:
    #     scores_pre = []
    #     _,tt,zxxx=signal.stft(samples,fs=fs,nperseg=fs*td/1000,noverlap=0, return_onesided=False)
    #     for i in range(len(tt)-1):
    #         scores_pre.append(slice7_LO(np.absolute(fft.fftshift(zxxx[:,i]))))
    #     max_where_maybe = scores_pre.index(np.amax(scores_pre[1:]))
    #     print("max_where_maybe = ",max_where_maybe)
    #     if max_where_maybe >= 2 and max_where_maybe <= int(len(samples)/1000)-3:
    #         true_siganl_tmp = samples[int(len(samples)/1000)*(max_where_maybe-2) : int(len(samples)/1000)*(max_where_maybe+2)]
    #     elif max_where_maybe <= 1:
    #         true_siganl_tmp = samples[0 : int(len(samples)/1000)*(max_where_maybe+2)]
    #     else:
    #         true_siganl_tmp = samples[int(len(samples)/1000)*(max_where_maybe-2) : int(len(samples)/1000)*(max_where_maybe)]
    # else:
    #     true_siganl_tmp = samples
    true_siganl_tmp = samples[int(processcut1000_idx*len(samples)/numprocesscut):int((processcut1000_idx+1)*len(samples)/numprocesscut)]
    sliced_sample_len = len(true_siganl_tmp)
    sliced_sample = true_siganl_tmp[0:sliced_sample_len]
    # root = 1
    # root = 301
    root = 600
    try:
        transform_filter = precal_func_config["transform_filter"][f"{root}_{sliced_sample_len}_{FFT_SIZE}"]
    except:
        transform_filter = precalculate_transform_filter(root, sliced_sample_len, FFT_SIZE)
        precal_func_config["transform_filter"][f"{root}_{sliced_sample_len}_{FFT_SIZE}"] = transform_filter

    zcfilter_conj_fft = transform_filter["Filter_T"]
    sliced_sample_fft = fft.fft(sliced_sample)
    zc_sample_f_mul = zcfilter_conj_fft * sliced_sample_fft
    zc_sample_t_conv = fft.ifft(zc_sample_f_mul)
    IFFTdatacut=zc_sample_t_conv[1024:len(zc_sample_t_conv)]

    CS = np.cumsum(np.abs(true_siganl_tmp)**2)
    cumsum_front = CS[0 : len(CS)-FFT_SIZE]
    cumsum_back = CS[FFT_SIZE : len(CS)]
    CSK=np.sqrt(cumsum_back-cumsum_front+1)
    tmpscore1=np.absolute(IFFTdatacut/CSK)**2

    # plt.plot(tmpscore1)
    # plot.show()

    fasti=np.argmax(tmpscore1)
    #print(fasti)

    window_fast = true_siganl_tmp[fasti:fasti + FFT_SIZE]
    running_sum_fast = np.sum(window_fast)
    window1_fast = window_fast - (running_sum_fast / FFT_SIZE)
    prod_fast = np.sum(window1_fast * transform_filter["zcfilter_conj"]) / FFT_SIZE
    variance = np.var(window1_fast)
    fast_scores = prod_fast /(np.sqrt(variance) * transform_filter["zcfilter_conj_var_sqrt"])
    ##############################################################################################################
    #zcfilter_conj=transform_filter["zcfilter_conj"]
    #zcfilter_conj_var_sqrt=transform_filter["zcfilter_conj_var_sqrt"]
    #window_size = len(zcfilter_conj)
    #recip_window_size = 1/window_size
    #scorestemp=np.zeros(1024,dtype=complex)
    #idx=0
    #for idx in range(1024):
    #     if fasti<512:
    #         continue
    #     if len(true_siganl_tmp)-fasti<500:
    #         continue
    #     window_fast = true_siganl_tmp[fasti+idx-512:fasti+idx-512 + window_size]
    #     running_sum_fast = np.sum(window_fast)
    #     window1_fast = window_fast - (running_sum_fast * recip_window_size)
    #     prod_fast = np.sum(window1_fast * zcfilter_conj) * recip_window_size
    #     variance = np.var(window1_fast)
    #     scorestemp[idx] = prod_fast /(np.sqrt(variance) * zcfilter_conj_var_sqrt)

    # abs_scores = np.abs(scorestemp)**2
    # # plt.plot(tmpscore1)
    # # plt.show()
    # fasti3=np.argmax(abs_scores)
    # # print(fasti3)
    # # # print(fasti+fasti3-512)
    # # # # print(abs_scores)
    # # # print(np.max(abs_scores))

    # if abs_scores[fasti3] > correlationscore:
    #     print("AAAAAAAAAAAAAAAAAAAAA",fasti3)   
    # #     print(abs_scores[fasti3])
    # #     startpoint=fasti+fasti3-512
    # #     findsignal=1
    # #     print(i)
    # #     print(startpoint)
    # #     plt.plot(abs_scores)
    # #     plt.show()
    # #     plt.plot(np.absolute(tmpscore1)**2)
    # #     plt.show()
    # ###############################################################################################################
    # print(fast_scores)
    # passing_scores=-1
    padding = 50
    if np.absolute(fast_scores)**2 > correlationscore:      #判斷1/1000的samples中correlation有無高過設定的score，倘若有則繼續解碼，反之跳出直接進入下一loop
        # I = samples.real
        # Q = samples.imag
        # I.astype(int)
        # Q.astype(int)
        # t = np.zeros((len(I), 2), dtype=np.short)
        # t[:, 0] = I[:] 
        # t[:, 1] = Q[:] 
        # t = t.reshape((2*len(I), ))
        # t.tofile("CRCtest.iq")
        passing_scores = fasti
        start_index = passing_scores
        actual_start_index = start_index - padding - precal_func_config["zc_seq_offset"]
        if actual_start_index >= 0:
            burst = true_siganl_tmp[int(actual_start_index):int(actual_start_index)+9980]
            if int(actual_start_index)+9980+9980+200 <= len(true_siganl_tmp):
                noise = true_siganl_tmp[int(actual_start_index)+9980+200:int(actual_start_index)+9980+9980+200]
            else:
                noise = true_siganl_tmp[:9980]
            signalpower=20*np.log10(np.mean(np.abs(burst)))
            noisepower=20*np.log10(np.mean(np.abs(noise)))
            
            if len(burst)==9980:
                print("hackrf_num = ", hackrf_num)
                print("signalpower = ",signalpower)
                print("noisepower = ",noisepower)
                # plt.figure()
                # freq, times, spectrogram = signal.spectrogram(burst, fs=15.36e6,window = ('tukey', 0.01),return_onesided=False)
                # plt.pcolormesh(times, fft.fftshift(freq), 20*np.log(fft.fftshift(spectrogram, axes=0)), shading='gouraud')
                # plt.show()
                solve_burst(burst, start_timestamp, precal_func_config, hackrf_num, worker_idx, processcut1000_idx)
                # I = burst.real
                # Q = burst.imag
                # I.astype(int)
                # Q.astype(int)
                # t = np.zeros((len(I), 2), dtype=np.short)
                # t[:, 0] = I[:] 
                # t[:, 1] = Q[:] 
                # t = t.reshape((2*len(I), ))
                # t.tofile(f"CRCtest_{start_timestamp}.iq")
                # burst111 = true_siganl_tmp[int(actual_start_index)-5000:int(actual_start_index)+9980+5000]
                

                
            else:
                pass
    else:
        # print("Did not find any bursts")
        pass    

if __name__ == '__main__':
    pass
    
