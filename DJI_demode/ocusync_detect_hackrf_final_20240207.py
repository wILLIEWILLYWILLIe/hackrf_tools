import datetime
import numpy as np
import serial
import threading
import queue
import time
# from smdevice.sm_api import *
from find_burst import *
from mathematic_func import *
import scipy
from scipy.fft import fftshift
# import hackrf as pyhackrf
from pyhackrf2 import HackRF


# ==============================================================================================
directory_path = os.path.join(cwd, 'tmp_bits')
if not os.path.exists(os.path.join(cwd, 'tmp_bits')):
    os.mkdir(directory_path)
result = subprocess.run(f"sudo chmod -R o+w {directory_path}", shell=True, stdout=subprocess.PIPE, check=True, text=True)

directory_path = os.path.join(cwd, 'drone')
if not os.path.exists(os.path.join(cwd, 'drone')):
    os.mkdir(directory_path)
result = subprocess.run(f"sudo chmod -R o+w {directory_path}", shell=True, stdout=subprocess.PIPE, check=True, text=True)


class ArduinoControl:
	def __init__(
		self,
		serial_port,
		baudrate = 115200
	):
		self.__serial_port = serial_port
		self.__baudrate = baudrate
		self.__serial = serial.Serial(port=self.__serial_port, baudrate=self.__baudrate)
	
	def write(self, data_byte):
		self.__serial.write(data_byte)
	
	def rest(self):
		self.__serial.close()
		self.__serial = serial.Serial(port=self.__serial_port, baudrate=self.__baudrate)


class Findsignal:
	def __init__(
		self,
		collectdata     = None, 
		fs              = 15.36e6,
		Td              = 0.02, 
		Tdfast          = 0.002,
		Arduino         = None,
		numprocesscut   = 8   ,
		numprocesscutfast = 1 ,
		fft_size        = 1024,
		correlationscore= 0.1,
		ant_list = [b'd0', b'd1', b'd2', b'd3', b'd4', b'd5', b'd6', b'd7'],
		run2G = 1,
	):
		self.__fs = fs
		self.__Td = Td    #two methods, each need sliced different section in time domain
		self.__Tdfast = Tdfast
		self.numprocesscut = numprocesscut   #two methods, each need sliced different section in time domain
		self.numprocesscutfast = numprocesscutfast
		self.numprocess2G = 15   #15 channel
		self.numprocess5G = 24   #24 channel
		self.numprocessALL = 3
		self.process_list_cut2G = []
		self.process_list_cut5G = []
		self.__ArduinoControl = Arduino
		N = int(15.36e6*self.__Td/self.numprocesscut)
		Filter_T_10M,zcfilter_conj_10M,zcfilter_conj_var_sqrt_10M = precalculate_transform_filter_findsignal(1,N)
		filter20M = zcsequence_findsignal20M(2048,1)
		Filter_T_20M,zcfilter_conj_20M,zcfilter_conj_var_sqrt_20M = precalculate_transform_filter_original(filter20M,N)
		self.Filter_T_10M = Filter_T_10M
		self.zcfilter_conj_10M = zcfilter_conj_10M
		self.zcfilter_conj_var_sqrt_10M = zcfilter_conj_var_sqrt_10M
		self.Filter_T_20M = Filter_T_20M
		self.zcfilter_conj_20M = zcfilter_conj_20M
		self.zcfilter_conj_var_sqrt_20M = zcfilter_conj_var_sqrt_20M
		self.fft_size = fft_size
		self.correlationscore = correlationscore
		q = queue.Queue()
		self.q = q
		# findq = queue.Queue()
		# self.findq = findq
		self.confirm_signal_queue = queue.Queue()
		#/////////////////////////////////////////////////////
		self.start_collect = collectdata
		self.ant_list = ant_list
		self.now_ant_list_index = 0
		self.RUN2G = run2G
		self.round_check = 0
		self.exit_hackrf_idx_list = []
		#/////////////////////////////////////////////////////

#mathmetic calculaiton/////////////////////////////////////
	def find_burst_sol4_video(self, sampleTcut, samplesFcut_twice, typeM, csk, fc):
		fft_size=self.fft_size
		if typeM==10:
			Filter = self.Filter_T_10M
			zcfilter_conj = self.zcfilter_conj_10M
			zcfilter_conj_var_sqrt = self.zcfilter_conj_var_sqrt_10M
		elif typeM==20:
			Filter = self.Filter_T_20M
			zcfilter_conj = self.zcfilter_conj_20M
			zcfilter_conj_var_sqrt = self.zcfilter_conj_var_sqrt_20M
		# else:
		# 	Filter = self.Filter_T_O4   #no need for 40M, we can use 20M to find 
		# 	zcfilter_conj = self.zcfilter_conj_O4
		# 	zcfilter_conj_var_sqrt = self.zcfilter_conj_var_sqrt_O4

		window_size = len(zcfilter_conj)
		recip_window_size = 1/window_size
		FFTdata = Filter * samplesFcut_twice
		IFFTdata = scipy.fft.ifft(FFTdata)
		IFFTdatacut = IFFTdata[fft_size : len(IFFTdata)]
		tmpscore1 = np.absolute(IFFTdatacut / csk)
		fasti = np.argmax(tmpscore1)
		# print(fasti)
		# scorestemp=np.zeros(2048,dtype=complex)
		# idx=0
		# for idx in range(20):#可再加速
		# 	if fasti<10:
		# 		continue
		# 	if len(sampleTcut)-fasti<10:
		# 		continue
		window_fast = sampleTcut[fasti:fasti + window_size]
		running_sum_fast = np.sum(window_fast)
		window1_fast = window_fast - (running_sum_fast * recip_window_size)
		prod_fast = np.sum(window1_fast * zcfilter_conj) * recip_window_size
		variance = np.var(window1_fast)
		fast_scores = prod_fast /(np.sqrt(variance) * zcfilter_conj_var_sqrt)
		# scorestemp[idx] = prod_fast /(np.sqrt(variance) * zcfilter_conj_var_sqrt)

		abs_scores = np.abs(fast_scores)**2
		# fasti3=np.argmax(abs_scores)
		# print(fasti+fasti3-5)
		# # print(abs_scores)
		# print(np.max(abs_scores))
		if abs_scores > self.correlationscore:
			print(abs_scores)
			# startpoint=fasti+fasti3-2048
			self.q.put(fc)
	
	def cut_signal(self,i,sampleTcut,samplesFcut,fc):
		fft_size = self.fft_size
		cs = np.cumsum(np.abs(sampleTcut)**2)
		cumsum_front = cs[0 : len(cs)-fft_size]
		cumsum_back = cs[fft_size : len(cs)]
		csk=np.sqrt(cumsum_back-cumsum_front+1)
		# PO4 =threading.Thread(target=Findsignal.find_burst_sol4_video, args=(self,sampleTcut,samplesFcut,100,CSK,fc))
		P10M = threading.Thread(target = self.find_burst_sol4_video, args=(sampleTcut,samplesFcut,10,csk,fc))
		P20M = threading.Thread(target = self.find_burst_sol4_video, args=(sampleTcut,samplesFcut,20,csk,fc))
		# PO4.start()
		P10M.start()
		P20M.start()
		P10M.join()
		P20M.join()
		# PO4.join()
	
	def fasttest(self,i,samplesTcut,samplesFcut,fc):
		# print("fc",fc)
		overscore = []
		corrscore=0.2
		# corrscoresmall=0.1
		A_cal_all=np.conj(samplesTcut[0:len(samplesTcut)- 1024])
		B_cal_all=samplesTcut[1024:len(samplesTcut)]
		prob_cal_all=A_cal_all*B_cal_all
		prob_cal_allsum = np.cumsum(prob_cal_all)
		prob_cal_sumcut = (prob_cal_allsum[72:]-prob_cal_allsum[:-72])/72
		test3 = np.abs(prob_cal_sumcut)**2
		mediantest3 = np.percentile(test3, 90)

		cal_all = samplesTcut
		cal_all_allsum = np.cumsum(cal_all)
		cal_all_sumcut=(cal_all_allsum[72:]-cal_all_allsum[:-72])/72

		cal_all_squ=np.abs(samplesTcut)**2
		cal_all_squ_allsum = np.cumsum(cal_all_squ)
		cal_all_squ_sumcut=(cal_all_squ_allsum[72:]-cal_all_squ_allsum[:-72])/72
		cal_all_std=np.sqrt(np.abs(cal_all_squ_sumcut-np.abs(cal_all_sumcut)**2))
		cal_all_std_mul=(cal_all_std[:-1024]*cal_all_std[1024:])**2
		prodc1=test3/cal_all_std_mul
		prodc1 = np.where(test3 > mediantest3, prodc1, 0)
		scores=np.abs(prodc1)
		# plt.plot(scores)
		# plt.show()

		[mylist,] = np.where(scores > corrscore)
		dst_list = list(map(int, mylist))
		tempSC=-1
		tempin=-1
		for inn in dst_list:
			if inn-50>tempin and tempin>=0:
				overscore.append(tempin)
				tempSC=-1
			if scores[inn]>tempSC:
				tempSC=scores[inn]
				tempin=inn
		overscore.append(tempin)
		# print(overscore)

		for inin in overscore:
			testA1=scores[inin+1096-200:inin+1096+200]
			[A1index,] = np.where(testA1 > corrscore)
			if A1index.size > 0:
				EVO_findA1=1   
			else:
				continue
			testB1=scores[inin+548-300:inin+548+300]
			[B1index,] = np.where(testB1 > scores[inin]/2)
			if B1index.size > 0:
				continue
			else:
				EVO_findB1=1
			testA2=scores[inin+1096+1096-200:inin+1096+1096+200]
			[A2index,] = np.where(testA2 > corrscore)
			if A2index.size > 0:
				EVO_findA2=1   
			else:
				continue
			testB2=scores[inin+548+1096-300:inin+548+1096+300]
			[B2index,] = np.where(testB2 > scores[inin+1096]/2)
			if B2index.size > 0:
				continue
			else:
				EVO_findB2=1
			
			testA3=scores[inin+1096+1096*2-200:inin+1096+1096*2+200]
			[A3index,] = np.where(testA3 > corrscore)
			if A3index.size > 0:
				EVO_findA3=1   
			else:
				continue
			testB3=scores[inin+548+1096*2-300:inin+548+1096*2+300]
			[B3index,] = np.where(testB3 > scores[inin+1096*2]/2)
			if B3index.size > 0:
				continue
			else:
				EVO_findB3=1
			
			# testABC=scores[inin+1096+1096+548-1096:inin+1096+1096+548+1096]
			# [ABCindex,] = np.where(testABC > corrscoresmall)
			# if ABCindex.size > 0:
			# 	continue
			# else:
			# 	EVO_findABC=1

			# plt.plot(scores[inin+1096-200:inin+1096*4+200])
			# plt.show()
			self.q.put(fc)
	
	def find_signal(self,i,samplesF,band,run_fast):
		fc_sliced = band+(i-1)*2.32e6
		if run_fast==1:
			timecut=self.__Tdfast/self.numprocesscutfast
		else:
			timecut=self.__Td/self.numprocesscut
		samplesFcut=samplesF[int(len(samplesF)/2+(i-1)/2*4.64e6*timecut-15.36e6*timecut/2):int(len(samplesF)/2+(i-1)/2*4.64e6*timecut+15.36e6*timecut/2)]
		sampleTcut=scipy.fft.ifft(scipy.fft.fftshift(samplesFcut))
		# freq, times, spectrogram = signal.spectrogram(sampleTcut, fs=15.36e6,window = ('tukey', 0.01),return_onesided=False)
		# plt.pcolormesh(times, fftshift(freq), 20*np.log(fftshift(spectrogram, axes=0)), shading='gouraud')
		# plt.show()

		if run_fast == 0:
			Findsignal.cut_signal(self, i, sampleTcut, samplesFcut, fc_sliced)
		else :
			Findsignal.fasttest(self, i, sampleTcut, samplesFcut, fc_sliced)

	def numprocess(self, i, samples, band, run_fast, center_freq):
		if run_fast==1:
			numprocesscut = self.numprocesscutfast   # = 1
		else:
			numprocesscut = self.numprocesscut       # = 8
		samplescut = samples[int(i*len(samples)/numprocesscut):int((i+1)*len(samples)/numprocesscut)]
		samplesF = scipy.fft.fftshift(scipy.fft.fft(samplescut))

		if center_freq != 0:
			Findsignal.cut_signal(self,i,samplescut,samplesF,center_freq)
		else:
			self.numprocessALL = 2  #hackrf each band separate 2 channels
			process_list= [[] for i in range(numprocesscut)]
			for j in range(self.numprocessALL):
				process_list[i].append(threading.Thread(target=Findsignal.find_signal, args=(self,j,samplesF,band,run_fast)))
			for j in range(self.numprocessALL):
				process_list[i][j].start()
			for j in range(self.numprocessALL): 
				process_list[i][j].join()

	def getspecialsignal(self):
		# span_all_channel = [2.4145e9,2.4295e9,2.4445e9,2.4595e9,5.7565e9,5.7765e9,5.7965e9,5.8165e9]
		i = 16
		print("="*10, "Enter Get Special Signal", "="*10)
		while i:
			round=16-i
			# k=int(i%4)+4
			# k=int(i%4)
			#k=int(i%8)
			print("="*3,"Special signal round",round, "="*3)
			# start_time = datetime.datetime.now()
			# hackrf_center_freq = totalchannel[k]  # 1 GHz

			self.hackrf_getspecialsignal_workerlist = []
			exit_event = threading.Event()
			for hackrf_index in range(len(self.hackrf_list)):
				hackrf = self.hackrf_list[hackrf_index]
				hackrf_worker = threading.Thread(target = self.hackrf_receive, args = (hackrf, exit_event, self.droneid_channel[hackrf_index]))
				hackrf_worker.start()
				self.hackrf_getspecialsignal_workerlist.append(hackrf_worker)
			for hackrf_worker in self.hackrf_getspecialsignal_workerlist:
				hackrf_worker.join()
			# end123 = datetime.datetime.now()
			# print("執行時間：%f 秒" % (end123 - start_time).total_seconds())
			i=i-1

	#選取切天線判定機制,速度慢距離遠(無O4)
	def realtimeLoop_coor(self, hackrf, channel_list, queue, Exit_event):
		exit_event = Exit_event
		RUN2G = self.RUN2G
		channel_list_5G = channel_list[0]
		channel_list_2G = channel_list[1]
		channel_list_5G_index = 0
		while not queue.qsize() > 0 and channel_list_5G_index < len(channel_list_5G) and not exit_event.is_set():
			try:
				hackrf.set_freq(channel_list_5G[channel_list_5G_index])
				# print(f"{channel_list_5G[channel_list_5G_index]}")
				hackrf.set_sample_rate(self.__fs)
				hackrf_total_sample_count = self.__fs * self.__Td
				samples = hackrf.read_samples(hackrf_total_sample_count)
				samples = samples-1+1j
				# print(max(np.abs(samples)))
				process_list_cut5G=[]
				for i in range(self.numprocesscut):
					process_list_cut5G.append(threading.Thread(target=Findsignal.numprocess, args=(self,i,samples,
																					channel_list_5G[channel_list_5G_index],0,0)))
				for i in range(self.numprocesscut):
					process_list_cut5G[i].start()
				for i in range(self.numprocesscut): 
					process_list_cut5G[i].join(timeout = 10)
					if process_list_cut5G[i].is_alive():
						# 如果超時，你可以做一些處理，例如設定一個回傳值
						print("some hackrf dead... over timeout")
						exit_event.set()
				channel_list_5G_index = channel_list_5G_index + 1
			except:
				print("some hackrf dead...")
				exit_event.set()

		if queue.qsize()>0:
			# print("maybe signal")
			# freq_center = []
			while not queue.empty():
				k = queue.get()
				# freq_center.append(k)
				print(f"video stream fc = {k}")
			queue.queue.clear()
			print("find signal")
			RUN2G = 0
			self.confirm_signal_queue.put(1)
					
		if RUN2G == 1:
			channel_list_2G_index = 0
			while not queue.qsize() > 0 and channel_list_2G_index < len(channel_list_2G) and not exit_event.is_set():
				try:
					hackrf.set_freq(channel_list_2G[channel_list_2G_index])
					# print(f"{channel_list_2G[channel_list_2G_index]}")
					hackrf.set_sample_rate(self.__fs)
					hackrf_total_sample_count = self.__fs * self.__Td
					samples = hackrf.read_samples(hackrf_total_sample_count)
					samples = samples-1+1j
					process_list_cut2G=[]
					for i in range(self.numprocesscut):
						process_list_cut2G.append(threading.Thread(target=Findsignal.numprocess, args=(self,i,samples,channel_list_2G[channel_list_2G_index],0,0)))
					for i in range(self.numprocesscut):
						process_list_cut2G[i].start()
					for i in range(self.numprocesscut): 
						process_list_cut2G[i].join(timeout = 10)
						if process_list_cut2G[i].is_alive():
						# 如果超時，你可以做一些處理，例如設定一個回傳值
							print("some hackrf dead... over timeout")
							exit_event.set()
					channel_list_2G_index = channel_list_2G_index + 1
				except:
					print("some hackrf dead...")
					exit_event.set()

			if queue.qsize()>0:
				# freq_center=[]
				while not queue.empty():
					k=queue.get()
					# freq_center.append(k)
					print(f"video stream fc = {k}")
				queue.queue.clear()
				print("find signal")
				self.confirm_signal_queue.put(1)
	
	#選取切天線判定機制,速度快誤判少(無Ocusync 4.0)
	def realtimeLoop(self, hackrf, channel_list, queue, Exit_event):
		exit_event = Exit_event
		RUN2G = self.RUN2G
		run_fast=1
		channel_list_5G = channel_list[0]
		channel_list_2G = channel_list[1]
		channel_list_5G_index = 0
		while not queue.qsize() > 0 and channel_list_5G_index < len(channel_list_5G) and not exit_event.is_set():
			try:
				hackrf.set_freq(channel_list_5G[channel_list_5G_index])
				# print(f"{channel_list_5G[channel_list_5G_index]}")
				hackrf.set_sample_rate(self.__fs)
				hackrf_total_sample_count = self.__fs * self.__Tdfast
				samples = hackrf.read_samples(hackrf_total_sample_count)
				samples = samples-1+1j
				Findsignal.numprocess(self, i = 0, samples = samples, band = channel_list_5G[channel_list_5G_index], run_fast = 1, center_freq = 0)	
				channel_list_5G_index = channel_list_5G_index + 1
			except:
				print("some hackrf dead...")
				exit_event.set()

		if queue.qsize() > 0:
			print("maybe signal")
			freq_center = []
			while not queue.empty():
				k1 = queue.get()
				k = int(k1)
				if k not in freq_center:
					freq_center.append(k)
					print(k)
			queue.queue.clear()

			for cf in freq_center:
				center_freq = cf
				####################### 5G Band ############################################
				hackrf.set_freq(center_freq)
				hackrf.set_sample_rate(15.36e6)
				hackrf_total_sample_count = 15.36e6 * self.__Td
				samples = hackrf.read_samples(hackrf_total_sample_count)
				samples = samples-1+1j

				run_fast=0
				process_list_cut5G=[]
				for i in range(self.numprocesscut):
					process_list_cut5G.append(threading.Thread(target=Findsignal.numprocess, args=(self, i, samples, center_freq, run_fast, center_freq)))
				for i in range(self.numprocesscut):
					process_list_cut5G[i].start()
				for i in range(self.numprocesscut): 
					process_list_cut5G[i].join()
				if queue.qsize()>0:
					break
			
			if queue.qsize()>0:
				print("find signal")
				while not queue.empty():
					k=queue.get()
					print(k)
				queue.queue.clear()
				RUN2G=0
				# Findsignal.getspecialsignal(self)
				self.confirm_signal_queue.put(1)
					
		if RUN2G==1:   #為了節省時間，可以關掉不看2.4頻段
			channel_list_2G_index = 0
			while not queue.qsize() > 0 and channel_list_2G_index < len(channel_list_2G) and not exit_event.is_set():
				try:
					hackrf.set_freq(channel_list_2G[channel_list_2G_index])
					# print(f"{channel_list_2G[channel_list_2G_index]}")
					hackrf.set_sample_rate(self.__fs)
					hackrf_total_sample_count = self.__fs * self.__Tdfast
					samples = hackrf.read_samples(hackrf_total_sample_count)
					samples = samples-1+1j
					Findsignal.numprocess(self, i = 0, samples = samples, band = channel_list_2G[channel_list_2G_index], run_fast = 1, center_freq = 0)
					channel_list_2G_index = channel_list_2G_index + 1
				except:
					print("some hackrf dead...")
					exit_event.set()

			if queue.qsize()>0:
				print("maybe signal")
				freq_center=[]
				while not queue.empty():
					k1 = queue.get()
					k = int(k1)
					if k not in freq_center:
						freq_center.append(k)
						print(k)
				queue.queue.clear()

				for cf in freq_center:
					center_freq = cf
					# ###################### 2G Band ############################################
					hackrf.set_freq(center_freq)
					hackrf.set_sample_rate(15.36e6)
					hackrf_total_sample_count = 15.36e6 * self.__Td
					samples = hackrf.read_samples(hackrf_total_sample_count)
					samples = samples-1+1j

					run_fast=0
					process_list_cut2G=[]
					for i in range(self.numprocesscut):
						process_list_cut2G.append(threading.Thread(target=Findsignal.numprocess, args=(self,i,samples, center_freq, run_fast, center_freq)))
					for i in range(self.numprocesscut):
						process_list_cut2G[i].start()
					for i in range(self.numprocesscut): 
						process_list_cut2G[i].join()
					if queue.qsize()>0:
						break

				if queue.qsize()>0:
					print("find signal")
					while not queue.empty():
						k=queue.get()
						print(k)
					queue.queue.clear()
					# Findsignal.getspecialsignal(self)
					self.confirm_signal_queue.put(1)
	
	#選取切天線判定機制,誤判多速度快(有O4)
	def realtimeLoop_auto_only(self, hackrf, channel_list, queue, Exit_event):
		exit_event = Exit_event
		RUN2G = self.RUN2G
		#fast find signal first
		#############################################################
		# self.hackrf_receive(hackrf, fcnow1[0])
		channel_list_5G = channel_list[0]
		channel_list_2G = channel_list[1]
		channel_list_5G_index = 0
		while not queue.qsize() > 0 and channel_list_5G_index < len(channel_list_5G) and not exit_event.is_set():
			try:
				hackrf.set_freq(channel_list_5G[channel_list_5G_index])
				# print(f"{channel_list_5G[channel_list_5G_index]}")
				hackrf.set_sample_rate(self.__fs)
				hackrf_total_sample_count = self.__fs * self.__Tdfast
				samples = hackrf.read_samples(hackrf_total_sample_count)
				samples = samples-1+1j
				Findsignal.numprocess(self, i = 0, samples = samples, band = channel_list_5G[channel_list_5G_index], run_fast = 1, center_freq = 0)	
				channel_list_5G_index = channel_list_5G_index + 1
			except:
				print("some hackrf dead...")
				exit_event.set()

		if queue.qsize()>0:
			print("maybe signal")
			queue.queue.clear()
			#ROUND2 prevent misleading
			#############################################################
			channel_list_5G_index = 0
			while not queue.qsize() > 0 and channel_list_5G_index < len(channel_list_5G) and not exit_event.is_set():
				try:
					hackrf.set_freq(channel_list_5G[channel_list_5G_index])
					# print(f"{channel_list_5G[channel_list_5G_index]}")
					hackrf.set_sample_rate(self.__fs)
					hackrf_total_sample_count = self.__fs * self.__Tdfast
					samples = hackrf.read_samples(hackrf_total_sample_count)
					samples = samples-1+1j
					Findsignal.numprocess(self, i = 0, samples = samples, band = channel_list_5G[channel_list_5G_index], run_fast = 1, center_freq = 0)	
					channel_list_5G_index = channel_list_5G_index + 1
				except:
					print("some hackrf dead...")
					exit_event.set()

			if queue.qsize()>0:
				print("find signal")
				queue.queue.clear()
				RUN2G = 0
				# Findsignal.getspecialsignal(self)
				self.confirm_signal_queue.put(1)
					
		if RUN2G==1:
			channel_list_2G_index = 0
			while not queue.qsize() > 0 and channel_list_2G_index < len(channel_list_2G) and not exit_event.is_set():
				try:
					hackrf.set_freq(channel_list_2G[channel_list_2G_index])
					# print(f"{channel_list_2G[channel_list_2G_index]}")
					hackrf.set_sample_rate(self.__fs)
					hackrf_total_sample_count = self.__fs * self.__Tdfast
					samples = hackrf.read_samples(hackrf_total_sample_count)
					samples = samples-1+1j
					Findsignal.numprocess(self, i = 0, samples = samples, band = channel_list_2G[channel_list_2G_index], run_fast = 1, center_freq = 0)	
					channel_list_2G_index = channel_list_2G_index + 1
				except:
					print("some hackrf dead...")
					exit_event.set()

			if queue.qsize()>0:
				print("maybe signal")
				queue.queue.clear()
				channel_list_2G_index = 0
				while not queue.qsize() > 0 and channel_list_2G_index < len(channel_list_2G) and not exit_event.is_set():
					try:
						hackrf.set_freq(channel_list_2G[channel_list_2G_index])
						# print(f"{channel_list_2G[channel_list_2G_index]}")
						hackrf.set_sample_rate(self.__fs)
						hackrf_total_sample_count = self.__fs * self.__Tdfast
						samples = hackrf.read_samples(hackrf_total_sample_count)
						samples = samples-1+1j
						Findsignal.numprocess(self, i = 0, samples = samples, band = channel_list_2G[channel_list_2G_index], run_fast = 1, center_freq = 0)	
						channel_list_2G_index = channel_list_2G_index + 1
					except:
						print("some hackrf dead...")
						exit_event.set()
				
				if queue.qsize()>0:
					print("find signal")
					queue.queue.clear()
					# Findsignal.getspecialsignal(self)
					self.confirm_signal_queue.put(1)
#/////////////////////////////////////////////////////////


	def readfile(self, file_to_read):
		# file_to_read = "AIR3_10Mfilter.iq"
		start_timestamp = datetime.datetime.now()
		
		# samplesTcut = samples
		# samplesF = np.fft.fftshift(np.fft.fft(samples))
		# samplesFcut = samplesF[int(len(samplesF)/2-(15.36e6*0.02/2)):int(len(samplesF)/2+(15.36e6*0.02/2))]
		# samplesTcut = np.fft.ifft(np.fft.fftshift(samplesFcut))
		# samplesTcut = samplesTcut[43197:43197+1024]
		# filternew=samplesTcut.copy()

		# file_to_read = "AAA.iq"
		with open(file_to_read, mode="rb") as file:
			my_bytes = file.read()
		import_file = np.frombuffer(my_bytes, dtype=np.int16)

		samples = import_file[0:len(import_file)-1:2] + 1j*import_file[1:len(import_file):2]
		samples = samples
		#freq, times, spectrogram = signal.spectrogram(samples, fs=15.36e6,window = ('tukey', 0.01),return_onesided=False)

		precal_func_config = pre_cal_func(15.36e6)
		find_burst_sol4(0,samples,start_timestamp, precal_func_config, 2)
		find_burst_sol4(1,samples,start_timestamp, precal_func_config, 2)

		#plt.figure(1)
		#plt.pcolormesh(times, fftshift(freq), 20*np.log(fftshift(spectrogram, axes=0)), shading='gouraud')
		#plt.show() 
		# freq_domain = np.fft.fftshift(np.fft.fft(samples))
		# freq_domain = np.delete(freq_domain,len(freq_domain)//2)
		# freq_domain = np.abs(freq_domain)**2
		# plt.figure(2)
		# plt.plot(freq_domain)
		# plt.show() 
	
	#nosweep => Did signal only
	def start_hackrf_nosweep(self):
		try:
			break_event = 0
			self.set_droneid_channel()   #依照hackrf數量進行channel區分
			while break_event != 1:
				start_timestamp = datetime.datetime.now()
				self.hackrf_workerlist = []
				for hackrf_index, hackrf in enumerate(self.hackrf_list):
					Exit_event = self.exit_event_list[hackrf_index]
					channel_list = self.droneid_channel[hackrf_index]
					hackrf_worker = threading.Thread(target = self.hackrf_receive, args = (hackrf, Exit_event, channel_list))
					hackrf_worker.start()
					self.hackrf_workerlist.append(hackrf_worker)
				for hackrf_worker in self.hackrf_workerlist:
					hackrf_worker.join(timeout = 30)

				for Exit_event_idx,  Exit_event in enumerate(self.exit_event_list):
					if Exit_event.is_set():
						break_event = 1
						print("Exit event set")

				try:
					cmd = "hackrf_info"
					result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
					# print(result.stdout)
					sub_string = "Found HackRF"
					usb_check_hackrf_num = result.stdout.count(sub_string)
					print(f"Now {usb_check_hackrf_num} hackrf still working")
					if usb_check_hackrf_num < len(self.hackrf_workerlist):
						break_event = 1
						print("in break loop")
				except:
					break_event = 1
					print("in break loop")

				if self.__ArduinoControl != None:
					string_tobytes = self.ant_list[self.now_ant_list_index]
					self.__ArduinoControl.write(string_tobytes)
				self.now_ant_list_index = (self.now_ant_list_index + 1) % len(self.ant_list)
				end = datetime.datetime.now()
				print("執行時間：%f 秒" % (end - start_timestamp).total_seconds())

			self.break_process_start()
		
		except KeyboardInterrupt:
			self.break_process_start()
	
	#sweep => video transmit signal first then Did signal 
	def start_hackrf_sweep(self, sweep_type):
		try:
			break_event = 0
			self.set_droneid_channel()
			self.set_sweep_channel_list()
			while break_event != 1:
				start_timestamp = datetime.datetime.now()
				print("="*5,f"NOW SWITCH TO ANT{str(self.ant_list[self.now_ant_list_index])[-2]}","="*5)
				self.hackrf_workerlist = []
				for hackrf_index, hackrf in enumerate(self.hackrf_list):
					Exit_event = self.exit_event_list[hackrf_index]
					if sweep_type == 1: 
						hackrf_worker = threading.Thread(target = self.realtimeLoop, args = (hackrf, self.sweep_channel_list[hackrf_index], self.q, Exit_event))
					elif sweep_type == 2: 
						hackrf_worker = threading.Thread(target = self.realtimeLoop_coor, args = (hackrf, self.sweep_channel_list[hackrf_index], self.q, Exit_event))
					else: 
						hackrf_worker = threading.Thread(target = self.realtimeLoop_auto_only, args = (hackrf, self.sweep_channel_list[hackrf_index], self.q, Exit_event))
					hackrf_worker.start()
					self.hackrf_workerlist.append(hackrf_worker)
				for hackrf_worker in self.hackrf_workerlist:
					hackrf_worker.join(timeout = 30)
				for Exit_event_idx,  Exit_event in enumerate(self.exit_event_list):
					if Exit_event.is_set():
						break_event = 1
						
				try:
					cmd = "hackrf_info"
					result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
					# print(result.stdout)
					sub_string = "Found HackRF"
					usb_check_hackrf_num = result.stdout.count(sub_string)
					print(f"Now {usb_check_hackrf_num} hackrf still working")
					if usb_check_hackrf_num < len(self.hackrf_workerlist):
						break_event = 1
						print("in break loop")
				except:
					break_event = 1
					print("in break loop")

				if self.confirm_signal_queue.qsize() != 0:
					self.confirm_signal_queue.queue.clear()
					self.getspecialsignal()  #確定有圖傳後，掃Did

				if self.__ArduinoControl != None:
					resttimestart = datetime.datetime.now()
					string_tobytes = self.ant_list[self.now_ant_list_index]
					self.__ArduinoControl.write(string_tobytes)
					resttimeend = datetime.datetime.now()
					resttime = (resttimeend - resttimestart).total_seconds()
					if resttime > 0.01:
						self.__ArduinoControl.rest()
						string_tobytes = self.ant_list[self.now_ant_list_index]
						self.__ArduinoControl.write(string_tobytes)
				self.now_ant_list_index = (self.now_ant_list_index + 1)% len(self.ant_list)
				end = datetime.datetime.now()
				print("執行時間：%f 秒" % (end - start_timestamp).total_seconds())
				
			self.break_process_start()
		
		except KeyboardInterrupt:
			pass
			# self.break_process_start()

	#according to channel list, the code which actual collect iq data
	def hackrf_receive(self, hackrf, Exit_event, channel_list = [2.4145e9, 2.4295e9, 2.4445e9, 2.4595e9, 5.7565e9, 5.7765e9, 5.7965e9, 5.8165e9]):
		#only for DJI remoteid 
		i = 0
		while i < len(channel_list):
			# print("in loop")
			try:
				start_time = datetime.datetime.now()
				center_freq = channel_list[i]
				# print(f"Now {int(center_freq)} Hz at antenna {self.now_ant_list_index}")
				hackrf.center_freq = center_freq
				hackrf.sample_rate = 15.36e6
				hackrf_total_sample_count = 15.36e6*0.35   #sample rate * time duration
				samples = hackrf.read_samples(hackrf_total_sample_count)
				samples = samples-1+1j
				# freq, times, spectrogram = signal.spectrogram(samples, fs=15.36e6*2,window = ('tukey', 0.01),return_onesided=False)
				# plt.pcolormesh(times, fftshift(freq), 20*np.log(fftshift(spectrogram, axes=0)), shading='gouraud')
				# plt.show()
				hackrf_num = hackrf.get_serial_no()
				self.start_collect.realtimeLoop(samples, start_time, hackrf_num)
				i = i+1
				end = datetime.datetime.now()
				print(f"Now {int(center_freq)} Hz at antenna {self.now_ant_list_index} 執行時間：{(end - start_time).total_seconds()} 秒")
			except:
				Exit_event.set()
				break

	def arduino_init(self):
		if self.ant_list != []:
			string_tobytes = self.ant_list[self.now_ant_list_index]
			self.__ArduinoControl.write(string_tobytes)
		else:
			raise ValueError("="*10,"antenna setting forum wrong","="*10)
	
	def hackrf_init(self, num_hackrf = 1, hackrf_sample_rate = 15.36e6, lock_hackrf_idx = 0,
				 hackrf_duration = 0.35, hackrf_vga_gain = 40, hackrf_lna_gain = 24):
		# self.num_hackrf = num_hackrf
		if lock_hackrf_idx == 0:
			cmd = "hackrf_info"
			result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
			sub_string = "Found HackRF"
			self.num_hackrf = result.stdout.count(sub_string)
			device_index_list = range(self.num_hackrf)
		else:
			self.num_hackrf = len(lock_hackrf_idx)
			device_index_list = lock_hackrf_idx
		
		self.hackrf_list = []
		self.hackrf_device_pointer_list = []
		self.__fs = hackrf_sample_rate
		self.hackrf_total_sample_count = hackrf_duration * hackrf_sample_rate	
		self.exit_event_list = []
		self.hackrf_vga_gain = hackrf_vga_gain
		self.hackrf_lna_gain = hackrf_lna_gain
		for i in device_index_list:
			# hackrf = pyhackrf.HackRF(device_index = i)
			hackrf = HackRF(device_index = i)
			hackrf_sample_rate = int(hackrf_sample_rate)
			print(hackrf_sample_rate)
			hackrf.sample_rate = hackrf_sample_rate
			hackrf.lna_gain = self.hackrf_lna_gain
			hackrf.vga_gain = self.hackrf_vga_gain
			self.hackrf_list.append(hackrf)
			# self.hackrf_device_pointer_list.append(hackrf.dev_p.value)
			self.exit_event_list.append(threading.Event())
	
	def set_sweep_channel_list(self):
		#num_hackrf =4 => all channel, =1 => only fc_sweep1, =2 => sweep fc_sweep1 and fc_sweep2 ,etc
		if len(self.hackrf_list) == 1:
			fc_sweep1 = [[5.765e9, 5.780e9, 5.735e9, 5.750e9, 5.825e9, 5.840e9, 5.795e9, 5.810e9], [2.415e9, 2.445e9, 2.430e9, 2.460e9]]
			sweep_channel_list = [fc_sweep1]
		elif len(self.hackrf_list) == 2:
			fc_sweep1 = [[5.765e9, 5.780e9, 5.735e9, 5.750e9], [2.415e9, 2.445e9]]
			fc_sweep2 = [[5.825e9, 5.840e9, 5.795e9, 5.810e9], [2.430e9, 2.460e9]]
			sweep_channel_list = [fc_sweep1, fc_sweep2]
		elif len(self.hackrf_list) == 3:
			fc_sweep1 = [[5.735e9, 5.750e9, 5.825e9], [2.415e9]]
			fc_sweep2 = [[5.795e9, 5.810e9, 5.840e9], [2.430e9]]
			fc_sweep3 = [[5.765e9, 5.780e9], [2.445e9, 2.460e9]]
			sweep_channel_list = [fc_sweep1, fc_sweep2, fc_sweep3]
		elif len(self.hackrf_list) == 4:
			fc_sweep1 = [[5.735e9, 5.750e9], [2.415e9]]
			fc_sweep2 = [[5.795e9, 5.810e9], [2.430e9]]
			fc_sweep3 = [[5.765e9, 5.780e9], [2.445e9]]
			fc_sweep4 = [[5.825e9, 5.840e9], [2.460e9]]
			sweep_channel_list = [fc_sweep1, fc_sweep2, fc_sweep3, fc_sweep4]
		self.sweep_channel_list = sweep_channel_list
	
	def set_droneid_channel(self):
		if len(self.hackrf_list) == 1:
			channel1   = [2.4145e9, 2.4295e9, 2.4445e9, 2.4595e9, 5.7565e9, 5.7765e9, 5.7965e9, 5.8165e9]
			droneid_channel = [channel1]
		elif len(self.hackrf_list) == 2:
			channel2_1 = [2.4145e9, 2.4295e9, 5.7565e9, 5.7765e9]
			channel2_2 = [2.4445e9, 2.4595e9, 5.7965e9, 5.8165e9]
			droneid_channel = [channel2_1, channel2_2]
		elif len(self.hackrf_list) == 3:
			channel3_1 = [2.4145e9, 2.4295e9, 5.7565e9]
			channel3_2 = [2.4445e9, 5.7765e9, 5.8165e9]
			channel3_3 = [2.4595e9, 5.7965e9, 5.8165e9]
			droneid_channel = [channel3_1, channel3_2, channel3_3]
		elif len(self.hackrf_list) == 4:
			channel4_1 = [2.4145e9, 5.7565e9]
			channel4_2 = [2.4295e9, 5.7765e9]
			channel4_3 = [2.4445e9, 5.7965e9]
			channel4_4 = [2.4595e9, 5.8165e9]
			droneid_channel = [channel4_1, channel4_2, channel4_3, channel4_4]
		self.droneid_channel = droneid_channel

	def break_process_start(self):
		print("\nProgram interrupted by the user.")
		for hackrf in self.hackrf_list:
			hackrf.close()
		print("\nhackrf close successfully.")
		self.start_collect.close()
		print("\ncollect data worker close successfully.")


class CollectData:
	def __init__(
		self, 
		numprocesscut = 100
	):
		self.__worker_q = queue.Queue(30)
		self.__workerlist = []
		self.numprocesscut = numprocesscut

	def __worker(self, worker_idx, exit_event):
		while True:
			if exit_event.is_set():
				break
			while not exit_event.is_set() and self.__worker_q.qsize != 0:
				try:
					item = self.__worker_q.get(timeout=1)
					process_list_cut=[]
					for processcut1000_idx in range(self.numprocesscut):
						process_list_cut.append(threading.Thread(target=find_burst_sol4, args=(processcut1000_idx, item[0], item[1], self.precal_func_config, 
																				   self.numprocesscut, item[2], worker_idx)))
					for processcut1000_idx in range(self.numprocesscut):
						process_list_cut[processcut1000_idx].start()
					for processcut1000_idx in range(self.numprocesscut): 
						process_list_cut[processcut1000_idx].join()
				except:
					continue

	def initWorker(self, num_worker):
		start_collect.precal_func_config = pre_cal_func(fs = 15.36e6)   #sample frequency = 15.36e6 to solve DJI ocusync signal, DON'T CHANGE
		self.worker_exit_event = threading.Event()
		for worker_idx in range(num_worker):
			self.__workerlist.append(threading.Thread(target = self.__worker, args = (worker_idx, self.worker_exit_event)))
	
	def startWorker(self):
		for worker in self.__workerlist:
			worker.start()

	def realtimeLoop(self,samples,start_time, hackrf_num):
		self.__worker_q.put([samples, start_time, hackrf_num])

	def close(self):
		self.worker_exit_event.set()
		for worker in self.__workerlist:
			worker.join() 
		# print("Close")



if __name__ == '__main__':
	# mp.freeze_support()
	readfile = 0   #1則回放   0則即時運作
	if readfile == 1:
		findfirst = Findsignal()
		start_timestamp = datetime.datetime.now()
		file_to_read = "/home/iwave/mattaim/idBS/idBS/wintoubuntu/CRCtest_2024-01-05 15:30:07.877170.iq"
		findfirst.readfile(file_to_read)
		end = datetime.datetime.now()
		print("執行時間：%f 秒" % (end - start_timestamp).total_seconds())
	else:
		sweepANT = 0   #0=固定天線方向, 1=sweep ant
		sweep_video_channel = 0   #1=照ant_list順序判圖傳後才進入解drone id, 0=照ant_list順序切天線直接解droneid
		sweep_type = 2   #選取切天線判定機制    選1速度快誤判少(無Ocusync 4.0)    選2速度慢距離遠(無O4)    選3誤判多速度快(有O4)
		if sweepANT == 0:
			ant_list = [b'd0']
		else:
			ant_list = [b'd0', b'd1', b'd2', b'd3', b'd4', b'd5', b'd6', b'd7']   #天線方向記得設定
		start_collect = CollectData()
		start_collect.initWorker(num_worker = 5)
		start_collect.startWorker()
		try:
			Arduino1 = ArduinoControl("/dev/ttyACM0", 115200)
			findfirst = Findsignal(collectdata = start_collect, ant_list =  ant_list, Arduino = Arduino1)
			findfirst.arduino_init()
			time.sleep(1)
		except:
			findfirst = Findsignal(collectdata = start_collect, ant_list =  ant_list)
			sweep_video_channel = 0
		if sweep_video_channel == 1:
			findfirst.hackrf_init(hackrf_sample_rate = 20e6)
			findfirst.start_hackrf_sweep(sweep_type)
		else:
			findfirst.hackrf_init()
			findfirst.start_hackrf_nosweep()


