import numpy as np
import scipy
import matplotlib.pyplot as plt
# ff1,tt,zxxx1(10752, 1001)

def generate_scrambler_seq(num_bits):
	x1_init = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	x2_init = [0,0,0,1,1,1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,1,0,0]
	Mpn = num_bits
	Nc = 1600
	x1 = np.zeros(Nc + Mpn + 31)
	x2 = np.zeros(Nc + Mpn + 31)
	c = np.zeros(Mpn)
	x1[0:31] = x1_init
	x2[0:31] = x2_init
	for n in range(Mpn+Nc):
		x1[n+30] = (x1[n+2] + x1[n-1])%2
		x2[n+30] = (x2[n+2] + x2[n+1] +x2[n] + x2[n-1])%2
	for i in range(0,Mpn,1):
		c[i] = (x1[i+Nc]+x2[i+Nc])%2
	c = np.asarray(c,dtype = 'int')
	return c

def quantize_qpsk(data_carriers,i):
	quantized_bits = np.zeros(len(data_carriers)*2)
	#plt.figure(i)
	#if i == 2:
	#	plt.xlim(-2, 2)
	#	plt.ylim(-2, 2)
	#	plt.plot(np.real(data_carriers),np.imag(data_carriers),"o")
	#	plt.show()
	for idx in range(len(data_carriers)):
		sample = data_carriers[idx]
		if np.real(sample) > 0 and np.imag(sample) > 0:
			bits = [0, 0]
		elif np.real(sample) > 0 and np.imag(sample) < 0:
			bits = [0, 1]
		elif np.real(sample) < 0 and np.imag(sample) > 0:
			bits = [1, 0]
		elif np.real(sample) < 0 and np.imag(sample) < 0:	
			bits = [1, 1]
		else:
			bits = [0, 0]
		quantized_bits[2*idx:2*(idx+1)] = bits
	return quantized_bits

def get_data_carrier_indices(fft_size):
	data_carrier_count = 600
	dc =(fft_size/2)
	indice = np.arange(601)-300
	indice2=np.delete(indice,300)+int(fft_size / 2)
	return indice2

def slice7_LO(sliced_samples):
	Power_Mean6 = np.mean(sliced_samples[1175:1875])
	Power_Mean1 = np.mean(sliced_samples[2225:3275])
	Power_Mean2 = np.mean(sliced_samples[3275:4675])
	Power_Mean3 = np.mean(sliced_samples[4675:6075])
	Power_Mean4 = np.mean(sliced_samples[6075:7475])
	Power_Mean5 = np.mean(sliced_samples[7475:8525])
	Power_Mean7 = np.mean(sliced_samples[8875:9575])

	PMs12n = np.absolute(Power_Mean2-Power_Mean1-np.amin(sliced_samples[3275:4675])+np.amin(sliced_samples[2225:3275]))
	PMs23n = np.absolute(Power_Mean3-Power_Mean2-np.amin(sliced_samples[4675:6075])+np.amin(sliced_samples[3275:4675]))
	PMs34n = np.absolute(Power_Mean4-Power_Mean3-np.amin(sliced_samples[6075:7475])+np.amin(sliced_samples[4675:6075]))
	PMs45n = np.absolute(Power_Mean5-Power_Mean4-np.amin(sliced_samples[7475:8525])+np.amin(sliced_samples[6075:7475]))

	PMs1n = Power_Mean1 - np.amin(sliced_samples[2225:3275])
	PMs2n = Power_Mean2 - np.amin(sliced_samples[3275:4675])
	PMs3n = Power_Mean3 - np.amin(sliced_samples[4675:6075])
	PMs4n = Power_Mean4 - np.amin(sliced_samples[6075:7475])
	PMs5n = Power_Mean5 - np.amin(sliced_samples[7475:8525])
	
	if PMs12n > np.minimum(PMs1n,PMs2n):
		PMs1n = np.minimum(PMs1n,PMs2n)
		PMs2n = np.minimum(PMs1n,PMs2n)
	if PMs23n > np.minimum(PMs2n,PMs3n):
		PMs2n = np.minimum(PMs2n,PMs3n)
		PMs3n = np.minimum(PMs2n,PMs3n)
	if PMs34n > np.minimum(PMs3n,PMs4n):
		PMs3n = np.minimum(PMs3n,PMs4n)
		PMs4n = np.minimum(PMs3n,PMs4n)
	if PMs45n > np.minimum(PMs4n,PMs5n):
		PMs4n = np.minimum(PMs4n,PMs5n)
		PMs5n = np.minimum(PMs4n,PMs5n)
	if PMs34n > np.minimum(PMs3n,PMs4n):
		PMs3n = np.minimum(PMs3n,PMs4n)
		PMs4n = np.minimum(PMs3n,PMs4n)
	if PMs23n > np.minimum(PMs2n,PMs3n):
		PMs2n = np.minimum(PMs2n,PMs3n)
		PMs3n = np.minimum(PMs2n,PMs3n)
	if PMs12n > np.minimum(PMs1n,PMs2n):
		PMs1n = np.minimum(PMs1n,PMs2n)
		PMs2n = np.minimum(PMs1n,PMs2n)
	tmpPMABC = np.amax([PMs1n,PMs2n,PMs3n,PMs4n,PMs5n])

	slice7_score = PMs1n*PMs2n*PMs3n*PMs4n*PMs5n/(Power_Mean6**4)/(Power_Mean7**4)/tmpPMABC;
	return slice7_score

def zcsequence(fft_size, root = 600): 
	# root = 147
	zcseq = np.exp(-1j * np.pi * root * np.arange(601)*(np.arange(601)+1)/601)
	zcc = np.delete(zcseq, 300)
	indice = np.arange(601)-300
	indice2=np.delete(indice,300)+int(fft_size / 2)
	ZC_freq = np.zeros((fft_size),dtype=complex)
	ZC_freq[indice2] = zcc
	ZC = np.fft.ifft(np.fft.fftshift(ZC_freq))
	return ZC

def pre_cal_func(fs):
	FFT_SIZE = int(fs/15e3)
	long_cp_len = int(1/192000 * fs)
	short_cp_len = int(0.0000046875 * fs)
	cyclic_prefix_length_schedule = [long_cp_len, short_cp_len, short_cp_len, short_cp_len, short_cp_len, short_cp_len, short_cp_len, short_cp_len, long_cp_len]
	zc_seq_offset = (FFT_SIZE * 3) + long_cp_len + (short_cp_len * 3)
	data_carrier_indices = get_data_carrier_indices(FFT_SIZE)
	second_scrambler = generate_scrambler_seq(7200)  #7200=最後bit檔有幾個bit
	
	root = 1
	transform_filter_1_43008_1024 = precalculate_transform_filter(root, 43008, 1024)
	root = 301
	transform_filter_301_43008_1024 = precalculate_transform_filter(root, 43008, 1024)
	root = 600
	transform_filter_600_43008_1024 = precalculate_transform_filter(root, 43008, 1024)
	root = 600
	transform_filter_600_32256_1024 = precalculate_transform_filter(root, 32256, 1024)
	root = 600
	transform_filter_600_21504_1024 = precalculate_transform_filter(root, 43008, 1024)

	pre_cal_func = {
		"short_cp_len"                  : short_cp_len,
		"cyclic_prefix_length_schedule" : cyclic_prefix_length_schedule,
		"zc_seq_offset"                 : zc_seq_offset,
		"data_carrier_indices"          : data_carrier_indices,
		"second_scrambler"              : second_scrambler,
		"transform_filter": {
			"1_43008_1024"     : transform_filter_1_43008_1024,
			"301_43008_1024"   : transform_filter_301_43008_1024,
			"600_43008_1024"   : transform_filter_600_43008_1024,
			"600_32256_1024"   : transform_filter_600_32256_1024,
			"600_21504_1024"   : transform_filter_600_21504_1024,
		}
	}
	return pre_cal_func

def precalculate_transform_filter(root, N, fft_size):
	zcfilter = zcsequence(fft_size, root)
	zcfilter = zcfilter - np.mean(zcfilter)
	zcfilter_conj          = np.conj(zcfilter)
	zcfilter_conj_var      = np.var(zcfilter_conj)
	zcfilter_conj_var_sqrt = np.sqrt(zcfilter_conj_var)
	Filter_T = scipy.fft.fft(zcfilter_conj, n = N)		#n=N讓他自己補0拓寬到N個點
	dict_transform_filter = {
		"Filter_T" 				 : Filter_T,
		"zcfilter_conj" 		 : zcfilter_conj,
		"zcfilter_conj_var_sqrt" : zcfilter_conj_var_sqrt
	}
	return dict_transform_filter


def zcsequence_findsignal(fft_size,root):
    #10M
    zcseq = np.exp(-1j * np.pi * root * np.arange(601)*(np.arange(601)+1)/601)
    # plt.plot(np.imag(zcseq))
    # plt.show()
    data_carrier_count = 600
    zcc=np.delete(zcseq, 300)
    dc =(fft_size/2)
    indice = np.arange(601)-300
    indice2=np.delete(indice,300)+int(fft_size / 2)
    ZC_freq = np.zeros((fft_size),dtype=complex)
    ZC_freq[indice2] = zcc
    ZC = np.fft.ifft(np.fft.fftshift(ZC_freq))
    # plt.plot(zcc)
    # plt.show()
 ###############################################################################################3       
    #20M
    # zcseq = np.exp(-1j * np.pi * root * np.arange(1201)*(np.arange(1201)+1)/1201)
    # # plt.plot(np.imag(zcseq))
    # # plt.show()
    # data_carrier_count = 1200
    # zcc=np.delete(zcseq, 600)
    # dc =(fft_size/2)
    # indice = np.arange(1201)-600
    # indice2=np.delete(indice,600)+int(fft_size / 2)
    # ZC_freq = np.zeros((fft_size),dtype=complex)
    # ZC_freq[indice2] = zcc
    # ZC = np.fft.ifft(np.fft.fftshift(ZC_freq))
    return ZC

def zcsequence_findsignal20M(fft_size,root):
	root=1
	zcseq = np.exp(-1j * np.pi * root * np.arange(1201)*(np.arange(1201)+1)/1201)
	# plt.plot(np.imag(zcseq))
	# plt.show()
	data_carrier_count = 1200
	zcc=np.delete(zcseq, 600)
	dc =(2048/2)
	indice = np.arange(1201)-600
	indice2=np.delete(indice,600)+int(2048 / 2)
	ZC_freq = np.zeros((2048),dtype=complex)
	ZC_freq[indice2] = zcc
	# ZC = np.fft.ifft(np.fft.fftshift(ZC_freq))
	# ZCFF= np.fft.fftshift(np.fft.fft(ZC))
	#20M OF 10M
	ZC20_10FF=ZC_freq[512:512+1024]
	# ZC20_10FF=ZCFF[512:512+1024]
	ZC20_10TT = np.fft.ifft(np.fft.fftshift(ZC20_10FF))
	return ZC20_10TT

def precalculate_transform_filter_original(filternew,N):
    zcfilter=filternew
    zcfilter=zcfilter-np.mean(zcfilter)
    zcfilter_conj = np.conj(zcfilter)
    # zcfilter_conj =zcfilter
    zcfilter_conj_var = np.var(zcfilter_conj)
    zcfilter_conj_var_sqrt = np.sqrt(zcfilter_conj_var)
    Filter_F=scipy.fft.fftshift(scipy.fft.fft(zcfilter_conj,n=N))
    return Filter_F,zcfilter_conj,zcfilter_conj_var_sqrt

def precalculate_transform_filter_findsignal(root,N):
    fft_size=1024
    zcfilter=zcsequence_findsignal(fft_size,root)
    zcfilter=zcfilter-np.mean(zcfilter)
    zcfilter_conj = np.conj(zcfilter)
    zcfilter_conj_var = np.var(zcfilter_conj)
    zcfilter_conj_var_sqrt = np.sqrt(zcfilter_conj_var)
    Filter_T=scipy.fft.fftshift(scipy.fft.fft(zcfilter_conj,n=N))
    # Filter_T=scipy.fft.fft(zcfilter_conj,n=N)
    return Filter_T,zcfilter_conj,zcfilter_conj_var_sqrt



if __name__ == "__main__":
	precalculate_transform_filter(1,3068,1024)
