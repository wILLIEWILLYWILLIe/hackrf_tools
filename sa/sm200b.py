from .smdevice.sm_api import *
import numpy as np
from .sa_setting import SA_TRIGGER_TYPE

class SM200b:

    def __init__(self, serial_number:int):
        # connect the device
        self.__handle = sm_open_device_by_serial(int(serial_number))['device']

        # Refers to https://signalhound.com/sigdownloads/SDK/online_docs/sm_api/index.html#iqAcquisition -> "I/Q Sample Rates" section for sample rate table.
        self.__sample_rate = 250e6 # Fixed in segmented capture mode, see API instruction
        self.__bandwidth = 160e6 # Fixed in segmented capture mode, see API instruction
        self.__time_duration = None
        self.__ref_level = None
        self.__center_freq = None
        self.__trigger_type = None
        self.__ext_trig_timeout = None
        self.__iq_point_num = None

    def startCaptureIq(self):
        # Start the measurement, the device will begin looking for an external trigger or immediatelly receive.
        smSegIQCaptureStart(self.__handle, 0)

    def waitCaptureIq(self) -> bool:
        smSegIQCaptureWait(self.__handle, 0)
        is_timeout = sm_seg_IQ_capture_timeout(self.__handle, 0, 0)["timed_out"]
        if is_timeout:
            print("SA timeout")
            return False
        return True
    
    def readIq(self, iq_arr: np.ndarray):
        smSegIQCaptureRead(self.__handle, 0, 0, iq_arr, 0, self.__iq_point_num)
        smSegIQCaptureFinish(self.__handle, 0)

    def setRefLevel(self, ref_level: int):
        if ref_level == self.__ref_level:
            return
        sm_set_ref_level(self.__handle, ref_level)
        self.__ref_level = ref_level

    def setCenterFreq(self, center_freq: float):
        if center_freq == self.__center_freq:
            return
        sm_set_seg_IQ_center_freq(self.__handle, center_freq)
        self.__center_freq = center_freq

    def setTimeDuration(self, time_duration: float):
        self.__time_duration = time_duration
        self.__iq_point_num = int(self.__sample_rate*self.__time_duration)

    def setTrigger(self, trigger_type: SA_TRIGGER_TYPE, timeout: float):
        if (timeout == self.__ext_trig_timeout) and (trigger_type == self.__trigger_type):
            return
        if trigger_type == SA_TRIGGER_TYPE.IMMEDIATE:
            self.__setSegmentedCaptureMode(SM_TRIGGER_TYPE_IMM, 0)
        else:
            self.__setSegmentedCaptureMode(SM_TRIGGER_TYPE_EXT, timeout)
            if trigger_type == SA_TRIGGER_TYPE.RISING:
                sm_set_IQ_ext_trigger_edge(self.__handle, SM_TRIGGER_EDGE_RISING)
            else:
                sm_set_IQ_ext_trigger_edge(self.__handle, SM_TRIGGER_EDGE_FALLING)

        self.__trigger_type = trigger_type
        self.__ext_trig_timeout = timeout
        
    def __setSegmentedCaptureMode(self, sm_trigger_type, timeout:float):
        # Fix segment num to 1
        seg_num = 1
        sm_set_seg_IQ_segment_count(self.__handle, seg_num)
        sm_set_seg_IQ_segment(self.__handle, 0, sm_trigger_type, 0, self.__iq_point_num, timeout)
        sm_configure(self.__handle, SM_MODE_IQ_SEGMENTED_CAPTURE)
            
    def close(self):
        smAbort(self.__handle)
        smCloseDevice(self.__handle)
 