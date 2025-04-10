# -*- coding: utf-8 -*-

# Copyright (c) 2022 Signal Hound
# For licensing information, please see the API license in the software_licenses folder

from ctypes import CDLL
from ctypes import POINTER
from ctypes import byref
from ctypes import Structure
from ctypes import c_int
from ctypes import c_longlong
from ctypes import c_float
from ctypes import c_double
from ctypes import c_uint
from ctypes import c_ulonglong
from ctypes import c_ubyte
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_bool

import numpy
from sys import exit
import os

smlib = CDLL(f"{os.path.dirname(os.path.abspath(__file__))}/libsm_api.so")

# ---------------------------------- Defines -----------------------------------

SM_INVALID_HANDLE = -1

SM_TRUE = 1
SM_FALSE = 0

SM_MAX_DEVICES = 9

# For networked (10GbE devices)
SM_ADDR_ANY = b"0.0.0.0"
SM_DEFAULT_ADDR = b"192.168.2.10"
SM_DEFAULT_PORT = 51665

SM_AUTO_ATTEN = -1
# Valid atten values [0,6] or -1 for auto
SM_MAX_ATTEN = 6

SM_MAX_REF_LEVEL = 20.0

# Maximum number of sweeps that can be queued up
# Sweep indices [0,15]
SM_MAX_SWEEP_QUEUE_SZ = 16

SM200_MIN_FREQ = 100.0e3
SM200_MAX_FREQ = 20.6e9

SM435_MIN_FREQ = 100.0e3
SM435_MAX_FREQ = 44.2e9
SM435_MAX_FREQ_IF_OPT = 40.9e9

SM_MAX_IQ_DECIMATION = 4096

# The frequency at which the manually controlled preselector filters end.
# Past this frequency, the preselector filters are always enabled.
SM_PRESELECTOR_MAX_FREQ = 645.0e6

# Minimum RBW for fast sweep with Nuttall window
SM_FAST_SWEEP_MIN_RBW = 30.0e3

# Min/max span for device configured in RTSA measurement mode
SM_REAL_TIME_MIN_SPAN = 200.0e3
SM_REAL_TIME_MAX_SPAN = 160.0e6

# Sweep time range [1us, 100s]
SM_MIN_SWEEP_TIME = 1.0e-6
SM_MAX_SWEEP_TIME = 100.0

# Max number of bytes per SPI transfer
SM_SPI_MAX_BYTES = 4

# For GPIO sweeps
SM_GPIO_SWEEP_MAX_STEPS = 64

# For IQ GPIO switching
SM_GPIO_SWITCH_MAX_STEPS = 64
SM_GPIO_SWITCH_MIN_COUNT = 2
SM_GPIO_SWITCH_MAX_COUNT = 4194303 - 1 # 2^22 - 1

# FPGA internal temperature (Celsius)
# Returned from smGetDeviceDiagnostics()
SM_TEMP_WARNING = 95.0
SM_TEMP_MAX = 102.0

# Segmented I/Q captures, SM200B/SM435B only
SM_MAX_SEGMENTED_IQ_SEGMENTS = 250
SM_MAX_SEGMENTED_IQ_SAMPLES = 520e6

# IF output option devices only
SM435_IF_OUTPUT_FREQ = 1.5e9
SM435_IF_OUTPUT_MIN_FREQ = 24.0e9
SM435_IF_OUTPUT_MAX_FREQ = 43.5e9

SM_DATA_TYPE_32_FC = 0
SM_DATA_TYPE_16_SC = 1

SM_MODE_IDLE = 0
SM_MODE_SWEEPING = 1
SM_MODE_REAL_TIME = 2
SM_MODE_IQ_STREAMING = 3
SM_MODE_IQ_SEGMENTED_CAPTURE = 5
SM_MODE_IQ_SWEEP_LIST = 6
SM_MODE_AUDIO = 4
# Deprecated
SM_MODE_IQ = 3 # Use smModeIQStreaming

SM_SWEEP_SPEED_AUTO = 0
SM_SWEEP_SPEED_NORMAL = 1
SM_SWEEP_SPEED_FAST = 2

SM_IQ_STREAM_SAMPLE_RATE_NATIVE = 0
SM_IQ_STREAM_SAMPLE_RATE_LTE = 1

SM_POWER_STATE_ON = 0
SM_POWER_STATE_STANDBY = 1

SM_DETECTOR_AVERAGE = 0
SM_DETECTOR_MIN_MAX = 1

SM_SCALE_LOG = 0 # Sweep in dBm
SM_SCALE_LIN = 1 # Sweep in mV
SM_SCALE_FULL_SCALE = 2 # N/A

SM_VIDEO_LOG = 0
SM_VIDEO_VOLTAGE = 1
SM_VIDEO_POWER = 2
SM_VIDEO_SAMPLE = 3

SM_WINDOW_FLAT_TOP = 0
# 1 (N/A)
SM_WINDOW_NUTALL = 2
SM_WINDOW_BLACKMAN = 3
SM_WINDOW_HAMMING = 4
SM_WINDOW_GAUSSIAN_6_DB = 5
SM_WINDOW_RECT = 6

SM_TRIGGER_TYPE_IMM = 0
SM_TRIGGER_TYPE_VIDEO = 1
SM_TRIGGER_TYPE_EXT = 2
SM_TRIGGER_TYPE_FMT = 3

SM_TRIGGER_EDGE_RISING = 0
SM_TRIGGER_EDGE_FALLING = 1

SM_GPIO_STATE_OUTPUT = 0
SM_GPIO_STATE_INPUT = 1

SM_REFERENCE_USE_INTERNAL = 0
SM_REFERENCE_USE_EXTERNAL = 1

SM_DEVICE_TYPE_SM200A = 0
SM_DEVICE_TYPE_SM200B = 1
SM_DEVICE_TYPE_SM200C = 2
SM_DEVICE_TYPE_SM435B = 3
SM_DEVICE_TYPE_SM435C = 4

SM_AUDIO_TYPE_AM = 0
SM_AUDIO_TYPE_FM = 1
SM_AUDIO_TYPE_USB = 2
SM_AUDIO_TYPE_LSB = 3
SM_AUDIO_TYPE_CW = 4

SM_GPS_STATE_NOT_PRESENT = 0
SM_GPS_STATE_LOCKED = 1
SM_GPS_STATE_DISCIPLINED = 2

class SmGPIOStep(Structure):
    _fields_ = [("freq", c_double),
                ("mask", c_ubyte)] # gpio bits

# For troubleshooting purposes.
# (For standard diagnostics, use smGetDeviceDiagnostics)
class SmDeviceDiagnostics(Structure):
    _fields_ = [("voltage", c_float),
                ("currentInput", c_float),
                ("currentOCXO", c_float),
                ("current58", c_float),
                ("tempFPGAInternal", c_float),
                ("tempFPGANear", c_float),
                ("tempOCXO", c_float),
                ("tempVCO", c_float),
                ("tempRFBoardLO", c_float),
                ("tempPowerSupply", c_float)]


# --------------------------------- Mappings ----------------------------------

# USB devices only
smGetDeviceList = smlib.smGetDeviceList
smGetDeviceList.argtypes = [
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    POINTER(c_int)
]
smGetDeviceList2 = smlib.smGetDeviceList2
smGetDeviceList2.argtypes = [
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    POINTER(c_int)
]

# USB devices only
smOpenDevice = smlib.smOpenDevice
smOpenDeviceBySerial = smlib.smOpenDeviceBySerial

# Networked devices only
smOpenNetworkedDevice = smlib.smOpenNetworkedDevice

smCloseDevice = smlib.smCloseDevice
smPreset = smlib.smPreset
# Preset a device that has not been opened with the smOpenDevice functions
smPresetSerial = smlib.smPresetSerial

smNetworkedSpeedTest = smlib.smNetworkedSpeedTest

smGetDeviceInfo = smlib.smGetDeviceInfo
smGetFirmwareVersion = smlib.smGetFirmwareVersion
# SM435 only
smHasIFOutput = smlib.smHasIFOutput

smGetDeviceDiagnostics = smlib.smGetDeviceDiagnostics
smGetFullDeviceDiagnostics = smlib.smGetFullDeviceDiagnostics
# Networked devices only
smGetSFPDiagnostics = smlib.smGetSFPDiagnostics

smSetPowerState = smlib.smSetPowerState
smGetPowerState = smlib.smGetPowerState

# Overrides reference level when set to non-auto values
smSetAttenuator = smlib.smSetAttenuator
smGetAttenuator = smlib.smGetAttenuator

# Uses this when attenuation is automatic
smSetRefLevel = smlib.smSetRefLevel
smGetRefLevel = smlib.smGetRefLevel

# Set preselector state for all measurement modes
smSetPreselector = smlib.smSetPreselector
smGetPreselector = smlib.smGetPreselector

# Configure I/O routines
smSetGPIOState = smlib.smSetGPIOState
smGetGPIOState = smlib.smGetGPIOState
smWriteGPIOImm = smlib.smWriteGPIOImm
smReadGPIOImm = smlib.smReadGPIOImm
smWriteSPI = smlib.smWriteSPI
# For standard sweeps only
smSetGPIOSweepDisabled = smlib.smSetGPIOSweepDisabled
smSetGPIOSweep = smlib.smSetGPIOSweep
smSetGPIOSweep.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(SmGPIOStep, ndim=1, flags='C'),
    c_int
]
# For IQ streaming only
smSetGPIOSwitchingDisabled = smlib.smSetGPIOSwitchingDisabled
smSetGPIOSwitching = smlib.smSetGPIOSwitching
smSetGPIOSwitching.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(c_ubyte, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(c_uint, ndim=1, flags='C'),
    c_int
]

# Enable the external reference out port
smSetExternalReference = smlib.smSetExternalReference
smGetExternalReference = smlib.smGetExternalReference
# Specify whether to use the internal reference or reference on the ref in port
smSetReference = smlib.smSetReference
smGetReference = smlib.smGetReference

# Enable whether or not the API auto updates the timebase calibration
# value when a valid GPS lock is acquired.
smSetGPSTimebaseUpdate = smlib.smSetGPSTimebaseUpdate
smGetGPSTimebaseUpdate = smlib.smGetGPSTimebaseUpdate
smGetGPSHoldoverInfo = smlib.smGetGPSHoldoverInfo

# Returns whether the GPS is locked, can be called anytime
smGetGPSState = smlib.smGetGPSState

smSetSweepSpeed = smlib.smSetSweepSpeed
smSetSweepCenterSpan = smlib.smSetSweepCenterSpan
smSetSweepStartStop = smlib.smSetSweepStartStop
smSetSweepCoupling = smlib.smSetSweepCoupling
smSetSweepDetector = smlib.smSetSweepDetector
smSetSweepScale = smlib.smSetSweepScale
smSetSweepWindow = smlib.smSetSweepWindow
smSetSweepSpurReject = smlib.smSetSweepSpurReject

smSetRealTimeCenterSpan = smlib.smSetRealTimeCenterSpan
smSetRealTimeRBW = smlib.smSetRealTimeRBW
smSetRealTimeDetector = smlib.smSetRealTimeDetector
smSetRealTimeScale = smlib.smSetRealTimeScale
smSetRealTimeWindow = smlib.smSetRealTimeWindow

smSetIQBaseSampleRate = smlib.smSetIQBaseSampleRate
smSetIQDataType = smlib.smSetIQDataType
smSetIQCenterFreq = smlib.smSetIQCenterFreq
smGetIQCenterFreq = smlib.smGetIQCenterFreq
smSetIQSampleRate = smlib.smSetIQSampleRate
smSetIQBandwidth = smlib.smSetIQBandwidth
smSetIQExtTriggerEdge = smlib.smSetIQExtTriggerEdge
smSetIQTriggerSentinel = smlib.smSetIQTriggerSentinel
# Please read the API manual before using this function.
smSetIQQueueSize = smlib.smSetIQQueueSize

smSetIQSweepListDataType = smlib.smSetIQSweepListDataType
smSetIQSweepListCorrected = smlib.smSetIQSweepListCorrected
smSetIQSweepListSteps = smlib.smSetIQSweepListSteps
smGetIQSweepListSteps = smlib.smGetIQSweepListSteps
smSetIQSweepListFreq = smlib.smSetIQSweepListFreq
smSetIQSweepListRef = smlib.smSetIQSweepListRef
smSetIQSweepListAtten = smlib.smSetIQSweepListAtten
smSetIQSweepListSampleCount = smlib.smSetIQSweepListSampleCount

# Segmented I/Q configuration, SM200B/SM435B only
smSetSegIQDataType = smlib.smSetSegIQDataType
smSetSegIQCenterFreq = smlib.smSetSegIQCenterFreq
smSetSegIQVideoTrigger = smlib.smSetSegIQVideoTrigger
smSetSegIQExtTrigger = smlib.smSetSegIQExtTrigger
smSetSegIQFMTParams = smlib.smSetSegIQFMTParams
smSetSegIQFMTParams.argtypes = [
    c_int, c_int,
    numpy.ctypeslib.ndpointer(c_double, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(c_double, ndim=1, flags='C'),
    c_int
]
smSetSegIQSegmentCount = smlib.smSetSegIQSegmentCount
smSetSegIQSegment = smlib.smSetSegIQSegment

smSetAudioCenterFreq = smlib.smSetAudioCenterFreq
smSetAudioType = smlib.smSetAudioType
smSetAudioFilters = smlib.smSetAudioFilters
smSetAudioFMDeemphasis = smlib.smSetAudioFMDeemphasis

smConfigure = smlib.smConfigure
smGetCurrentMode = smlib.smGetCurrentMode
smAbort = smlib.smAbort

smGetSweepParameters = smlib.smGetSweepParameters
smGetRealTimeParameters = smlib.smGetRealTimeParameters
# Retrieve I/Q parameters for streaming and segmented I/Q captures.
smGetIQParameters = smlib.smGetIQParameters
# Retrieve the correction factor/scalar for streaming and segmented I/Q captures.
smGetIQCorrection = smlib.smGetIQCorrection
# Retrieve correction factors for an I/Q sweep list measurement.
smIQSweepListGetCorrections = smlib.smIQSweepListGetCorrections
smIQSweepListGetCorrections.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C')
]

smSegIQGetMaxCaptures = smlib.smSegIQGetMaxCaptures

# Performs a single sweep, blocking function
smGetSweep = smlib.smGetSweep
smGetSweep.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    POINTER(c_longlong)
]

# Queued sweep mechanisms
smStartSweep = smlib.smStartSweep
smFinishSweep = smlib.smFinishSweep
smFinishSweep.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    POINTER(c_int),
    POINTER(c_longlong)
]

smGetRealTimeFrame = smlib.smGetRealTimeFrame
smGetRealTimeFrame.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    POINTER(c_int),
    POINTER(c_longlong)
]

smGetIQ = smlib.smGetIQ
smGetIQ.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.complex64, ndim=1, flags='C'),
    c_int,
    numpy.ctypeslib.ndpointer(c_double, ndim=1, flags='C'),
    c_int, POINTER(c_longlong), c_int, POINTER(c_int), POINTER(c_int)
]

smIQSweepListGetSweep = smlib.smIQSweepListGetSweep
smIQSweepListGetSweep.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(c_longlong, ndim=1, flags='C')
]
smIQSweepListStartSweep = smlib.smIQSweepListStartSweep
smIQSweepListStartSweep.argtypes = [
    c_int, c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(c_longlong, ndim=1, flags='C')
]
smIQSweepListFinishSweep = smlib.smIQSweepListFinishSweep

# Segmented I/Q acquisition functions, SM200B/SM435B only
smSegIQCaptureStart = smlib.smSegIQCaptureStart
smSegIQCaptureWait = smlib.smSegIQCaptureWait
smSegIQCaptureWaitAsync = smlib.smSegIQCaptureWaitAsync
smSegIQCaptureTimeout = smlib.smSegIQCaptureTimeout
smSegIQCaptureTime = smlib.smSegIQCaptureTime
smSegIQCaptureRead = smlib.smSegIQCaptureRead
smSegIQCaptureRead.argtypes = [
    c_int, c_int, c_int,
    numpy.ctypeslib.ndpointer(numpy.complex64, ndim=1, flags='C'),
    c_int, c_int
]
smSegIQCaptureFinish = smlib.smSegIQCaptureFinish
smSegIQCaptureFull = smlib.smSegIQCaptureFull
smSegIQCaptureFull.argtypes = [
    c_int, c_int,
    numpy.ctypeslib.ndpointer(numpy.complex64, ndim=1, flags='C'),
    c_int, c_int, POINTER(c_longlong), POINTER(c_int)
]
# Convenience function to resample a 250 MS/s capture to the LTE rate of 245.76 MS/s
smSegIQLTEResample = smlib.smSegIQLTEResample
smSegIQLTEResample.argtypes = [
    numpy.ctypeslib.ndpointer(c_float, ndim=1, flags='C'),
    c_int,
    numpy.ctypeslib.ndpointer(c_float, ndim=1, flags='C'),
    POINTER(c_int), c_bool
]

smSetIQFullBandAtten = smlib.smSetIQFullBandAtten
smSetIQFullBandCorrected = smlib.smSetIQFullBandCorrected
smSetIQFullBandSamples = smlib.smSetIQFullBandSamples
smSetIQFullBandTriggerType = smlib.smSetIQFullBandTriggerType
smSetIQFullBandVideoTrigger = smlib.smSetIQFullBandVideoTrigger
smSetIQFullBandTriggerTimeout = smlib.smSetIQFullBandTriggerTimeout
smGetIQFullBand = smlib.smGetIQFullBand
smGetIQFullBand.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    c_int
]
smGetIQFullBandSweep = smlib.smGetIQFullBandSweep
smGetIQFullBandSweep.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    c_int, c_int, c_int
]

smGetAudio = smlib.smGetAudio
smGetAudio.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(c_float, ndim=1, flags='C')
]

smGetGPSInfo = smlib.smGetGPSInfo
smGetGPSInfo.argtypes = [
    c_int, c_int, POINTER(c_int), POINTER(c_longlong),
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
    numpy.ctypeslib.ndpointer(c_char, ndim=1, flags='C'),
    POINTER(c_int)
]

# Device must have GPS write capability. See API manual for more information.
smWriteToGPS = smlib.smWriteToGPS
smWriteToGPS.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(c_ubyte, ndim=1, flags='C'),
    c_int
]

# Accepts values between [10-90] as the temp threshold for when the fan turns on
smSetFanThreshold = smlib.smSetFanThreshold
smGetFanThreshold = smlib.smGetFanThreshold

# Must be SM435 device with IF output option
smSetIFOutput = smlib.smSetIFOutput

smGetCalDate = smlib.smGetCalDate

# Configure 10GbE network parameters via the network
smBroadcastNetworkConfig = smlib.smBroadcastNetworkConfig
# Configure 10GbE network parameters via USB
# Device handle for these functions cannot be used with others
smNetworkConfigGetDeviceList = smlib.smNetworkConfigGetDeviceList
smNetworkConfigGetDeviceList.argtypes = [
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    POINTER(c_int)
]
smNetworkConfigOpenDevice = smlib.smNetworkConfigOpenDevice
smNetworkConfigCloseDevice = smlib.smNetworkConfigCloseDevice
smNetworkConfigGetMAC = smlib.smNetworkConfigGetMAC
smNetworkConfigSetIP = smlib.smNetworkConfigSetIP
smNetworkConfigGetIP = smlib.smNetworkConfigGetIP
smNetworkConfigSetPort = smlib.smNetworkConfigSetPort
smNetworkConfigGetPort = smlib.smNetworkConfigGetPort

smGetAPIVersion = smlib.smGetAPIVersion
smGetAPIVersion.restype = c_char_p
smGetErrorString = smlib.smGetErrorString
smGetErrorString.restype = c_char_p
smGetProductID = smlib.smGetProductID
smGetProductID.restype = c_char_p


# ---------------------------------- Utility ----------------------------------

def error_check(func):
    def print_status_if_error(*args, **kwargs):
        return_vars = func(*args, **kwargs)
        if "status" not in return_vars.keys():
            return return_vars
        status = return_vars["status"]
        if status != 0:
            print (f"{'Error' if status < 0 else 'Warning'} {status}: {sm_get_error_string(status)['error_string']} in {func.__name__}()")
        if status < 0:
            exit()
        return return_vars
    return print_status_if_error

def to_sm_bool(b):
    return SM_TRUE if b is True else SM_FALSE

def from_sm_bool(sm_b):
    return True if sm_b is SM_TRUE else False


# --------------------------------- Functions ---------------------------------

@error_check
def sm_get_device_list():
    serials = numpy.zeros(SM_MAX_DEVICES).astype(c_int)
    device_count = c_int(-1)

    status = smGetDeviceList(serials, byref(device_count))

    return {
        "status": status,
        "serials": serials,
        "device_count": device_count.value
    }

@error_check
def sm_get_device_list2():
    serials = numpy.zeros(SM_MAX_DEVICES).astype(c_int)
    device_types = numpy.zeros(SM_MAX_DEVICES).astype(c_int)
    device_count = c_int(-1)

    status = smGetDeviceList2(serials, device_types, byref(device_count))

    return {
        "status": status,
        "serials": serials,
        "device_types": device_types,
        "device_count": device_count.value
    }

@error_check
def sm_open_device():
    device = c_int(-1)

    status = smOpenDevice(byref(device))

    return {
        "status": status,
        "device": device.value
    }

@error_check
def sm_open_device_by_serial(serial_number):
    device = c_int(-1)

    status = smOpenDeviceBySerial(byref(device), serial_number)

    return {
        "status": status,
        "device": device.value
    }

@error_check
def sm_open_networked_device(host_addr, device_addr, port):
    device = c_int(-1)

    status = smOpenNetworkedDevice(byref(device), host_addr, device_addr, port)

    return {
        "status": status,
        "device": device.value
    }

@error_check
def sm_close_device(device):
    return {
        "status": smCloseDevice(device)
    }

@error_check
def sm_preset(device):
    return {
        "status": smPreset(device)
    }

@error_check
def sm_preset_serial(serial_number):
    return {
        "status": smPreset(serial_number)
    }

@error_check
def sm_networked_speed_test(device, duration_seconds):
    bytes_per_second = c_double(-1)

    status = smNetworkedSpeedTest(device, c_double(duration_seconds), byref(bytes_per_second))

    return {
        "status": status,
        "bytes_per_second": bytes_per_second.value
    }

@error_check
def sm_get_device_info(device):
    device_type = c_int(-1)
    serial_number = c_int(-1)

    status = smGetDeviceInfo(device, byref(device_type), byref(serial_number))

    return {
        "status": status,
        "device_type": device_type.value,
        "serial_number": serial_number.value
    }

@error_check
def sm_get_firmware_version(device):
    major = c_int(-1)
    minor = c_int(-1)
    revision = c_int(-1)

    status = smGetFirmwareVersion(device, byref(major), byref(minor), byref(revision))

    return {
        "status": status,
        "major": major.value,
        "minor": minor.value,
        "revision": revision.value
    }

@error_check
def sm_has_IF_output(device):
    present = c_int(-1)

    status = smHasIFOutput(device, byref(present))

    return {
        "status": status,
        "present": from_sm_bool(present.value)
    }

@error_check
def sm_get_device_diagnostics(device):
    voltage = c_float(-1)
    current = c_float(-1)
    temperature = c_float(-1)

    status = smGetDeviceDiagnostics(device, byref(voltage), byref(current),
                                    byref(temperature))

    return {
        "status": status,
        "voltage": voltage.value,
        "current": current.value,
        "temperature": temperature.value
    }

@error_check
def sm_get_full_device_diagnostics(device):
    diagnostics = SmDeviceDiagnostics()

    status = smGetFullDeviceDiagnostics(device, byref(diagnostics))

    return {
        "status": status,
        "diagnostics": diagnostics
    }

@error_check
def sm_get_SFP_diagnostics(device):
    temp = c_float(-1)
    voltage = c_float(-1)
    tx_power = c_float(-1)
    rx_power = c_float(-1)

    status = smGetSFPDiagnostics(device, byref(temp), byref(voltage),
                                 byref(tx_power), byref(rx_power))

    return {
        "status": status,
        "temp": temp.value,
        "voltage": voltage.value,
        "tx_power": tx_power.value,
        "rx_power": rx_power.value
    }

@error_check
def sm_set_power_state(device, power_state):
    return {
        "status": smSetPowerState(device, power_state)
    }

@error_check
def sm_get_power_state(device):
    power_state = c_int(-1)

    status = smGetPowerState(device, byref(power_state))

    return {
        "status": status,
        "power_state": power_state.value
    }

@error_check
def sm_set_attenuator(device, atten):
    return {
        "status": smSetAttenuator(device, atten)
    }

@error_check
def sm_get_attenuator(device):
    atten = c_int(-1)

    status = smGetAttenuator(device, byref(atten))

    return {
        "status": status,
        "atten": atten.value
    }

@error_check
def sm_set_ref_level(device, ref_level):
    return {
        "status": smSetRefLevel(device, c_double(ref_level))
    }

@error_check
def sm_get_ref_level(device):
    ref_level = c_double(-1)

    status = smGetRefLevel(device, byref(ref_level))

    return {
        "status": status,
        "ref_level": ref_level.value
    }

@error_check
def sm_set_preselector(device, enabled):
    return {
        "status": smSetPreselector(device, to_sm_bool(enabled))
    }

@error_check
def sm_get_preselector(device):
    enabled_smb = c_int(-1)

    status = smGetPreselector(device, byref(enabled_smb))

    return {
        "status": status,
        "enabled": from_sm_bool(enabled_smb.value)
    }

@error_check
def sm_set_GPIO_state(device, lower_state, upper_state):
    return {
        "status": smSetGPIOState(device, lower_state, upper_state)
    }

@error_check
def sm_get_GPIO_state(device):
    lower_state = c_int(-1)
    upper_state = c_int(-1)

    status = smGetGPIOState(device, byref(lower_state), byref(upper_state))

    return {
        "status": status,
        "lower_state": lower_state.value,
        "upper_state": upper_state.value
    }

@error_check
def sm_write_GPIO_imm(device, data):
    return {
        "status": smWriteGPIOImm(device, data)
    }

@error_check
def sm_read_GPIO_imm(device):
    data = c_ubyte(-1)

    status = smReadGPIOImm(device, byref(data))

    return {
        "status": status,
        "data": data.value
    }

@error_check
def sm_write_SPI(device, data, byte_count):
    return {
        "status": smWriteSPI(device, data, byte_count)
    }

@error_check
def sm_set_GPIO_sweep_disabled(device):
    return {
        "status": smSetGPIOSweepDisabled(device)
    }

@error_check
def sm_set_GPIO_sweep(device, steps, step_count):
    return {
        "status": smSetGPIOSweep(device, steps, step_count)
    }

@error_check
def sm_set_GPIO_switching_disabled(device):
    return {
        "status": smSetGPIOSwitchingDisabled(device)
    }

@error_check
def sm_set_GPIO_switching(device, gpio, counts, gpio_steps):
    return {
        "status": smSetGPIOSwitching(device, gpio, counts, gpio_steps)
    }

@error_check
def sm_set_external_reference(device, enabled):
    return {
        "status": smSetExternalReference(device, enabled)
    }

@error_check
def sm_get_external_reference(device):
    enabled_smb = c_int(-1)

    status = smGetExternalReference(device, byref(enabled_smb))

    return {
        "status": status,
        "enabled": from_sm_bool(enabled_smb.value)
    }

@error_check
def sm_set_reference(device, reference):
    return {
        "status": smSetReference(device, reference)
    }

@error_check
def sm_get_reference(device):
    reference = c_int(-1)

    status = smGetReference(device, byref(reference))

    return {
        "status": status,
        "reference": reference.value
    }

@error_check
def sm_set_GPS_timebase_update(device, enabled):
    return {
        "status": smSetGPSTimebaseUpdate(device, enabled)
    }

@error_check
def sm_get_GPS_timebase_update(device):
    enabled_smb = c_int(-1)

    status = smGetGPSTimebaseUpdate(device, byref(enabled_smb))

    return {
        "status": status,
        "enabled": from_sm_bool(enabled_smb.value)
    }

@error_check
def sm_get_GPS_holdover_info(device):
    using_GPS_holdover_smb = c_int(-1)
    last_holdover_time = c_ulonglong(-1)

    status = smGetGPSHoldoverInfo(device, byref(using_GPS_holdover_smb),
                                  byref(last_holdover_time))

    return {
        "status": status,
        "using_GPS_holdover": from_sm_bool(using_GPS_holdover_smb.value),
        "last_holdover_time": last_holdover_time.value
    }

@error_check
def sm_get_GPS_state(device):
    GPS_state = c_int(-1)

    status = smGetGPSState(device, byref(GPS_state))

    return {
        "status": status,
        "GPS_state": GPS_state.value
    }

@error_check
def sm_set_sweep_speed(device, sweep_speed):
    return {
        "status": smSetSweepSpeed(device, sweep_speed)
    }

@error_check
def sm_set_sweep_center_span(device, center_freq_Hz, span_Hz):
    return {
        "status": smSetSweepCenterSpan(device,
                                       c_double(center_freq_Hz),
                                       c_double(span_Hz))
    }

@error_check
def sm_set_sweep_start_stop(device, start_freq_Hz, stop_freq_Hz):
    return {
        "status": smSetSweepStartStop(device,
                                      c_double(start_freq_Hz),
                                      c_double(stop_freq_Hz))
    }

@error_check
def sm_set_sweep_coupling(device, rbw, vbw, sweep_time):
    return {
        "status": smSetSweepCoupling(device,
                                     c_double(rbw), c_double(vbw),
                                     c_double(sweep_time))
    }

@error_check
def sm_set_sweep_detector(device, detector, video_units):
    return {
        "status": smSetSweepDetector(device, detector, video_units)
    }

@error_check
def sm_set_sweep_scale(device, scale):
    return {
        "status": smSetSweepScale(device, scale)
    }

@error_check
def sm_set_sweep_window(device, window):
    return {
        "status": smSetSweepWindow(device, window)
    }

@error_check
def sm_set_sweep_spur_reject(device, spur_reject_enabled):
    return {
        "status": smSetSweepSpurReject(device, spur_reject_enabled)
    }

@error_check
def sm_set_real_time_center_span(device, center_freq_Hz, span_Hz):
    return {
        "status": smSetRealTimeCenterSpan(device,
                                          c_double(center_freq_Hz),
                                          c_double(span_Hz))
    }

@error_check
def sm_set_real_time_RBW(device, rbw):
    return {
        "status": smSetRealTimeRBW(device, c_double(rbw))
    }

@error_check
def sm_set_real_time_detector(device, detector):
    return {
        "status": smSetRealTimeDetector(device, detector)
    }

@error_check
def sm_set_real_time_scale(device, scale, frame_ref, frame_scale):
    return {
        "status": smSetRealTimeScale(device, scale,
                                     c_double(frame_ref), c_double(frame_scale))
    }

@error_check
def sm_set_real_time_window(device, window):
    return {
        "status": smSetRealTimeWindow(device, window)
    }

@error_check
def sm_set_IQ_base_sample_rate(device, sample_rate):
    return {
        "status": smSetIQBaseSampleRate(device, sample_rate)
    }

@error_check
def sm_set_IQ_data_type(device, data_type):
    return {
        "status": smSetIQDataType(device, data_type)
    }

@error_check
def sm_set_IQ_center_freq(device, center_freq_Hz):
    return {
        "status": smSetIQCenterFreq(device, c_double(center_freq_Hz))
    }

@error_check
def sm_get_IQ_center_freq(device):
    center_freq_Hz = c_double(-1)

    status = smGetIQCenterFreq(device, byref(center_freq_Hz))

    return {
        "status": status,
        "center_freq_Hz": center_freq_Hz.value
    }

@error_check
def sm_set_IQ_sample_rate(device, decimation):
    return {
        "status": smSetIQSampleRate(device, decimation)
    }

@error_check
def sm_set_IQ_bandwidth(device, enable_software_filter, bandwidth):
    return {
        "status": smSetIQBandwidth(device,
                                   enable_software_filter,
                                   c_double(bandwidth))
    }

@error_check
def sm_set_IQ_ext_trigger_edge(device, edge):
    return {
        "status": smSetIQExtTriggerEdge(device, edge)
    }

@error_check
def sm_set_IQ_queue_size(device, ms):
    return {
        "status": smSetIQQueueSize(device, c_float(ms))
    }

@error_check
def sm_set_IQ_sweep_list_data_type(device, dataType):
    return {
        "status": smSetIQSweepListDataType(device, dataType)
    }

@error_check
def sm_set_IQ_sweep_list_corrected(device, corrected):
    return {
        "status": smSetIQSweepListCorrected(device, corrected)
    }

@error_check
def sm_set_IQ_sweep_list_steps(device, steps):
    return {
        "status": smSetIQSweepListSteps(device, steps)
    }

@error_check
def sm_get_IQ_sweep_list_steps(device):
    steps = c_int(-1)

    status = smGetIQSweepListSteps(device, byref(steps))
    return {
        "status": status,
        "steps": steps.value
    }

@error_check
def sm_set_IQ_sweep_list_freq(device, step, freq):
    return {
        "status": smSetIQSweepListFreq(device, step, c_double(freq))
    }

@error_check
def sm_set_IQ_sweep_list_ref(device, step, level):
    return {
        "status": smSetIQSweepListRef(device, step, c_double(level))
    }

@error_check
def sm_set_IQ_sweep_list_atten(device, step, atten):
    return {
        "status": smSetIQSweepListAtten(device, step, atten)
    }

@error_check
def sm_set_IQ_sweep_list_sample_count(device, step, samples):
    return {
        "status": smSetIQSweepListSampleCount(device, step, samples)
    }

@error_check
def sm_set_seg_IQ_data_type(device, data_type):
    return {
        "status": smSetSegIQDataType(device, data_type)
    }

@error_check
def sm_set_seg_IQ_center_freq(device, center_freq_Hz):
    return {
        "status": smSetSegIQCenterFreq(device, c_double(center_freq_Hz))
    }

@error_check
def sm_set_seg_IQ_video_trigger(device, trigger_level, trigger_edge):
    return {
        "status": smSetSegIQVideoTrigger(device, c_double(trigger_level), trigger_edge)
    }

@error_check
def sm_set_seg_IQ_ext_trigger(device, ext_trigger_edge):
    return {
        "status": smSetSegIQExtTrigger(device, ext_trigger_edge)
    }

@error_check
def sm_set_seg_IQ_FMT_params(device, fft_size, frequencies, ampls, count):
    return {
        "status": smSetSegIQFMTParams(device, fft_size, frequencies, ampls, count)
    }

@error_check
def sm_set_seg_IQ_segment_count(device, segment_count):
    return {
        "status": smSetSegIQSegmentCount(device, segment_count)
    }

@error_check
def sm_set_seg_IQ_segment(device, segment, trigger_type, pre_trigger,
                          capture_size, timeout_seconds):
    return {
        "status": smSetSegIQSegment(device, segment, trigger_type, pre_trigger,
                                    capture_size, c_double(timeout_seconds))
    }

@error_check
def sm_set_audio_center_freq(device, center_freq_Hz):
    return {
        "status": smSetAudioCenterFreq(device, c_double(center_freq_Hz))
    }

@error_check
def sm_set_audio_type(device, audio_type):
    return {
        "status": smSetAudioType(device, audio_type)
    }

@error_check
def sm_set_audio_filters(device, if_bandwidth, audio_lpf, audio_hpf):
    return {
        "status": smSetAudioFilters(device, c_double(if_bandwidth),
                                    c_double(audio_lpf), c_double(audio_hpf))
    }

@error_check
def sm_set_audio_FM_deemphasis(device, deemphasis):
    return {
        "status": smSetAudioFMDeemphasis(device, c_double(deemphasis))
    }

@error_check
def sm_configure(device, mode):
    return {
        "status": smConfigure(device, mode)
    }

@error_check
def sm_get_current_mode(device):
    mode = c_int(-1)

    status = smGetCurrentMode(device, byref(mode))

    return {
        "status": status,
        "mode": mode.value
    }

@error_check
def sm_abort(device):
    return {
        "status": smAbort(device)
    }

@error_check
def sm_get_sweep_parameters(device):
    actual_RBW = c_double(-1)
    actual_VBW = c_double(-1)
    actual_start_freq = c_double(-1)
    bin_size = c_double(-1)
    sweep_size = c_int(-1)

    status = smGetSweepParameters(device, byref(actual_RBW), byref(actual_VBW),
                                  byref(actual_start_freq), byref(bin_size),
                                  byref(sweep_size))

    return {
        "status": status,
        "actual_RBW": actual_RBW.value,
        "actual_VBW": actual_VBW.value,
        "actual_start_freq": actual_start_freq.value,
        "bin_size": bin_size.value,
        "sweep_size": sweep_size.value
    }

@error_check
def sm_get_real_time_parameters(device):
    actual_RBW = c_double(-1)
    sweep_size = c_int(-1)
    actual_start_freq = c_double(-1)
    bin_size = c_double(-1)
    frame_width = c_int(-1)
    frame_height = c_int(-1)
    poi = c_double(-1)

    status = smGetRealTimeParameters(device,
                                     byref(actual_RBW),
                                     byref(sweep_size),
                                     byref(actual_start_freq),
                                     byref(bin_size),
                                     byref(frame_width),
                                     byref(frame_height),
                                     byref(poi))

    return {
        "status": status,
        "actual_RBW": actual_RBW.value,
        "sweep_size": sweep_size.value,
        "actual_start_freq": actual_start_freq.value,
        "bin_size": bin_size.value,
        "frame_width": frame_width.value,
        "frame_height": frame_height.value,
        "poi": poi.value
    }

@error_check
def sm_get_IQ_parameters(device):
    sample_rate = c_double(-1)
    bandwidth = c_double(-1)

    status = smGetIQParameters(device,
                               byref(sample_rate),
                               byref(bandwidth))

    return {
        "status": status,
        "sample_rate": sample_rate.value,
        "bandwidth": bandwidth.value
    }

@error_check
def sm_get_IQ_correction(device):
    scale = c_float(-1)

    status = smGetIQCorrection(device, byref(scale))

    return {
        "status": status,
        "scale": scale.value
    }

@error_check
def sm_IQ_sweep_list_get_corrections(device):
    ret = sm_get_IQ_sweep_list_steps(device)
    if ret["status"] != 0:
        return {
            "status": ret["status"]
        }
    steps = ret["steps"]

    corrections = numpy.zeros(steps).astype(numpy.float32)

    status = smIQSweepListGetCorrections(device, corrections)

    return {
        "status": status,
        "corrections": corrections
    }

@error_check
def sm_seg_IQ_get_max_captures(device):
    max_captures = c_int(-1)

    status = smSegIQGetMaxCaptures(device, byref(max_captures))

    return {
        "status": status,
        "max_captures": max_captures.value
    }

@error_check
def sm_get_sweep(device):
    ret = sm_get_sweep_parameters(device)
    if ret["status"] != 0:
        return {
            "status": ret["status"]
        }
    sweep_size = ret["sweep_size"]

    sweep_min = numpy.zeros(sweep_size).astype(numpy.float32)
    sweep_max = numpy.zeros(sweep_size).astype(numpy.float32)
    ns_since_epoch = c_longlong(-1)

    status = smGetSweep(device, sweep_min, sweep_max, byref(ns_since_epoch))

    return {
        "status": status,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "ns_since_epoch": ns_since_epoch.value
    }

@error_check
def sm_start_sweep(device, pos):
    return {
        "status": smStartSweep(device, pos)
    }

@error_check
def sm_finish_sweep(device, pos):
    ret = sm_get_sweep_parameters(device)
    if ret["status"] != 0:
        return {
            "status": ret["status"]
        }
    sweep_size = ret["sweep_size"]

    sweep_min = numpy.zeros(sweep_size).astype(numpy.float32)
    sweep_max = numpy.zeros(sweep_size).astype(numpy.float32)
    ns_since_epoch = c_longlong(-1)

    status = smFinishSweep(device, pos,
                           sweep_min, sweep_max,
                           byref(ns_since_epoch))

    return {
        "status": status,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "ns_since_epoch": ns_since_epoch.value
    }

@error_check
def sm_get_real_time_frame(device):
    ret = sm_get_real_time_parameters(device)
    if ret["status"] != 0:
        return {
            "status": ret["status"]
        }
    sweep_size = ret["sweep_size"]
    frame_width = ret["frame_width"]
    frame_height = ret["frame_height"]

    color_frame = numpy.zeros(frame_width * frame_height).astype(numpy.float32)
    alpha_frame = numpy.zeros(frame_width * frame_height).astype(numpy.float32)
    sweep_min = numpy.zeros(sweep_size).astype(numpy.float32)
    sweep_max = numpy.zeros(sweep_size).astype(numpy.float32)
    frame_count = c_int(-1)
    ns_since_epoch = c_longlong(-1)

    status = smGetRealTimeFrame(device, color_frame, alpha_frame,
                                sweep_min, sweep_max,
                                byref(frame_count), byref(ns_since_epoch))
    return {
        "status": status,
        "color_frame": color_frame,
        "alpha_frame": alpha_frame,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "frame_count": frame_count.value,
        "ns_since_epoch": ns_since_epoch.value
    }

@error_check
def sm_get_IQ(device, iq_buf_size, trigger_buf_size, purge):
    iq_buf = numpy.zeros(iq_buf_size).astype(numpy.complex64)
    triggers = numpy.zeros(trigger_buf_size).astype(c_double)
    ns_since_epoch = c_longlong(-1)
    sample_loss = c_int(-1)
    samples_remaining = c_int(-1)

    status = smGetIQ(device, iq_buf, iq_buf_size,
                     triggers, trigger_buf_size,
                     byref(ns_since_epoch), purge,
                     byref(sample_loss), byref(samples_remaining))

    return {
        "status": status,
        "iq_buf": iq_buf,
        "triggers": triggers,
        "ns_since_epoch": ns_since_epoch.value,
        "sample_loss": sample_loss.value,
        "samples_remaining": samples_remaining.value
    }

@error_check
def sm_IQ_sweep_list_get_sweep(device, samples, steps):
    dst = numpy.zeros(samples).astype(numpy.float32)
    timestamps = numpy.zeros(steps).astype(c_longlong)

    status = smIQSweepListGetSweep(device, dst, timestamps)

    return {
        "status": status,
        "dst": dst,
        "timestamps": timestamps
    }

@error_check
def sm_IQ_sweep_list_start_sweep(device, pos, samples, steps):
    dst = numpy.zeros(samples).astype(numpy.float32)
    timestamps = numpy.zeros(steps).astype(c_longlong)

    status = smIQSweepListStartSweep(device, pos, dst, timestamps)

    return {
        "status": status,
        "dst": dst,
        "timestamps": timestamps
    }

@error_check
def sm_IQ_sweep_list_finish_sweep(device, pos):
    return {
        "status": smIQSweepListFinishSweep(device, pos)
    }

@error_check
def sm_seg_IQ_capture_start(device, capture):
    return {
        "status": smSegIQCaptureStart(device, capture)
    }

@error_check
def sm_seg_IQ_capture_wait(device, capture):
    return {
        "status": smSegIQCaptureWait(device, capture)
    }

@error_check
def sm_seg_IQ_capture_wait_async(device, capture):
    completed_smb = c_int(-1)

    status = smSegIQCaptureWaitAsync(device, capture, byref(completed_smb))

    return {
        "status": status,
        "completed": from_sm_bool(completed_smb.value)
    }

@error_check
def sm_seg_IQ_capture_timeout(device, capture, segment):
    timed_out_smb = c_int(0)

    status = smSegIQCaptureTimeout(device, capture, segment, byref(timed_out_smb))

    return {
        "status": status,
        "timed_out": from_sm_bool(timed_out_smb.value)
    }

@error_check
def sm_seg_IQ_capture_time(device, capture, segment):
    ns_since_epoch = c_longlong(-1)

    status = smSegIQCaptureTime(device, capture, segment, byref(ns_since_epoch))

    return {
        "status": status,
        "ns_since_epoch": ns_since_epoch.value
    }

@error_check
def sm_seg_IQ_capture_read(device, capture, segment, offset, length):
    iq = numpy.zeros(length).astype(numpy.complex64)

    status = smSegIQCaptureRead(device, capture, segment, iq, offset, length)

    return {
        "status": status,
        "iq": iq
    }

@error_check
def sm_seg_IQ_capture_finish(device, capture):
    return {
        "status": smSegIQCaptureFinish(device, capture)
    }

@error_check
def sm_seg_IQ_capture_full(device, capture, offset, length):
    iq = numpy.zeros(length).astype(numpy.complex64)
    ns_since_epoch = c_longlong(-1)
    timed_out_smb = c_int(0)

    status = smSegIQCaptureFull(device, capture, iq, offset, length,
                                byref(ns_since_epoch), byref(timed_out_smb))

    return {
        "status": status,
        "iq": iq,
        "ns_since_epoch": ns_since_epoch.value,
        "timed_out": from_sm_bool(timed_out_smb.value)
    }

@error_check
def sm_seg_IQLTEResample(input_arr, input_len, output_arr, output_len, clear_delay_line):
    output_len_ret = c_double(output_len)
    status = smSegIQLTEResample(input_arr, input_len,
                                     output_arr, byref(output_len),
                                     clear_delay_line)
    return {
        "status": status,
        "output_len": output_len_ret
    }

@error_check
def sm_set_IQ_full_band_atten(device, atten):
    return {
        "status": smSetIQFullBandAtten(device, atten)
    }

@error_check
def sm_set_IQ_full_band_corrected(device, corrected):
    return {
        "status": smSetIQFullBandCorrected(device, corrected)
    }

@error_check
def sm_set_IQ_full_band_samples(device, samples):
    return {
        "status": smSetIQFullBandSamples(device, samples)
    }

@error_check
def sm_set_IQ_full_band_triggerType(device, triggerType):
    return {
        "status": smSetIQFullBandTriggerType(device, triggerType)
    }

@error_check
def sm_set_IQ_full_band_video_trigger(device, triggerLevel):
    return {
        "status": smSetIQFullBandVideoTrigger(device, triggerLevel)
    }

@error_check
def sm_set_IQ_full_band_trigger_timeout(device, triggerTimeout):
    return {
        "status": smSetIQFullBandTriggerTimeout(device, triggerTimeout)
    }

@error_check
def sm_get_IQ_full_band(device, freq, samples):
    iq = numpy.zeros(samples).astype(numpy.float32)

    status = smGetIQFullBand(device, iq, freq)

    return {
        "status": status,
        "iq": iq
    }

@error_check
def sm_get_IQ_full_band_sweep(device, startIndex, stepSize, steps):
    iq = numpy.zeros(steps * stepSize).astype(numpy.complex64)

    status = smGetIQFullBandSweep(device, iq, startIndex, stepSize, steps)

    return {
        "status": status,
        "iq": iq
    }

@error_check
def sm_get_audio(device):
    audio = numpy.zeros(1000).astype(c_float)

    status = smGetAudio(device, audio)

    return {
        "status": status,
        "audio": audio
    }

@error_check
def sm_get_GPS_info(device, refresh, nmea_len):
    updated = c_int(-1)
    sec_since_epoch = c_longlong(-1)
    latitude = c_double(-1)
    longitude = c_double(-1)
    altitude = c_double(-1)
    nmea = numpy.zeros(nmea_len.value).astype(c_char)

    status = smGetGPSInfo(device, refresh, byref(updated), byref(sec_since_epoch),
                          byref(latitude), byref(longitude), byref(altitude),
                          nmea, byref(nmea_len))

    return {
        "status": status,
        "updated": updated.value,
        "sec_since_epoch": sec_since_epoch.value,
        "latitude": latitude.value,
        "longitude": longitude.value,
        "altitude": altitude.value,
        "nmea": nmea,
        "nmea_len": nmea_len.value
    }

@error_check
def sm_write_to_GPS(device, mem, length):
    return {
        "status": smWriteToGPS(device, mem, length)
    }

@error_check
def sm_set_fan_threshold(device, temp):
    return {
        "status": smSetFanThreshold(device, temp)
    }

@error_check
def sm_get_fan_threshold(device):
    temp = c_int(-1)

    status = smGetFanThreshold(device, byref(temp))

    return {
        "status": status,
        "temp": temp.value
    }

@error_check
def sm_set_IF_output(device, frequency):
    return {
        "status": smSetIFOutput(device, frequency)
    }

@error_check
def sm_get_cal_date(device):
    last_cal_date = c_ulonglong(-1)

    status = smGetCalDate(device, byref(last_cal_date))

    return {
        "status": status,
        "last_cal_date": last_cal_date.value
    }

@error_check
def sm_broadcast_network_config(host_addr, device_addr, port, non_volatile):
    return {
        "status": smBroadcastNetworkConfig(host_addr, device_addr,
                                           port, non_volatile)
    }

@error_check
def sm_network_config_get_device_list():
    serials = numpy.zeros(SM_MAX_DEVICES).astype(c_int)
    device_types = numpy.zeros(SM_MAX_DEVICES).astype(c_int)
    device_count = c_int(-1)

    status = smNetworkConfigGetDeviceList(serials, device_types, byref(device_count))

    return {
        "status": status,
        "serials": serials,
        "device_types": device_types,
        "device_count": device_count.value
    }

@error_check
def sm_network_config_open_device(serial_number):
    device = c_int(-1)

    status = smNetworkConfigOpenDevice(byref(device), serial_number)

    return {
        "status": status,
        "device": device.value
    }

@error_check
def sm_network_config_close_device(device):
    return {
        "status": smNetworkConfigCloseDevice(device)
    }

@error_check
def sm_network_config_get_MAC(device):
    mac = c_char_p("")

    status = smNetworkConfigGetMAC(device, byref(mac))

    return {
        "status": status,
        "mac": mac.value
    }

@error_check
def sm_network_config_set_IP(device, addr, nonVolatile):
    return {
        "status": smNetworkConfigSetIP(device, addr, nonVolatile)
    }

@error_check
def sm_network_config_get_IP(device):
    addr = c_char_p("")

    status = smNetworkConfigGetIP(device, byref(addr))

    return {
        "status": status,
        "addr": addr.value
    }

@error_check
def sm_network_config_set_port(device, port, nonVolatile):
    return {
        "status": smNetworkConfigSetPort(device, port, nonVolatile)
    }

@error_check
def sm_network_config_get_port(device):
    port = c_int(-1)

    status = smNetworkConfigGetPort(device, byref(port))

    return {
        "status": status,
        "port": port
    }

@error_check
def sm_get_API_version():
    return {
        "api_version": smGetAPIVersion()
    }

@error_check
def sm_get_error_string(status):
    return {
        "error_string": smGetErrorString(status)
    }

@error_check
def sm_get_product_ID():
    return {
        "product_id": smGetProductID()
    }

# Deprecated functions
smSetIQUSBQueueSize = smlib.smSetIQUSBQueueSize
@error_check
def sm_set_IQ_USB_queue_size(device, ms):
    return {
        "status": smSetIQUSBQueueSize(device, c_float(ms))
    }

# Deprecated macros
SM200A_AUTO_ATTEN = SM_AUTO_ATTEN
SM200A_MAX_ATTEN = SM_MAX_ATTEN
SM200A_MAX_REF_LEVEL = SM_MAX_REF_LEVEL
SM200A_MAX_SWEEP_QUEUE_SZ = SM_MAX_SWEEP_QUEUE_SZ
SM200A_MIN_FREQ = SM200_MIN_FREQ
SM200A_MAX_FREQ = SM200_MAX_FREQ
SM200A_MAX_IQ_DECIMATION = SM_MAX_IQ_DECIMATION
SM200A_PRESELECTOR_MAX_FREQ = SM_PRESELECTOR_MAX_FREQ
SM200A_FAST_SWEEP_MIN_RBW = SM_FAST_SWEEP_MIN_RBW
SM200A_RTSA_MIN_SPAN = SM_REAL_TIME_MIN_SPAN
SM200A_RTSA_MAX_SPAN = SM_REAL_TIME_MAX_SPAN
SM200A_MIN_SWEEP_TIME = SM_MIN_SWEEP_TIME
SM200A_MAX_SWEEP_TIME = SM_MAX_SWEEP_TIME
SM200A_SPI_MAX_BYTES = SM_SPI_MAX_BYTES
SM200A_GPIO_SWEEP_MAX_STEPS = SM_GPIO_SWEEP_MAX_STEPS
SM200A_GPIO_SWITCH_MAX_STEPS = SM_GPIO_SWITCH_MAX_STEPS
SM200A_GPIO_SWITCH_MIN_COUNT = SM_GPIO_SWITCH_MIN_COUNT
SM200A_GPIO_SWITCH_MAX_COUNT = SM_GPIO_SWITCH_MAX_COUNT
SM200A_TEMP_WARNING = SM_TEMP_WARNING
SM200A_TEMP_MAX = SM_TEMP_MAX
SM200B_MAX_SEGMENTED_IQ_SEGMENTS = SM_MAX_SEGMENTED_IQ_SEGMENTS
SM200B_MAX_SEGMENTED_IQ_SAMPLES = SM_MAX_SEGMENTED_IQ_SAMPLES
SM200_ADDR_ANY = SM_ADDR_ANY
SM200_DEFAULT_ADDR = SM_DEFAULT_ADDR
SM200_DEFAULT_PORT = SM_DEFAULT_PORT
