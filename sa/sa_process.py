import numpy as np
import logging
from multiprocessing import Queue, Process, Event, Lock
from typing import Dict
from .shm import ShmFinder, IQ_DATA_TYPE
from .sm200b import SM200b


def getLogger(name: str, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


class SaProcess(Process):
    LOG_LEVEL = logging.DEBUG

    def __init__(self, serial_number: str, shm_name_lock_pair: Dict[str, Lock]):
        Process.__init__(self, name=f"SA_{serial_number}", daemon=True)
        self.__stop_event = Event()
        self.__serial_number = serial_number
        # Events
        self.__init_done_event = Event()
        self.__ready_capture_event = Event()
        # Queues
        self.__result_queue = Queue()
        self.__setting_queue = Queue()

        self.shm_name_lock_pair = shm_name_lock_pair

    def run(self):
        self.onStart()
        while not self.__stop_event.is_set():
            self.loop()
        self.onTerminate()

    def onStart(self):
        self.__logger = getLogger(self._name, self.LOG_LEVEL)
        self.__shm_manager = ShmFinder(self.shm_name_lock_pair)
        self.__device = SM200b(self.__serial_number)
        self.__init_done_event.set()
        self.__logger.debug("start")

    def onTerminate(self):
        self.__device.close()
        self.__logger.debug("terminated")

    def terminate(self):
        self.__stop_event.set()
        # put dummy data to wakeup the loop
        self.__setting_queue.put(None)

    def loop(self):
        setting = self.__setting_queue.get()
        if self.__stop_event.is_set():
            return

        # configure SA
        self.__device.setCenterFreq(setting["center_freq"])
        self.__device.setRefLevel(setting["ref_level"])
        self.__device.setTimeDuration(setting["time_duration"])
        self.__device.setTrigger(setting["trigger_type"], setting["ext_trig_timeout"])

        # get avaliable SHM
        shm = self.__shm_manager.getAvailableShm()

        # link IQ array
        iq_point_num = int(setting["sample_rate"]*setting["time_duration"])
        iq_arr = np.ndarray(shape=(iq_point_num,), dtype=IQ_DATA_TYPE, buffer=shm.buf)

        # Ready for capturing
        self.__device.startCaptureIq()
        self.__ready_capture_event.set()

        # wait for capture and read IQ
        self.__device.waitCaptureIq()
        self.__device.readIq(iq_arr)
        self.__ready_capture_event.clear()

        iq_info = {
            "shm_name": shm.name,
            "iq_point_num": iq_point_num
        }
        self.__result_queue.put(iq_info)

    def waitInit(self):
        self.__init_done_event.wait()

    def waitReadyCapture(self):
        self.__ready_capture_event.wait()

    def captureIq(self, setting: Dict):
        self.__setting_queue.put(setting)

    def getIq(self):
        return self.__result_queue.get()
