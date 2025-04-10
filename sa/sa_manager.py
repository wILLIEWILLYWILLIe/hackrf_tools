from typing import Dict
from .sa_process import SaProcess
from multiprocessing import Lock
from .smdevice.sm_api import sm_get_device_list2

def getSerialNumbers() -> list[int]:
    devices_info = sm_get_device_list2()
    avai_serials = devices_info['serials'][:devices_info['device_count']]
    return [str(serial) for serial in avai_serials]


class SaManager:
    def __init__(self):
        self.__sa_process_dict: Dict[int, SaProcess] = {}

    def createSaProcess(self, serial_number: str, shm_name_lock_pair: Dict[str, Lock]):
        sa_process = SaProcess(
            serial_number=serial_number,
            shm_name_lock_pair=shm_name_lock_pair
        )
        sa_process.start()
        sa_process.waitInit()
        self.__sa_process_dict[serial_number] = sa_process
        print(f"SA Process {serial_number} created")

    def captureIq(self, serial_number:str, setting:Dict):
        self.__sa_process_dict[serial_number].captureIq(setting)

    def getIq(self, serial_number:str):
        return self.__sa_process_dict[serial_number].getIq()
    
    def waitReadyCapture(self, serial_number:str):
        self.__sa_process_dict[serial_number].waitReadyCapture()

    def terminate(self):
        for sa_process in self.__sa_process_dict.values():
            sa_process.terminate()
        for sa_process in self.__sa_process_dict.values():
            sa_process.join()


