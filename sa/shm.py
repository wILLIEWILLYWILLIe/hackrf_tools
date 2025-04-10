from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory
import numpy as np
from typing import Dict


IQ_DATA_TYPE = np.complex64
IQ_DATA_BYTES = 8

def toShmSize(iq_point_num: int) -> int:
    return int(iq_point_num*IQ_DATA_BYTES)

class ShmManager:
    def __init__(self):
        # To prevent the shared memory from being released by the garbage collector, they are stored in a list.
        # Child processes must link to shared memories by names.
        self.__shm_list: list[SharedMemory] = []
        self.__shm_name_lock_pairs: Dict[str, Lock] = {}

    def create(self, sa_num: int, shm_num: int, shm_size: int):
        for i in range(sa_num*shm_num):
            name = f"shm_{i}"
            self.__shm_list.append(SharedMemory(create=True, name=name, size=shm_size))
            self.__shm_name_lock_pairs[name] = Lock()
            print(f"SHM created name={name}, size={shm_size}")

    def releaseAll(self):
        for shm in self.__shm_list:
            shm.close()
            shm.unlink()
        print("===All SHM released===")

    def getNameLockPairs(self) -> Dict[str, Lock]:
        return self.__shm_name_lock_pairs


class ShmFinder:
    def __init__(self, shm_name_lock_pairs: Dict[int, Lock]):
        self.__shm_name_lock_pairs = shm_name_lock_pairs
        self.__curr_shm_index = 0
        self.__linkToShms()

    def __linkToShms(self):
        self.__shms: list[SharedMemory] = []
        for name in self.__shm_name_lock_pairs.keys():
            self.__shms.append(SharedMemory(create=False, name=name))
        self.__shm_num = len(self.__shms)

    def getShm(self, name: str) -> SharedMemory:
        for shm in self.__shms:
            if shm.name == name:
                while not self.__shm_name_lock_pairs[name].acquire(block=False):
                    pass
                return shm

    def getAvailableShm(self) -> SharedMemory:
        while True:
            for _ in range(self.__shm_num):
                self.__switchShmIndex()
                shm = self.__shms[self.__curr_shm_index]
                if self.__shm_name_lock_pairs[shm.name].acquire(block=False):
                    return shm

    def __switchShmIndex(self):
        self.__curr_shm_index = (self.__curr_shm_index + 1) % self.__shm_num
