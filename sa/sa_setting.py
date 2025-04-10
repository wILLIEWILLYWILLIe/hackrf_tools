from .smdevice.sm_api import *
from enum import Enum
from dataclasses import dataclass

class SA_TRIGGER_TYPE(str, Enum):
    RISING = 'rising'
    FALLING = 'falling'
    IMMEDIATE = 'immediate'