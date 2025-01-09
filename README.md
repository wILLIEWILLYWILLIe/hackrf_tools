# HACKRF TOOL
## Hackrf ENV Setup
Before starting, ensure that HackRF packages are properly installed on your Ubuntu system.

1. Install hackrf driver with the following link:
https://hackrf.readthedocs.io/en/latest/installing_hackrf_software.html

2. Clone the HackRF Repository
```bash 
git clone https://github.com/greatscottgadgets/hackrf.git
```
3. Once you have the source downloaded, build and install HackRF Host Tools
```bash 
cd hackrf/host
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
```
4. Handle Build Errors If you encounter errors during the build process, ensure the required dependencies are installed:
```bash
sudo apt install cmake  # If missing cmake
sudo apt install libusb-1.0-0-dev # If missing LIBUSB for USB communication
sudo apt install libfftw3-dev #If missing FFT lib
sudo apt install pkg-config 
sudo apt install usbutils
```
5. Verify Installation Run the following command to check if HackRF is properly detected:
```bash 
hackrf_info
```
If the device is connected and installed correctly, the serial number and other information about the HackRF device will be displayed.
6. Return to the Root Directory
```bash 
cd ../../..
```

## Code ENV Setup
1. Clone the repo first
```bash
git clone https://github.com/wILLIEWILLYWILLIe/hackrf_tools.git
cd hackrf_tools
```
2. Create a Virtual Environment If you're running the tools for the first time, create a virtual environment to isolate dependencies:
```bash
python3 -m venv .venv
```
3. Activate the Virtual Environment
``` bash 
source .venv/bin/activate
```
4. Install all required libraries using the provided requirements.txt file
```bash 
pip install -r requirements.txt
```
## How to use 
### Receive Signal to Spectrogram
To receive a signal using HackRF, execute the following:
```bash 
python3 hack_receive.py
```
### Transmit Signal
To transmit a signal using HackRF, execute the following:
```bash 
bash tx.sh
```
