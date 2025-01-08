## HACKRF TOOL
### Hackrf ENV Setup
Note to first verify hackrf packages are correctly install in you ubuntu

Install hackrf driver with the following link
https://hackrf.readthedocs.io/en/latest/installing_hackrf_software.html

Clone the repo into pc
```bash 
git clone https://github.com/greatscottgadgets/hackrf.git
```
Once you have the source downloaded, the host tools can be built as follows:
```bash 
cd hackrf/host
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
```
If Error occurs, check whether there is packages needed for cmake
```bash
sudo apt install cmake  # If missing cmake
sudo apt install libusb-1.0-0-dev # If missing LIBUSB for USB communication
sudo apt install libfftw3-dev #If missing FFT lib
sudo apt install pkg-config 
sudo apt install usbutils
```
If setup successfully, we will observe serial number by
```bash 
hackrf_info
```

### Code ENV Setup
1. Create venv for the first time if needed
```bash
python3 -m venv .venv
```
2. Move to environment
``` bash 
source .venv/bin/activate
```
3. Install the lib with 
```bash 
pip install -r requirements.txt
```
### Receive
```bash 
python3 hack_receive.py
```

### Transmit
```bash 
bash tx.sh
```
