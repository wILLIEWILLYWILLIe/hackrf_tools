#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
from hackrf import HackRF

# 全域變數，用於儲存接收資料
rx_buffer = []

# RX callback，用於把每次接收的資料累積到 rx_buffer
def rx_callback(data, length):
    # data 是 bytes，需轉成 complex 或 int16 再進行處理
    # HackRF 預設輸出 IQ = 8位元 (signed 8-bit, I+Q interleaved)
    global rx_buffer
    rx_iq = np.frombuffer(data, dtype=np.int8).astype(np.float32)
    # 將交織的 I、Q 拆分並轉為複數
    # 假設 data: I0, Q0, I1, Q1, ...
    I = rx_iq[0::2]
    Q = rx_iq[1::2]
    complex_iq = I + 1j*Q
    rx_buffer.append(complex_iq)

    return 0  # return 0 表示繼續接收

def main():
    hackrf = HackRF()
    device = hackrf.open()

    # 設定參數
    freq = 2.45e9  # 或 5.79e9
    sample_rate = 5e6
    tx_gain = 0  # IF gain, 視需要調整
    rx_gain = 40 # LNA gain, 視需要調整
    vga_gain = 20 # Baseband VGA gain

    # HackRF 初始化
    device.set_sample_rate(sample_rate)
    device.set_freq(freq)
    
    # 設定接收增益
    device.set_lna_gain(rx_gain)
    device.set_vga_gain(vga_gain)

    # 產生一段發射波形（例如簡單的脈衝或 CW）
    # 這裡示範產生一個簡單的 100us 脈衝
    # HackRF I/Q = int8 格式，範圍 -128 ~ +127
    pulse_samples = int(sample_rate * 100e-6)  # 100 微秒脈衝長度
    pulse_data = np.zeros(pulse_samples*2, dtype=np.int8)  # 乘 2 的原因: I, Q 交織
    # 填入 127 表示最大幅度（要注意不要過大而失真）
    for i in range(pulse_samples):
        pulse_data[2*i]   = 127  # I
        pulse_data[2*i+1] = 0    # Q

    # TDD 流程: 先 Tx -> 停止 Tx -> 立即 Rx
    print("[INFO] Start TX pulse...")
    device.start_tx_mode_sync(pulse_data.tobytes())  
    # start_tx_mode_sync(): pyhackrf 提供的同步 TX 模式 (阻塞直到送完)

    print("[INFO] TX done, stop TX.")
    device.stop_tx_mode()

    # 讓對端或目標有時間「反射」(若需要)
    time.sleep(0.001)  # 1 毫秒等待，可依情況調整

    # 開始接收
    global rx_buffer
    rx_buffer = []  # 清空緩衝
    print("[INFO] Start RX...")
    device.start_rx(rx_callback)

    # 接收一段時間，例如 10 ms
    time.sleep(0.01)

    device.stop_rx()
    print("[INFO] Stop RX.")

    # 關閉裝置
    device.close()

    # 將接收資料合併
    if len(rx_buffer) > 0:
        rx_data = np.concatenate(rx_buffer)
        print(f"[INFO] Received {rx_data.size} IQ samples.")
        # 您可在這裡對 rx_data 做雷達信號處理（例如相關運算、FFT 等）
        # 簡單示範存檔:
        rx_data.tofile("rx_data.bin")
        print("[INFO] Saved to rx_data.bin.")
    else:
        print("[WARNING] No data received.")

if __name__ == "__main__":
    main()
