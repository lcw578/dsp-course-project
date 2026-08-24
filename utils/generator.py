# -*- coding: utf-8 -*-
"""
信号生成模块
负责：生成 ASK/FSK/PSK 数字调制信号、叠加信道噪声
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def generate_binary_sequence(n_bits, seed=None):
    """
    生成随机二进制比特流
    
    Parameters
    ----------
    n_bits : int
        比特数量
    seed : int, optional
        随机种子（用于可复现性）
    
    Returns
    -------
    bits : ndarray
        0/1 二进制序列
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(config.RANDOM_SEED)
    return rng.randint(0, 2, n_bits)


def generate_ask(bits, fs=None, fc=None, A1=None, A0=None, samples_per_bit=None):
    """
    生成 2-ASK (OOK) 调制信号
    
    原理：s(n) = A(m) * cos(2π * fc * n * Ts)
    其中 A(m) = A1 if bit=1, A0 if bit=0
    
    Parameters
    ----------
    bits : ndarray
        二进制比特序列
    fs : float
        采样率
    fc : float
        载波频率
    A1, A0 : float
        bit=1/0 对应的幅度
    samples_per_bit : int
        每比特采样点数
    
    Returns
    -------
    signal : ndarray
        ASK 调制信号
    t : ndarray
        时间轴
    """
    fs = fs or config.FS
    fc = fc or config.FC_ASK
    A1 = A1 if A1 is not None else config.ASK_AMPLITUDE_1
    A0 = A0 if A0 is not None else config.ASK_AMPLITUDE_0
    samples_per_bit = samples_per_bit or config.SAMPLES_PER_BIT
    
    n_total = len(bits) * samples_per_bit
    t = np.arange(n_total) / fs
    signal = np.zeros(n_total)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        amplitude = A1 if bit == 1 else A0
        t_seg = np.arange(samples_per_bit) / fs
        signal[start:end] = amplitude * np.cos(2 * np.pi * fc * t_seg)
    
    return signal, t


def generate_fsk(bits, fs=None, f1=None, f2=None, samples_per_bit=None):
    """
    生成 2-FSK 调制信号
    
    原理：bit=0 → cos(2π * f1 * n * Ts)
          bit=1 → cos(2π * f2 * n * Ts)
    
    Parameters
    ----------
    bits : ndarray
        二进制比特序列
    fs : float
        采样率
    f1, f2 : float
        bit=0/1 对应的载波频率
    samples_per_bit : int
        每比特采样点数
    
    Returns
    -------
    signal : ndarray
        FSK 调制信号
    t : ndarray
        时间轴
    """
    fs = fs or config.FS
    f1 = f1 or config.FC_FSK_F1
    f2 = f2 or config.FC_FSK_F2
    samples_per_bit = samples_per_bit or config.SAMPLES_PER_BIT
    
    n_total = len(bits) * samples_per_bit
    t = np.arange(n_total) / fs
    signal = np.zeros(n_total)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        freq = f2 if bit == 1 else f1
        t_seg = np.arange(samples_per_bit) / fs
        signal[start:end] = np.cos(2 * np.pi * freq * t_seg)
    
    return signal, t


def generate_psk(bits, fs=None, fc=None, phase_0=None, phase_1=None, samples_per_bit=None):
    """
    生成 2-PSK (BPSK) 调制信号
    
    原理：bit=0 → cos(2π * fc * n * Ts + φ0)
          bit=1 → cos(2π * fc * n * Ts + φ1)
    
    Parameters
    ----------
    bits : ndarray
        二进制比特序列
    fs : float
        采样率
    fc : float
        载波频率
    phase_0, phase_1 : float
        bit=0/1 对应的相位 (rad)
    samples_per_bit : int
        每比特采样点数
    
    Returns
    -------
    signal : ndarray
        PSK 调制信号
    t : ndarray
        时间轴
    """
    fs = fs or config.FS
    fc = fc or config.FC_PSK
    phase_0 = phase_0 if phase_0 is not None else config.PSK_PHASE_0
    phase_1 = phase_1 if phase_1 is not None else config.PSK_PHASE_1
    samples_per_bit = samples_per_bit or config.SAMPLES_PER_BIT
    
    n_total = len(bits) * samples_per_bit
    t = np.arange(n_total) / fs
    signal = np.zeros(n_total)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        phase = phase_1 if bit == 1 else phase_0
        t_seg = np.arange(samples_per_bit) / fs
        signal[start:end] = np.cos(2 * np.pi * fc * t_seg + phase)
    
    return signal, t


def add_awgn(signal, snr_db=None):
    """
    叠加加性高斯白噪声 (AWGN)
    
    原理：噪声功率 = 信号功率 / (10^(SNR/10))
    
    Parameters
    ----------
    signal : ndarray
        纯净信号
    snr_db : float
        信噪比 (dB)
    
    Returns
    -------
    noisy_signal : ndarray
        含噪信号
    noise : ndarray
        噪声分量（便于后续分析）
    """
    snr_db = snr_db if snr_db is not None else config.SNR_DB
    
    # 计算信号功率
    signal_power = np.mean(signal ** 2)
    
    # 根据 SNR 计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # 生成高斯白噪声
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    noisy_signal = signal + noise
    return noisy_signal, noise


def generate_all_signals(n_bits=None, snr_db=None, seed=None):
    """
    一次性生成所有调制类型的纯净信号和含噪信号
    
    Returns
    -------
    results : dict
        包含每种调制方式的 bits, clean_signal, noisy_signal, noise, t
    """
    n_bits = n_bits or config.N_BITS
    snr_db = snr_db if snr_db is not None else config.SNR_DB
    
    bits = generate_binary_sequence(n_bits, seed=seed)
    
    results = {}
    
    # ASK
    ask_clean, t_ask = generate_ask(bits)
    ask_noisy, ask_noise = add_awgn(ask_clean, snr_db)
    results['ASK'] = {
        'bits': bits,
        'clean': ask_clean,
        'noisy': ask_noisy,
        'noise': ask_noise,
        't': t_ask
    }
    
    # FSK
    fsk_clean, t_fsk = generate_fsk(bits)
    fsk_noisy, fsk_noise = add_awgn(fsk_clean, snr_db)
    results['FSK'] = {
        'bits': bits,
        'clean': fsk_clean,
        'noisy': fsk_noisy,
        'noise': fsk_noise,
        't': t_fsk
    }
    
    # PSK
    psk_clean, t_psk = generate_psk(bits)
    psk_noisy, psk_noise = add_awgn(psk_clean, snr_db)
    results['PSK'] = {
        'bits': bits,
        'clean': psk_clean,
        'noisy': psk_noisy,
        'noise': psk_noise,
        't': t_psk
    }
    
    return results
