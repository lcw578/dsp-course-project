# -*- coding: utf-8 -*-
"""
解调与误码率 (BER) 评估模块
负责：ASK 包络检波、FSK 相关解调、PSK 相干解调、误码率统计
从通信系统层面量化各滤波方法对解调性能的实际改善
"""

import numpy as np
from scipy import signal as sig
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def _split_bits(signal_data, samples_per_bit):
    """将信号按比特切分为二维数组 (n_bits, samples_per_bit)"""
    samples_per_bit = samples_per_bit or config.SAMPLES_PER_BIT
    n_bits = len(signal_data) // samples_per_bit
    return signal_data[:n_bits * samples_per_bit].reshape(n_bits, samples_per_bit)


def demod_ask(signal_data, fs=None, samples_per_bit=None):
    """
    2-ASK (OOK) 包络检波解调
    
    原理：Hilbert 变换提取包络 → 比特能量积累 → 动态中点阈值判决
    
    Parameters
    ----------
    signal_data : ndarray
        接收信号
    fs : float
        采样率
    samples_per_bit : int
        每比特采样点数
    
    Returns
    -------
    bits : ndarray
        解调出的比特序列
    """
    fs = fs or config.FS
    samples_per_bit = samples_per_bit or config.SAMPLES_PER_BIT
    
    envelope = np.abs(sig.hilbert(signal_data))
    frames = _split_bits(envelope, samples_per_bit)
    energies = np.mean(frames ** 2, axis=1)
    
    # 阈值取能量 10%/90% 分位中点，对离群帧 (如自适应滤波收敛瞬态) 鲁棒
    lo = np.percentile(energies, 10)
    hi = np.percentile(energies, 90)
    threshold = (lo + hi) / 2
    return (energies > threshold).astype(int)


def demod_fsk(signal_data, fs=None, f1=None, f2=None, samples_per_bit=None):
    """
    2-FSK 相关解调（匹配滤波器组）
    
    原理：每比特分别与两个频率的本地载波作相关运算，相关值大者判决
    
    Parameters
    ----------
    signal_data : ndarray
        接收信号
    fs : float
        采样率
    f1, f2 : float
        bit=0/1 对应的载波频率
    samples_per_bit : int
        每比特采样点数
    
    Returns
    -------
    bits : ndarray
        解调出的比特序列
    """
    fs = fs or config.FS
    f1 = f1 or config.FC_FSK_F1
    f2 = f2 or config.FC_FSK_F2
    samples_per_bit = samples_per_bit or config.SAMPLES_PER_BIT
    
    frames = _split_bits(signal_data, samples_per_bit)
    t_seg = np.arange(samples_per_bit) / fs
    ref1 = np.cos(2 * np.pi * f1 * t_seg)
    ref2 = np.cos(2 * np.pi * f2 * t_seg)
    
    corr1 = np.abs(frames @ ref1)
    corr2 = np.abs(frames @ ref2)
    return (corr2 > corr1).astype(int)


def demod_psk(signal_data, fs=None, fc=None, samples_per_bit=None):
    """
    2-PSK (BPSK) 相干解调
    
    原理：每比特与本地同频同相载波相关，按极性判决
          φ0=0 → 相关值为正 → bit=0；φ1=π → 相关值为负 → bit=1
    
    Parameters
    ----------
    signal_data : ndarray
        接收信号
    fs : float
        采样率
    fc : float
        载波频率
    samples_per_bit : int
        每比特采样点数
    
    Returns
    -------
    bits : ndarray
        解调出的比特序列
    """
    fs = fs or config.FS
    fc = fc or config.FC_PSK
    samples_per_bit = samples_per_bit or config.SAMPLES_PER_BIT
    
    frames = _split_bits(signal_data, samples_per_bit)
    t_seg = np.arange(samples_per_bit) / fs
    ref = np.cos(2 * np.pi * fc * t_seg)
    
    corr = frames @ ref
    return (corr < 0).astype(int)


DEMOD_FUNCS = {
    'ASK': demod_ask,
    'FSK': demod_fsk,
    'PSK': demod_psk,
}


def compute_ber(true_bits, detected_bits):
    """
    计算误码率 (Bit Error Rate)
    
    Parameters
    ----------
    true_bits, detected_bits : ndarray
        真实/解调比特序列
    
    Returns
    -------
    ber : float
        误码率 ∈ [0, 1]
    """
    n = min(len(true_bits), len(detected_bits))
    if n == 0:
        return 1.0
    return float(np.mean(np.asarray(true_bits[:n]) != np.asarray(detected_bits[:n])))


def evaluate_ber(mod_type, signal_data, true_bits, fs=None):
    """
    对给定信号执行对应解调方式并计算 BER
    
    Parameters
    ----------
    mod_type : str
        'ASK' / 'FSK' / 'PSK'
    signal_data : ndarray
        接收信号（含噪或滤波后）
    true_bits : ndarray
        发射比特序列
    
    Returns
    -------
    ber : float
        误码率
    """
    detected = DEMOD_FUNCS[mod_type](signal_data, fs=fs)
    return compute_ber(true_bits, detected)
