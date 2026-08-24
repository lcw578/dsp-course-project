# -*- coding: utf-8 -*-
"""
滤波器设计与迭代优化模块
负责：FIR/IIR 滤波器设计、幅频/相频特性仿真、迭代优化、LMS 自适应滤波
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ====================== FIR 滤波器设计 ======================

def design_fir(order, cutoff_low, cutoff_high, fs=None, window='hamming'):
    """
    FIR 带通滤波器设计（窗函数法）
    
    Parameters
    ----------
    order : int
        滤波器阶数
    cutoff_low, cutoff_high : float
        带通截止频率 (Hz)
    fs : float
        采样率
    window : str
        窗函数类型 ('hamming', 'hann', 'blackman', 'kaiser')
    
    Returns
    -------
    b : ndarray
        FIR 滤波器系数
    a : ndarray
        固定为 [1]（FIR无反馈）
    """
    fs = fs or config.FS
    nyq = fs / 2.0
    
    # 归一化截止频率
    low = cutoff_low / nyq
    high = cutoff_high / nyq
    
    # 确保频率在有效范围内
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    # 窗函数法设计 FIR
    if window == 'kaiser':
        b = sig.firwin(order + 1, [low, high], pass_zero=False, 
                      window=('kaiser', 5.0))
    else:
        b = sig.firwin(order + 1, [low, high], pass_zero=False, 
                      window=window)
    
    a = np.array([1.0])
    return b, a


def design_iir(order, cutoff_low, cutoff_high, fs=None, ftype='butter'):
    """
    IIR 带通滤波器设计
    
    Parameters
    ----------
    order : int
        滤波器阶数
    cutoff_low, cutoff_high : float
        带通截止频率 (Hz)
    fs : float
        采样率
    ftype : str
        滤波器类型 ('butter', 'cheby1', 'ellip')
    
    Returns
    -------
    b : ndarray
        分子多项式系数
    a : ndarray
        分母多项式系数
    """
    fs = fs or config.FS
    nyq = fs / 2.0
    
    low = cutoff_low / nyq
    high = cutoff_high / nyq
    
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    if ftype == 'butter':
        b, a = sig.butter(order, [low, high], btype='band')
    elif ftype == 'cheby1':
        b, a = sig.cheby1(order, 1, [low, high], btype='band')  # 1dB纹波
    elif ftype == 'ellip':
        b, a = sig.ellip(order, 1, 40, [low, high], btype='band')  # 1dB纹波, 40dB阻带衰减
    else:
        raise ValueError(f"不支持的滤波器类型: {ftype}")
    
    return b, a


def apply_filter(b, a, signal_data):
    """
    执行滤波操作 (使用零相位滤波避免相位失真)
    
    Parameters
    ----------
    b, a : ndarray
        滤波器系数
    signal_data : ndarray
        输入信号
    
    Returns
    -------
    filtered : ndarray
        滤波后信号
    """
    # 使用 filtfilt 实现零相位滤波
    # 需要信号长度至少是 3 * max(len(a), len(b))
    padlen = 3 * max(len(a), len(b))
    if len(signal_data) <= padlen:
        # 信号太短，使用普通滤波
        filtered = sig.lfilter(b, a, signal_data)
    else:
        try:
            filtered = sig.filtfilt(b, a, signal_data)
        except ValueError:
            filtered = sig.lfilter(b, a, signal_data)
    
    return filtered


# ====================== 滤波器特性分析 ======================

def filter_response(b, a, fs=None, n_points=2048):
    """
    计算滤波器幅频和相频响应
    
    Returns
    -------
    freqs : ndarray
        频率轴 (Hz)
    magnitude_db : ndarray
        幅频响应 (dB)
    phase_deg : ndarray
        相频响应 (度)
    """
    fs = fs or config.FS
    w, h = sig.freqz(b, a, worN=n_points, fs=fs)
    
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-15)
    phase_deg = np.degrees(np.unwrap(np.angle(h)))
    
    return w, magnitude_db, phase_deg


def plot_filter_response(b, a, fs=None, title='', save_path=None):
    """
    绘制滤波器的幅频特性和相频特性
    """
    fs = fs or config.FS
    freqs, mag_db, phase_deg = filter_response(b, a, fs)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'滤波器频率响应特性 {title}', fontsize=14, fontweight='bold')
    
    # 幅频特性
    ax1.plot(freqs, mag_db, color='#2196F3', linewidth=1.5)
    ax1.set_title('幅频特性 |H(f)|', fontsize=12)
    ax1.set_ylabel('增益 (dB)', fontsize=10)
    ax1.set_xlim(0, fs / 2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=-3, color='r', linestyle='--', alpha=0.5, label='-3dB线')
    ax1.legend(fontsize=9)
    
    # 相频特性
    ax2.plot(freqs, phase_deg, color='#FF9800', linewidth=1.5)
    ax2.set_title('相频特性 ∠H(f)', fontsize=12)
    ax2.set_ylabel('相位 (°)', fontsize=10)
    ax2.set_xlabel('频率 (Hz)', fontsize=10)
    ax2.set_xlim(0, fs / 2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 滤波器响应图 → {save_path}")
    plt.close()
    
    return freqs, mag_db, phase_deg


def plot_filter_zero_pole(b, a, title='', save_path=None):
    """
    绘制滤波器的零极点分布图
    """
    zeros, poles, gain = sig.tf2zpk(b, a)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title(f'滤波器零极点分布 {title}', fontsize=14, fontweight='bold')
    
    # 单位圆
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.0, alpha=0.5, label='单位圆')
    
    # 零点
    ax.scatter(zeros.real, zeros.imag, marker='o', s=100, 
              facecolors='none', edgecolors='#2196F3', linewidths=2, 
              label=f'零点 ({len(zeros)}个)', zorder=5)
    
    # 极点
    ax.scatter(poles.real, poles.imag, marker='x', s=100, 
              color='#F44336', linewidths=2, 
              label=f'极点 ({len(poles)}个)', zorder=5)
    
    ax.set_xlabel('实部 Re(z)', fontsize=11)
    ax.set_ylabel('虚部 Im(z)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    max_range = max(
        np.max(np.abs(zeros.real)) if len(zeros) > 0 else 1,
        np.max(np.abs(poles.real)) if len(poles) > 0 else 1,
        np.max(np.abs(zeros.imag)) if len(zeros) > 0 else 1,
        np.max(np.abs(poles.imag)) if len(poles) > 0 else 1,
        1.2
    )
    ax.set_xlim(-max_range * 1.3, max_range * 1.3)
    ax.set_ylim(-max_range * 1.3, max_range * 1.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 滤波器零极点图 → {save_path}")
    plt.close()


# ====================== 迭代优化 ======================

def compute_snr(clean, noisy):
    """计算信噪比 (dB)"""
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-15:
        return 100.0  # 近似无噪声
    return 10 * np.log10(signal_power / noise_power)


def compute_correlation(original, filtered):
    """计算信号保真度 (皮尔逊相关系数)"""
    # 对齐长度
    min_len = min(len(original), len(filtered))
    x = original[:min_len]
    y = filtered[:min_len]
    
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    
    corr = np.corrcoef(x, y)[0, 1]
    return corr


def iterative_optimize(clean_signal, noisy_signal, fs=None, 
                        cutoff_low=500, cutoff_high=1500,
                        fir_orders=None, fir_windows=None,
                        iir_orders=None, iir_types=None,
                        cutoff_scales=None):
    """
    迭代优化滤波器参数
    遍历阶数、窗函数/类型、截止频率缩放组合，选择最优方案
    
    Returns
    -------
    results : list of dict
        每种组合的评价结果
    best_result : dict
        最优结果
    """
    fs = fs or config.FS
    fir_orders = fir_orders or config.FIR_ORDERS
    fir_windows = fir_windows or config.FIR_WINDOWS
    iir_orders = iir_orders or config.IIR_ORDERS
    iir_types = iir_types or config.IIR_TYPES
    cutoff_scales = cutoff_scales or config.CUTOFF_SCALES
    
    nyq = fs / 2.0
    snr_in_ref = compute_snr(clean_signal, noisy_signal)
    
    results = []
    
    # 遍历截止频率缩放 (调整带宽 → 改变 Z 域零极点位置)
    for scale in cutoff_scales:
        eff_low = max(50.0, min(cutoff_low * scale, nyq - 100))
        eff_high = max(eff_low + 100, min(cutoff_high * scale, nyq - 50))
        
        # FIR 参数扫描
        for order in fir_orders:
            for window in fir_windows:
                try:
                    t_start = time.time()
                    b, a = design_fir(order, eff_low, eff_high, fs, window)
                    filtered = apply_filter(b, a, noisy_signal)
                    t_elapsed = time.time() - t_start
                    
                    snr_out = compute_snr(clean_signal, filtered)
                    corr = compute_correlation(clean_signal, filtered)
                    
                    results.append({
                        'type': 'FIR',
                        'order': order,
                        'window': window,
                        'cutoff_scale': scale,
                        'cutoff': (eff_low, eff_high),
                        'snr_improvement': snr_out - snr_in_ref,
                        'snr_out': snr_out,
                        'correlation': corr,
                        'time': t_elapsed,
                        'b': b,
                        'a': a,
                        'filtered': filtered,
                        # 综合评分：保真度 0.6 + 归一化SNR改善 0.4
                        'score': corr * 0.6 + min(max(snr_out - snr_in_ref, 0.0), 15.0) / 15.0 * 0.4
                    })
                except Exception as e:
                    print(f"  [跳过] FIR x{scale:g} order={order}, window={window}: {e}")
        
        # IIR 参数扫描
        for order in iir_orders:
            for ftype in iir_types:
                try:
                    t_start = time.time()
                    b, a = design_iir(order, eff_low, eff_high, fs, ftype)
                    filtered = apply_filter(b, a, noisy_signal)
                    t_elapsed = time.time() - t_start
                    
                    snr_out = compute_snr(clean_signal, filtered)
                    corr = compute_correlation(clean_signal, filtered)
                    
                    results.append({
                        'type': 'IIR',
                        'order': order,
                        'window': ftype,
                        'cutoff_scale': scale,
                        'cutoff': (eff_low, eff_high),
                        'snr_improvement': snr_out - snr_in_ref,
                        'snr_out': snr_out,
                        'correlation': corr,
                        'time': t_elapsed,
                        'b': b,
                        'a': a,
                        'filtered': filtered,
                        'score': corr * 0.6 + min(max(snr_out - snr_in_ref, 0.0), 15.0) / 15.0 * 0.4
                    })
                except Exception as e:
                    print(f"  [跳过] IIR x{scale:g} order={order}, type={ftype}: {e}")
    
    # 选出最优结果
    if results:
        best_result = max(results, key=lambda x: x['score'])
    else:
        best_result = None
    
    return results, best_result


def plot_optimization_results(results, mod_type='', save_path=None):
    """
    绘制迭代优化参数扫描结果
    """
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=config.FIGURE_SIZE_LARGE)
    fig.suptitle(f'{mod_type} 滤波器迭代优化结果', fontsize=16, fontweight='bold')
    
    # 分离 FIR 和 IIR 结果
    fir_results = [r for r in results if r['type'] == 'FIR']
    iir_results = [r for r in results if r['type'] == 'IIR']
    
    # 图1: SNR 改善对比
    ax = axes[0, 0]
    labels_fir = [f"O{r['order']}\n{r['window'][:3]}" for r in fir_results]
    labels_iir = [f"O{r['order']}\n{r['window'][:3]}" for r in iir_results]
    
    x1 = np.arange(len(fir_results))
    x2 = np.arange(len(iir_results))
    
    if fir_results:
        ax.bar(x1, [r['snr_improvement'] for r in fir_results], 
              color='#2196F3', alpha=0.7, label='FIR')
        ax.set_xticks(x1)
        ax.set_xticklabels(labels_fir, fontsize=6, rotation=45)
    ax.set_title('SNR 改善量 (dB)', fontsize=12)
    ax.set_ylabel('ΔSNR (dB)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    
    # 图2: IIR SNR 改善
    ax = axes[0, 1]
    if iir_results:
        ax.bar(x2, [r['snr_improvement'] for r in iir_results], 
              color='#F44336', alpha=0.7, label='IIR')
        ax.set_xticks(x2)
        ax.set_xticklabels(labels_iir, fontsize=7, rotation=45)
    ax.set_title('IIR SNR 改善量 (dB)', fontsize=12)
    ax.set_ylabel('ΔSNR (dB)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    
    # 图3: 信号保真度对比
    ax = axes[1, 0]
    all_labels = [f"{'F' if r['type']=='FIR' else 'I'}{r['order']}" for r in results]
    all_corr = [r['correlation'] for r in results]
    colors = ['#2196F3' if r['type'] == 'FIR' else '#F44336' for r in results]
    ax.bar(range(len(results)), all_corr, color=colors, alpha=0.7)
    ax.set_title('信号保真度（相关系数）', fontsize=12)
    ax.set_ylabel('相关系数', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='目标线 0.9')
    ax.legend(fontsize=9)
    
    # 图4: 综合评分
    ax = axes[1, 1]
    scores = [r['score'] for r in results]
    best_idx = np.argmax(scores)
    bar_colors = ['#4CAF50' if i == best_idx else '#9E9E9E' for i in range(len(results))]
    ax.bar(range(len(results)), scores, color=bar_colors, alpha=0.8)
    ax.set_title(f'综合评分（最优：{results[best_idx]["type"]} O{results[best_idx]["order"]} {results[best_idx]["window"]}）', 
                fontsize=11)
    ax.set_ylabel('综合得分', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 优化结果图 → {save_path}")
    plt.close()


def plot_best_filter_analysis(b, a, cutoff, mod_type='', best_info=None,
                              save_path=None):
    """
    最优滤波器分析图：幅频特性 (含通带阴影) + Z域零极点分布
    展示迭代优化后零极点参数的最终形态
    """
    zeros, poles, gain = sig.tf2zpk(b, a)
    freqs, mag_db, phase_deg = filter_response(b, a)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sub_title = ''
    if best_info:
        sub_title = (f" ({best_info['type']} order={best_info['order']} "
                     f"{best_info['window']}, 带宽×{best_info.get('cutoff_scale', 1):g})")
    fig.suptitle(f'{mod_type} 迭代优化最优滤波器分析{sub_title}',
                 fontsize=14, fontweight='bold')
    
    # 左: 幅频特性 + 通带区域
    ax1.plot(freqs, mag_db, color='#2196F3', linewidth=1.5)
    lo, hi = cutoff
    ax1.axvspan(lo, hi, color='#4CAF50', alpha=0.12, label=f'设计通带 [{lo:.0f}, {hi:.0f}] Hz')
    ax1.set_title('幅频特性 |H(f)|', fontsize=12)
    ax1.set_xlabel('频率 (Hz)', fontsize=10)
    ax1.set_ylabel('增益 (dB)', fontsize=10)
    ax1.set_xlim(0, config.FS / 2)
    ax1.set_ylim(-100, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 右: 零极点分布
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.0, alpha=0.5,
             label='单位圆')
    if len(zeros) > 0:
        ax2.scatter(zeros.real, zeros.imag, marker='o', s=70,
                   facecolors='none', edgecolors='#2196F3', linewidths=1.8,
                   label=f'零点 ({len(zeros)}个)')
    if len(poles) > 0:
        ax2.scatter(poles.real, poles.imag, marker='x', s=70,
                   color='#F44336', linewidths=1.8,
                   label=f'极点 ({len(poles)}个)')
    ax2.set_title('Z域零极点分布', fontsize=12)
    ax2.set_xlabel('Re(z)', fontsize=10)
    ax2.set_ylabel('Im(z)', fontsize=10)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='upper left')
    r_max = max(
        np.max(np.abs(zeros)) if len(zeros) > 0 else 1,
        np.max(np.abs(poles)) if len(poles) > 0 else 1,
        1.2
    )
    ax2.set_xlim(-r_max * 1.25, r_max * 1.25)
    ax2.set_ylim(-r_max * 1.25, r_max * 1.25)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 最优滤波器分析图 → {save_path}")
    plt.close()


def plot_filter_comparison(clean, noisy, filtered_fir, filtered_iir, t, 
                           mod_type='', save_path=None):
    """
    绘制滤波前后波形对比图
    """
    display_samples = min(5 * config.SAMPLES_PER_BIT, len(t))
    t_ms = t[:display_samples] * 1000
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'{mod_type} 滤波效果对比', fontsize=16, fontweight='bold')
    
    # 原始纯净信号
    axes[0].plot(t_ms, clean[:display_samples], color='#2196F3', linewidth=0.8)
    axes[0].set_title('原始纯净信号', fontsize=12)
    axes[0].set_ylabel('幅度', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 含噪信号
    axes[1].plot(t_ms, noisy[:display_samples], color='#F44336', linewidth=0.8, alpha=0.8)
    axes[1].set_title(f'含噪信号 (SNR={config.SNR_DB}dB)', fontsize=12)
    axes[1].set_ylabel('幅度', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # FIR 滤波结果
    axes[2].plot(t_ms, clean[:display_samples], color='#2196F3', linewidth=0.5, alpha=0.4, label='原始')
    axes[2].plot(t_ms, filtered_fir[:display_samples], color='#4CAF50', linewidth=0.8, label='FIR滤波')
    axes[2].set_title('FIR 滤波结果', fontsize=12)
    axes[2].set_ylabel('幅度', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)
    
    # IIR 滤波结果
    axes[3].plot(t_ms, clean[:display_samples], color='#2196F3', linewidth=0.5, alpha=0.4, label='原始')
    axes[3].plot(t_ms, filtered_iir[:display_samples], color='#FF9800', linewidth=0.8, label='IIR滤波')
    axes[3].set_title('IIR 滤波结果', fontsize=12)
    axes[3].set_ylabel('幅度', fontsize=10)
    axes[3].set_xlabel('时间 (ms)', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 滤波对比图 → {save_path}")
    plt.close()


# ====================== LMS 自适应滤波 ======================

def lms_filter(desired, input_signal, mu=None, order=None, algorithm='lms'):
    """
    自适应滤波算法（LMS / NLMS）
    
    LMS:  w(n+1) = w(n) + 2μ·e(n)·x(n)
          步长受输入功率约束 (μ_max ≈ 1/(3·order·P_x))，收敛慢
    NLMS: w(n+1) = w(n) + μ·e(n)·x(n) / (‖x(n)‖² + δ)
          归一化步长与输入功率无关，μ∈(0,2]，收敛快且稳定
    
    Parameters
    ----------
    desired : ndarray
        期望信号 (参考信号/含噪信号)
    input_signal : ndarray
        输入信号 (噪声参考)
    mu : float
        步长 (学习率)
    order : int
        滤波器阶数
    algorithm : str
        'lms' 标准 LMS；'nlms' 归一化 LMS
    
    Returns
    -------
    output : ndarray
        滤波输出（估计的噪声）
    error : ndarray
        误差信号（去噪结果）
    weights : ndarray
        最终权值
    weight_history : ndarray
        权值演化历史
    mse_history : ndarray
        MSE 收敛曲线 (前 order 个样本为 NaN，未进入迭代)
    """
    if algorithm == 'nlms':
        mu = mu if mu is not None else config.NLMS_MU
        order = order or config.NLMS_ORDER
    else:
        mu = mu if mu is not None else config.LMS_MU
        order = order or config.LMS_ORDER
    
    N = len(desired)
    weights = np.zeros(order)
    output = np.zeros(N)
    error = np.zeros(N)
    weight_history = np.zeros((N, order))
    mse_history = np.full(N, np.nan)
    
    # 迭代尚未开始的前 order 个样本直接透传期望信号
    error[:order] = desired[:order]
    
    for n in range(order, N):
        # 输入向量 (含当前样本，参考信道与时延对齐，可估计当前噪声)
        x = input_signal[n:n - order:-1]
        
        # 滤波输出
        output[n] = np.dot(weights, x)
        
        # 误差
        error[n] = desired[n] - output[n]
        
        # 权值更新
        if algorithm == 'nlms':
            weights = weights + mu * error[n] * x / (np.dot(x, x) + 1e-12)
        else:
            weights = weights + 2 * mu * error[n] * x
        
        # 记录
        weight_history[n] = weights.copy()
        mse_history[n] = error[n] ** 2
    
    return output, error, weights, weight_history, mse_history


def lms_denoise(noisy_signal, noise_ref=None, mu=None, order=None, algorithm='lms'):
    """
    使用自适应滤波进行噪声消除
    
    当没有噪声参考信号时，使用延迟版本的含噪信号作为参考
    
    Parameters
    ----------
    noisy_signal : ndarray
        含噪信号
    noise_ref : ndarray, optional
        噪声参考信号
    mu, order : float, int
        自适应滤波参数
    algorithm : str
        'lms' 标准 LMS；'nlms' 归一化 LMS
    
    Returns
    -------
    denoised : ndarray
        去噪后信号
    mse_history : ndarray
        MSE 收敛历史
    """
    if noise_ref is None:
        # 使用延迟版本作为参考
        delay = order or (config.NLMS_ORDER if algorithm == 'nlms' else config.LMS_ORDER)
        noise_ref = np.roll(noisy_signal, delay)
        noise_ref[:delay] = 0
    
    output, error, weights, weight_history, mse_history = lms_filter(
        noisy_signal, noise_ref, mu, order, algorithm
    )
    
    # 误差信号即为去噪结果
    return error, mse_history, weights, weight_history


def evaluate_lms_mu(clean_signal, noisy_signal, noise_ref, mu_range=None,
                    order=None, algorithm='lms'):
    """
    评估不同自适应滤波步长对性能的影响
    
    Returns
    -------
    results : list of dict
        各步长的性能指标
    """
    if algorithm == 'nlms':
        default_range = config.NLMS_MU_RANGE
        order = order or config.NLMS_ORDER
    else:
        default_range = config.LMS_MU_RANGE
        order = order or config.LMS_ORDER
    mu_range = mu_range if mu_range is not None else default_range
    
    snr_in = compute_snr(clean_signal, noisy_signal)
    
    results = []
    for mu in mu_range:
        output, error, weights, weight_history, mse_history = lms_filter(
            noisy_signal, noise_ref, mu, order, algorithm
        )
        
        snr_out = compute_snr(clean_signal, error)
        corr = compute_correlation(clean_signal, error)
        
        # 收敛速度：MSE 首次降到稳态值 2 倍以下的样本数
        steady_state_mse = np.nanmean(mse_history[-max(len(mse_history) // 10, 1):])
        valid_mse = mse_history[order:]
        below = valid_mse < 2 * steady_state_mse
        if below.any():
            convergence_idx = int(np.argmax(below)) + order
        else:
            convergence_idx = len(mse_history) - order  # 观测窗内未收敛
        
        results.append({
            'algorithm': algorithm,
            'mu': mu,
            'snr_out': snr_out,
            'snr_improvement': snr_out - snr_in,
            'correlation': corr,
            'steady_state_mse': steady_state_mse,
            'convergence_samples': convergence_idx,
            'mse_history': mse_history,
            'error': error,
            'weight_history': weight_history
        })
    
    return results


def plot_lms_convergence(lms_results, mod_type='', save_path=None):
    """
    绘制自适应滤波详细分析图
    包含：MSE 收敛曲线、不同步长对比、权值演化、学习曲线
    """
    algo = lms_results[0].get('algorithm', 'lms') if lms_results else 'lms'
    algo_name = 'NLMS' if algo == 'nlms' else 'LMS'
    
    fig, axes = plt.subplots(2, 2, figsize=config.FIGURE_SIZE_LARGE)
    fig.suptitle(f'{mod_type} {algo_name} 自适应滤波详细分析', fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lms_results)))
    
    # 图1: MSE 收敛曲线（不同步长）
    ax = axes[0, 0]
    for i, res in enumerate(lms_results):
        # 跳过迭代开始前的 NaN 段
        mse_valid = res['mse_history'][~np.isnan(res['mse_history'])]
        # 对 MSE 进行移动平均以平滑显示
        window = min(100, len(mse_valid) // 20)
        if window > 1:
            mse_smooth = np.convolve(mse_valid, np.ones(window)/window, mode='valid')
        else:
            mse_smooth = mse_valid
        ax.semilogy(mse_smooth, color=colors[i], linewidth=1.0, 
                    label=f'μ={res["mu"]:.4f}', alpha=0.8)
    ax.set_title('MSE 收敛曲线（对数坐标）', fontsize=12)
    ax.set_xlabel('迭代次数', fontsize=10)
    ax.set_ylabel('MSE', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # 图2: SNR 改善 vs 步长
    ax = axes[0, 1]
    mus = [r['mu'] for r in lms_results]
    snrs = [r['snr_out'] for r in lms_results]
    corrs = [r['correlation'] for r in lms_results]
    ax2 = ax.twinx()
    bars = ax.bar(range(len(mus)), snrs, color='#2196F3', alpha=0.7, label='SNR')
    ax.set_xticks(range(len(mus)))
    ax.set_xticklabels([f'{mu:.4f}' for mu in mus], fontsize=8)
    ax.set_title('不同步长 μ 的性能对比', fontsize=12)
    ax.set_xlabel('步长 μ', fontsize=10)
    ax.set_ylabel('SNR (dB)', fontsize=10, color='#2196F3')
    
    ax2.plot(range(len(mus)), corrs, 'o-', color='#F44336', linewidth=1.5, label='相关系数')
    ax2.set_ylabel('相关系数', fontsize=10, color='#F44336')
    ax2.set_ylim(0, 1.1)
    
    ax.grid(True, alpha=0.3, axis='y')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    
    # 图3: 最优步长的权值演化
    ax = axes[1, 0]
    best_idx = np.argmax([r['snr_out'] for r in lms_results])
    best_wh = lms_results[best_idx]['weight_history']
    n_show = min(8, best_wh.shape[1])  # 最多显示8个权值
    for k in range(n_show):
        ax.plot(best_wh[:, k], linewidth=0.8, alpha=0.7, label=f'w[{k}]')
    ax.set_title(f'权值演化过程 (μ={lms_results[best_idx]["mu"]:.4f})', fontsize=12)
    ax.set_xlabel('迭代次数', fontsize=10)
    ax.set_ylabel('权值', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    
    # 图4: 学习曲线（分段平均 MSE）
    ax = axes[1, 1]
    best_res = lms_results[best_idx]
    mse = best_res['mse_history']
    mse = mse[~np.isnan(mse)]
    # 分段平均展示学习曲线
    n_segments = 50
    seg_len = len(mse) // n_segments
    if seg_len > 0:
        seg_mse = [np.mean(mse[i*seg_len:(i+1)*seg_len]) for i in range(n_segments)]
        ax.plot(range(n_segments), seg_mse, 'o-', color='#4CAF50', markersize=4, linewidth=1.5)
    ax.set_title(f'学习曲线（分段平均MSE）', fontsize=12)
    ax.set_xlabel('段序号', fontsize=10)
    ax.set_ylabel('平均 MSE', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] LMS 分析图 → {save_path}")
    plt.close()


def plot_lms_detailed_waveform(clean, noisy, denoised, t, 
                                mod_type='', mu=0.01, save_path=None):
    """
    绘制 LMS 去噪前后波形详细对比
    """
    display_samples = min(5 * config.SAMPLES_PER_BIT, len(t))
    t_ms = t[:display_samples] * 1000
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'{mod_type} LMS 自适应滤波效果 (μ={mu})', fontsize=16, fontweight='bold')
    
    # 含噪信号
    axes[0].plot(t_ms, noisy[:display_samples], color='#F44336', linewidth=0.8, alpha=0.8)
    axes[0].set_title('含噪信号', fontsize=12)
    axes[0].set_ylabel('幅度', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # LMS 去噪结果
    axes[1].plot(t_ms, clean[:display_samples], color='#2196F3', linewidth=0.5, alpha=0.4, label='原始')
    axes[1].plot(t_ms, denoised[:display_samples], color='#4CAF50', linewidth=0.8, label='LMS去噪')
    axes[1].set_title('LMS 自适应滤波结果', fontsize=12)
    axes[1].set_ylabel('幅度', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    
    # 误差（残差）
    residual = clean[:display_samples] - denoised[:display_samples]
    axes[2].plot(t_ms, residual, color='#FF9800', linewidth=0.8)
    axes[2].set_title('残差（原始 - 去噪）', fontsize=12)
    axes[2].set_ylabel('幅度', fontsize=10)
    axes[2].set_xlabel('时间 (ms)', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] LMS 波形对比图 → {save_path}")
    plt.close()
