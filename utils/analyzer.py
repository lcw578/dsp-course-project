# -*- coding: utf-8 -*-
"""
多域联合分析模块
负责：时域分析、频域分析(DFT/DTFT)、Z域分析(零极点)
实现"时域-频域-Z域"三维信号分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ====================== 时域分析 ======================

def convolution_smooth(signal_data, kernel_size=11):
    """
    时域卷积平滑
    使用矩形窗(均值滤波器)对信号进行卷积平滑
    
    Parameters
    ----------
    signal_data : ndarray
        输入信号
    kernel_size : int
        卷积核大小 (奇数)
    
    Returns
    -------
    smoothed : ndarray
        平滑后的信号 (same 模式)
    """
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(signal_data, kernel, mode='same')
    return smoothed


def compute_envelope(signal_data):
    """
    计算信号包络 (利用希尔伯特变换)
    
    Parameters
    ----------
    signal_data : ndarray
        输入信号
    
    Returns
    -------
    envelope : ndarray
        信号包络
    """
    analytic = sig.hilbert(signal_data)
    envelope = np.abs(analytic)
    return envelope


def time_domain_features(signal_data):
    """
    提取时域统计特征
    
    Returns
    -------
    features : dict
        包含均值、方差、峰值、包络方差等统计量
    """
    envelope = compute_envelope(signal_data)
    features = {
        'mean': np.mean(signal_data),
        'variance': np.var(signal_data),
        'rms': np.sqrt(np.mean(signal_data ** 2)),
        'peak': np.max(np.abs(signal_data)),
        'envelope_mean': np.mean(envelope),
        'envelope_var': np.var(envelope),
        'zero_crossings': np.sum(np.diff(np.sign(signal_data)) != 0)
    }
    return features


# ====================== 频域分析 ======================

def compute_dft(signal_data, N=None):
    """
    计算 N 点 DFT 离散频谱
    
    Parameters
    ----------
    signal_data : ndarray
        输入信号
    N : int
        DFT 点数 (默认为信号长度)
    
    Returns
    -------
    freqs : ndarray
        频率轴 (Hz)
    magnitude : ndarray
        幅度谱
    phase : ndarray
        相位谱
    """
    if N is None:
        N = len(signal_data)
    
    # 使用 FFT 高效计算 DFT
    X = np.fft.fft(signal_data, n=N)
    freqs = np.fft.fftfreq(N, d=1.0 / config.FS)
    
    # 只取正频率部分
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    X = X[positive_mask]
    
    # 单边谱归一化：DC 与 Nyquist 分量不加倍，其余加倍
    magnitude = np.abs(X) / N
    double_mask = (freqs > 0) & (freqs < config.FS / 2)
    magnitude[double_mask] *= 2
    phase = np.angle(X)
    
    return freqs, magnitude, phase


def compute_dtft(signal_data, n_points=4096):
    """
    计算 DTFT 连续频谱 (通过密集 FFT 逼近)
    
    DTFT 定义：X(ω) = Σ x[n] * e^(-jωn)
    通过零填充的 FFT 在密集频率点上近似 DTFT
    
    Parameters
    ----------
    signal_data : ndarray
        输入信号
    n_points : int
        频率采样点数 (越大越逼近连续)
    
    Returns
    -------
    omega : ndarray
        数字角频率 ω ∈ [0, π]
    magnitude : ndarray
        幅度谱 |X(ω)|
    phase : ndarray
        相位谱 ∠X(ω)
    """
    # 零填充到 n_points 实现密集频率采样
    N = max(n_points, len(signal_data))
    X = np.fft.fft(signal_data, n=N)
    
    # 取正频率部分 [0, π]
    half = N // 2
    omega = np.linspace(0, np.pi, half)
    X_half = X[:half]
    
    magnitude = np.abs(X_half)
    phase = np.angle(X_half)
    
    return omega, magnitude, phase


def compute_zpk(b, a):
    """
    Z 变换零极点计算
    
    Parameters
    ----------
    b : ndarray
        系统函数分子多项式系数
    a : ndarray
        系统函数分母多项式系数
    
    Returns
    -------
    zeros : ndarray
        零点
    poles : ndarray
        极点
    gain : float
        增益
    """
    zeros, poles, gain = sig.tf2zpk(b, a)
    return zeros, poles, gain


def signal_to_zpk(signal_data, order=10):
    """
    对信号进行 AR 模型拟合，提取 Z 域零极点
    用于分析不同调制方式的 Z 域特征
    
    Parameters
    ----------
    signal_data : ndarray
        输入信号 (取一段)
    order : int
        AR 模型阶数
    
    Returns
    -------
    zeros : ndarray
        零点
    poles : ndarray
        极点
    gain : float
        增益
    """
    # 使用 Yule-Walker 方法拟合 AR 模型
    # 计算自相关
    seg = signal_data[:min(1000, len(signal_data))]
    autocorr = np.correlate(seg, seg, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # 信号过弱 (如 ASK 静音段) 时自相关无意义，返回简单系统
    if autocorr[0] < 1e-12:
        b = np.array([1.0])
        a = np.array([1.0, -0.5])
        return sig.tf2zpk(b, a)
    
    autocorr = autocorr / autocorr[0]  # 归一化
    
    # Levinson-Durbin 递推求 AR 系数
    r = autocorr[1:order+1]
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = autocorr[abs(i - j)]
    
    try:
        a_coeffs = np.linalg.solve(R, r)
        a = np.concatenate(([1], -a_coeffs))
        b = np.array([1.0])
        zeros, poles, gain = sig.tf2zpk(b, a)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，返回简单结果
        b = np.array([1.0])
        a = np.array([1.0, -0.5])
        zeros, poles, gain = sig.tf2zpk(b, a)
    
    return zeros, poles, gain


# ====================== 绘图函数 ======================

def plot_time_domain_signals(signals_dict, save_path=None):
    """
    绘制 ASK/FSK/PSK 纯净信号 vs 含噪信号的时域波形对比
    
    Parameters
    ----------
    signals_dict : dict
        由 generator.generate_all_signals() 返回的信号字典
    save_path : str
        图片保存路径
    """
    mod_types = list(signals_dict.keys())
    n_mods = len(mod_types)
    
    fig, axes = plt.subplots(n_mods, 2, figsize=config.FIGURE_SIZE_LARGE)
    fig.suptitle('时域波形分析：纯净信号 vs 含噪信号', fontsize=16, fontweight='bold')
    
    # 只显示前5个比特的波形，便于观察
    display_samples = min(5 * config.SAMPLES_PER_BIT, len(list(signals_dict.values())[0]['t']))
    
    for idx, mod_type in enumerate(mod_types):
        data = signals_dict[mod_type]
        t = data['t'][:display_samples] * 1000  # 转为毫秒
        
        # 纯净信号
        axes[idx, 0].plot(t, data['clean'][:display_samples], 
                         color='#2196F3', linewidth=0.8, label='纯净信号')
        axes[idx, 0].set_title(f'{mod_type} 纯净信号', fontsize=12)
        axes[idx, 0].set_ylabel('幅度', fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend(loc='upper right', fontsize=9)
        
        # 含噪信号
        axes[idx, 1].plot(t, data['noisy'][:display_samples], 
                         color='#F44336', linewidth=0.8, alpha=0.8, label='含噪信号')
        axes[idx, 1].plot(t, data['clean'][:display_samples], 
                         color='#2196F3', linewidth=0.5, alpha=0.4, label='原始信号')
        axes[idx, 1].set_title(f'{mod_type} 含噪信号 (SNR={config.SNR_DB}dB)', fontsize=12)
        axes[idx, 1].set_ylabel('幅度', fontsize=10)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend(loc='upper right', fontsize=9)
    
    for ax in axes[-1]:
        ax.set_xlabel('时间 (ms)', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 时域波形图 → {save_path}")
    plt.close()


def plot_convolution_smooth(signals_dict, kernel_sizes=None, save_path=None):
    """
    绘制不同卷积核大小的平滑效果对比
    """
    kernel_sizes = kernel_sizes or config.SMOOTH_KERNELS
    mod_types = list(signals_dict.keys())
    n_mods = len(mod_types)
    
    fig, axes = plt.subplots(n_mods, len(kernel_sizes), figsize=(20, 4 * n_mods))
    fig.suptitle('时域卷积平滑效果对比（不同窗长）', fontsize=16, fontweight='bold')
    
    display_samples = min(5 * config.SAMPLES_PER_BIT, len(list(signals_dict.values())[0]['t']))
    
    for i, mod_type in enumerate(mod_types):
        data = signals_dict[mod_type]
        t = data['t'][:display_samples] * 1000
        
        for j, ks in enumerate(kernel_sizes):
            smoothed = convolution_smooth(data['noisy'], ks)
            
            ax = axes[i, j] if n_mods > 1 else axes[j]
            ax.plot(t, data['noisy'][:display_samples], 
                   color='#BDBDBD', linewidth=0.5, alpha=0.6, label='含噪')
            ax.plot(t, smoothed[:display_samples], 
                   color='#4CAF50', linewidth=1.0, label=f'平滑(k={ks})')
            ax.plot(t, data['clean'][:display_samples], 
                   color='#2196F3', linewidth=0.5, alpha=0.5, label='原始')
            ax.set_title(f'{mod_type}, 窗长={ks}', fontsize=10)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel('幅度', fontsize=9)
            if i == n_mods - 1:
                ax.set_xlabel('时间 (ms)', fontsize=9)
            ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 卷积平滑对比图 → {save_path}")
    plt.close()


def plot_dft_spectrum(signals_dict, save_path=None):
    """
    绘制 DFT 离散频谱对比 (ASK/FSK/PSK)
    """
    mod_types = list(signals_dict.keys())
    n_mods = len(mod_types)
    colors = {'ASK': '#2196F3', 'FSK': '#F44336', 'PSK': '#4CAF50'}
    
    fig, axes = plt.subplots(n_mods, 2, figsize=config.FIGURE_SIZE_LARGE)
    fig.suptitle('DFT 离散频谱分析', fontsize=16, fontweight='bold')
    
    for idx, mod_type in enumerate(mod_types):
        data = signals_dict[mod_type]
        color = colors.get(mod_type, '#9C27B0')
        
        # 纯净信号频谱
        freqs_c, mag_c, phase_c = compute_dft(data['clean'])
        # 含噪信号频谱
        freqs_n, mag_n, phase_n = compute_dft(data['noisy'])
        
        # 幅度谱
        freq_mask = freqs_c <= 3000  # 只显示到3kHz
        axes[idx, 0].plot(freqs_c[freq_mask], mag_c[freq_mask], 
                         color=color, linewidth=1.0, label='纯净信号')
        axes[idx, 0].plot(freqs_n[freq_mask], mag_n[freq_mask], 
                         color='gray', linewidth=0.5, alpha=0.5, label='含噪信号')
        axes[idx, 0].set_title(f'{mod_type} DFT 幅度谱', fontsize=12)
        axes[idx, 0].set_ylabel('幅度', fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend(fontsize=9)
        
        # 相位谱 (纯净信号)
        axes[idx, 1].plot(freqs_c[freq_mask], phase_c[freq_mask], 
                         color=color, linewidth=0.5, alpha=0.7)
        axes[idx, 1].set_title(f'{mod_type} DFT 相位谱', fontsize=12)
        axes[idx, 1].set_ylabel('相位 (rad)', fontsize=10)
        axes[idx, 1].grid(True, alpha=0.3)
    
    for ax in axes[-1]:
        ax.set_xlabel('频率 (Hz)', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] DFT 频谱图 → {save_path}")
    plt.close()


def plot_dtft_spectrum(signals_dict, save_path=None):
    """
    绘制 DTFT 连续频谱对比
    """
    mod_types = list(signals_dict.keys())
    n_mods = len(mod_types)
    colors = {'ASK': '#2196F3', 'FSK': '#F44336', 'PSK': '#4CAF50'}
    
    fig, axes = plt.subplots(n_mods, 2, figsize=config.FIGURE_SIZE_LARGE)
    fig.suptitle('DTFT 连续频谱分析（零填充FFT逼近）', fontsize=16, fontweight='bold')
    
    for idx, mod_type in enumerate(mod_types):
        data = signals_dict[mod_type]
        color = colors.get(mod_type, '#9C27B0')
        
        # 取一段信号计算 DTFT
        segment = data['clean'][:2 * config.SAMPLES_PER_BIT]
        omega, mag, phase = compute_dtft(segment, n_points=8192)
        
        segment_n = data['noisy'][:2 * config.SAMPLES_PER_BIT]
        omega_n, mag_n, phase_n = compute_dtft(segment_n, n_points=8192)
        
        # 幅度谱
        axes[idx, 0].plot(omega / np.pi, mag, 
                         color=color, linewidth=1.0, label='纯净信号')
        axes[idx, 0].plot(omega_n / np.pi, mag_n, 
                         color='gray', linewidth=0.5, alpha=0.4, label='含噪信号')
        axes[idx, 0].set_title(f'{mod_type} DTFT 幅度谱 |X(ω)|', fontsize=12)
        axes[idx, 0].set_ylabel('幅度', fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend(fontsize=9)
        
        # 相位谱
        axes[idx, 1].plot(omega / np.pi, phase, 
                         color=color, linewidth=0.5, alpha=0.7)
        axes[idx, 1].set_title(f'{mod_type} DTFT 相位谱 ∠X(ω)', fontsize=12)
        axes[idx, 1].set_ylabel('相位 (rad)', fontsize=10)
        axes[idx, 1].grid(True, alpha=0.3)
    
    for ax in axes[-1]:
        ax.set_xlabel('归一化频率 (×π rad/sample)', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] DTFT 频谱图 → {save_path}")
    plt.close()


def plot_zero_pole_map(signals_dict, save_path=None):
    """
    绘制 Z 变换零极点分布图（含单位圆）
    分析不同调制方式的 Z 域特征
    """
    mod_types = list(signals_dict.keys())
    n_mods = len(mod_types)
    colors = {'ASK': '#2196F3', 'FSK': '#F44336', 'PSK': '#4CAF50'}
    
    fig, axes = plt.subplots(1, n_mods, figsize=(6 * n_mods, 6))
    if n_mods == 1:
        axes = [axes]
    fig.suptitle('Z 变换零极点分布图', fontsize=16, fontweight='bold')
    
    for idx, mod_type in enumerate(mod_types):
        data = signals_dict[mod_type]
        color = colors.get(mod_type, '#9C27B0')
        ax = axes[idx]
        
        # 对信号进行 AR 模型拟合，提取零极点
        zeros, poles, gain = signal_to_zpk(data['clean'], order=12)
        
        # 绘制单位圆
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.0, alpha=0.5, label='单位圆')
        
        # 绘制零点 (o)
        if len(zeros) > 0:
            ax.scatter(zeros.real, zeros.imag, marker='o', s=80, 
                      facecolors='none', edgecolors=color, linewidths=2, 
                      label=f'零点 ({len(zeros)}个)', zorder=5)
        
        # 绘制极点 (x)
        if len(poles) > 0:
            ax.scatter(poles.real, poles.imag, marker='x', s=80, 
                      color=color, linewidths=2, 
                      label=f'极点 ({len(poles)}个)', zorder=5)
        
        ax.set_title(f'{mod_type} 零极点图', fontsize=13)
        ax.set_xlabel('实部 Re(z)', fontsize=10)
        ax.set_ylabel('虚部 Im(z)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        # 设置合适的显示范围
        max_range = max(
            np.max(np.abs(poles.real)) if len(poles) > 0 else 1,
            np.max(np.abs(poles.imag)) if len(poles) > 0 else 1,
            1.2
        )
        ax.set_xlim(-max_range * 1.3, max_range * 1.3)
        ax.set_ylim(-max_range * 1.3, max_range * 1.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 零极点分布图 → {save_path}")
    plt.close()


def plot_dft_dtft_comparison(signals_dict, save_path=None):
    """
    三种变换应用差异对比图：
    - 左列: DTFT 连续谱包络 (零填充逼近) 与 DFT 离散谱线叠加
      直观展示 "DFT 是对 DTFT 的等间隔采样" 及频率分辨率差异
    - 右列: Z 域零极点分布 (AR 建模)，体现 Z 变换的系统级谐振特征
    """
    mod_types = list(signals_dict.keys())
    n_mods = len(mod_types)
    colors = {'ASK': '#2196F3', 'FSK': '#F44336', 'PSK': '#4CAF50'}
    
    fig, axes = plt.subplots(n_mods, 2, figsize=(16, 4.5 * n_mods))
    if n_mods == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('三种变换应用差异对比：DFT离散谱线 = DTFT连续谱的等间隔采样 | Z域零极点谐振',
                 fontsize=15, fontweight='bold')
    
    freq_mask_max = 3000
    
    for idx, mod_type in enumerate(mod_types):
        data = signals_dict[mod_type]
        color = colors.get(mod_type, '#9C27B0')
        
        # ---- 左列: DTFT 连续谱 vs DFT 谱线 ----
        ax = axes[idx, 0]
        segment = data['clean'][:config.SAMPLES_PER_BIT]
        omega_dtft, mag_dtft, _ = compute_dtft(segment, n_points=4096)
        fs_axis_dtft = omega_dtft / np.pi * (config.FS / 2)
        
        freqs_dft, mag_dft, _ = compute_dft(segment)
        
        m_dtft = fs_axis_dtft <= freq_mask_max
        m_dft = freqs_dft <= freq_mask_max
        
        ax.plot(fs_axis_dtft[m_dtft], mag_dtft[m_dtft] / np.max(mag_dtft + 1e-15),
                color=color, linewidth=1.5, alpha=0.85, label='DTFT 连续谱')
        markerline, stemlines, _ = ax.stem(freqs_dft[m_dft],
                                           mag_dft[m_dft] / (np.max(mag_dft) + 1e-15),
                                           basefmt=' ')
        plt.setp(stemlines, color='gray', linewidth=0.6, alpha=0.6)
        plt.setp(markerline, color='#333333', markersize=2.5)
        
        dft_res = config.FS / len(segment)
        ax.set_title(f'{mod_type}  DFT谱线间隔 = FS/N = {dft_res:.1f} Hz',
                     fontsize=12)
        ax.set_xlabel('频率 (Hz)', fontsize=10)
        ax.set_ylabel('归一化幅度', fontsize=10)
        ax.set_xlim(0, freq_mask_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        
        # ---- 右列: Z 域零极点 ----
        ax = axes[idx, 1]
        zeros, poles, gain = signal_to_zpk(data['clean'], order=12)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.0, alpha=0.5,
                label='单位圆')
        strong_poles = poles[np.abs(poles) > 0.6] if len(poles) > 0 else poles
        if len(zeros) > 0:
            ax.scatter(zeros.real, zeros.imag, marker='o', s=60,
                      facecolors='none', edgecolors=color, linewidths=1.5,
                      alpha=0.6, label='零点')
        if len(strong_poles) > 0:
            ax.scatter(strong_poles.real, strong_poles.imag, marker='x', s=90,
                       color=color, linewidths=2,
                       label=f'强极点 ({len(strong_poles)}个)')
            for p in strong_poles:
                if p.imag > 0:
                    f_phys = np.arctan2(p.imag, p.real) / np.pi * config.FS / 2
                    if f_phys < freq_mask_max:
                        ax.annotate(f'{f_phys:.0f}Hz',
                                    xy=(p.real, p.imag),
                                    xytext=(p.real * 1.15, p.imag * 1.15),
                                    fontsize=8, color='#555555')
        ax.set_title(f'{mod_type}  Z域极点 → 谐振频率', fontsize=12)
        ax.set_xlabel('Re(z)', fontsize=10)
        ax.set_ylabel('Im(z)', fontsize=10)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper left')
        r_max = max(np.max(np.abs(poles)) if len(poles) > 0 else 1, 1.2)
        ax.set_xlim(-r_max * 1.3, r_max * 1.3)
        ax.set_ylim(-r_max * 1.3, r_max * 1.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 三种变换对比图 → {save_path}")
    plt.close()


def format_system_function(b, a, max_terms=8):
    """
    将滤波器系统函数 H(z) 格式化为可读多项式字符串
    
    H(z) = (b0 + b1·z⁻¹ + ...) / (a0 + a1·z⁻¹ + ...)
    
    Parameters
    ----------
    b, a : ndarray
        系统函数分子/分母系数
    max_terms : int
        每侧最多显示的项数，超出以省略号截断
    
    Returns
    -------
    text : str
        系统函数表达式（多行）
    """
    def _poly(coeffs):
        terms = []
        shown = 0
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-12:
                continue
            if shown >= max_terms:
                terms.append('...')
                break
            mag = abs(c)
            body = f'{mag:.4g}'
            if i == 0:
                terms.append(body)
            elif i == 1:
                terms.append(f'{c:+.4g}z⁻¹')
            else:
                terms.append(f'{c:+.4g}z⁻{i}')
            shown += 1
        return ' '.join(terms) if terms else '0'
    
    lines = [
        'H(z) = N(z) / D(z)',
        f'  N(z) = {_poly(b)}',
        f'  D(z) = {_poly(a)}',
        f'  (分子 {len(np.nonzero(np.abs(b)>1e-12)[0])} 项, '
        f'分母 {len(np.nonzero(np.abs(a)>1e-12)[0])} 项)'
    ]
    return '\n'.join(lines)


def plot_multi_domain_comparison(signals_dict, save_path=None):
    """
    绘制多域联合分析综合对比图
    时域波形 + DFT频谱 + 零极点 在同一张图中展示
    """
    mod_types = list(signals_dict.keys())
    n_mods = len(mod_types)
    colors = {'ASK': '#2196F3', 'FSK': '#F44336', 'PSK': '#4CAF50'}
    
    fig, axes = plt.subplots(n_mods, 3, figsize=(18, 5 * n_mods))
    fig.suptitle('多域联合分析综合对比（时域-频域-Z域）', fontsize=18, fontweight='bold')
    
    display_samples = min(3 * config.SAMPLES_PER_BIT, len(list(signals_dict.values())[0]['t']))
    
    for idx, mod_type in enumerate(mod_types):
        data = signals_dict[mod_type]
        color = colors.get(mod_type, '#9C27B0')
        
        # 列1: 时域
        t = data['t'][:display_samples] * 1000
        axes[idx, 0].plot(t, data['clean'][:display_samples], color=color, linewidth=0.8)
        axes[idx, 0].set_title(f'{mod_type} 时域波形', fontsize=12)
        axes[idx, 0].set_ylabel('幅度', fontsize=10)
        axes[idx, 0].set_xlabel('时间 (ms)', fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        
        # 列2: 频域 (DFT)
        freqs, mag, _ = compute_dft(data['clean'])
        freq_mask = freqs <= 2500
        axes[idx, 1].plot(freqs[freq_mask], mag[freq_mask], color=color, linewidth=1.0)
        axes[idx, 1].set_title(f'{mod_type} DFT 幅度谱', fontsize=12)
        axes[idx, 1].set_ylabel('幅度', fontsize=10)
        axes[idx, 1].set_xlabel('频率 (Hz)', fontsize=10)
        axes[idx, 1].grid(True, alpha=0.3)
        
        # 列3: Z域 (零极点)
        zeros, poles, gain = signal_to_zpk(data['clean'], order=12)
        theta = np.linspace(0, 2 * np.pi, 200)
        axes[idx, 2].plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.0, alpha=0.5)
        if len(zeros) > 0:
            axes[idx, 2].scatter(zeros.real, zeros.imag, marker='o', s=60, 
                                facecolors='none', edgecolors=color, linewidths=1.5, zorder=5)
        if len(poles) > 0:
            axes[idx, 2].scatter(poles.real, poles.imag, marker='x', s=60, 
                                color=color, linewidths=1.5, zorder=5)
        axes[idx, 2].set_title(f'{mod_type} 零极点分布', fontsize=12)
        axes[idx, 2].set_xlabel('Re(z)', fontsize=10)
        axes[idx, 2].set_ylabel('Im(z)', fontsize=10)
        axes[idx, 2].set_aspect('equal')
        axes[idx, 2].grid(True, alpha=0.3)
        axes[idx, 2].axhline(y=0, color='k', linewidth=0.5)
        axes[idx, 2].axvline(x=0, color='k', linewidth=0.5)
        max_range = max(np.max(np.abs(poles.real)) if len(poles) > 0 else 1,
                       np.max(np.abs(poles.imag)) if len(poles) > 0 else 1, 1.2)
        axes[idx, 2].set_xlim(-max_range * 1.3, max_range * 1.3)
        axes[idx, 2].set_ylim(-max_range * 1.3, max_range * 1.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"[保存] 多域联合对比图 → {save_path}")
    plt.close()
