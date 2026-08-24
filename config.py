# -*- coding: utf-8 -*-
"""
全局参数配置文件
统一管理采样率、载波频率、噪声强度等全局参数
"""

import os
import numpy as np

# ====================== 采样与时间参数 ======================
FS = 10000              # 采样率 (Hz)
DURATION = 0.1          # 单个符号持续时间 (s)
N_BITS = 20             # 信息比特数
SAMPLES_PER_BIT = int(FS * DURATION)  # 每比特采样点数

# ====================== ASK 调制参数 ======================
FC_ASK = 1000           # ASK 载波频率 (Hz)
ASK_AMPLITUDE_1 = 1.0   # ASK bit=1 幅度
ASK_AMPLITUDE_0 = 0.0   # ASK bit=0 幅度 (OOK)

# ====================== FSK 调制参数 ======================
FC_FSK_F1 = 800         # FSK bit=0 对应频率 (Hz)
FC_FSK_F2 = 1200        # FSK bit=1 对应频率 (Hz)

# ====================== PSK 调制参数 ======================
FC_PSK = 1000           # PSK 载波频率 (Hz)
PSK_PHASE_0 = 0         # PSK bit=0 对应相位 (rad)
PSK_PHASE_1 = np.pi     # PSK bit=1 对应相位 (rad)

# ====================== 信道参数 ======================
SNR_DB = 10             # 默认信噪比 (dB)
# SNR 扫描范围：包含深噪区间 (-25 ~ -15 dB) 用于体现 BER 与识别性能差异
SNR_RANGE = [-25, -20, -15, -10, 0, 10, 20]

# ====================== 滤波器参数 ======================
# FIR 参数
FIR_ORDERS = [16, 32, 64, 128]
FIR_WINDOWS = ['hamming', 'hann', 'blackman', 'kaiser']
FIR_DEFAULT_ORDER = 64
FIR_DEFAULT_WINDOW = 'hamming'

# IIR 参数
IIR_ORDERS = [2, 4, 6, 8]
IIR_TYPES = ['butter', 'cheby1', 'ellip']
IIR_DEFAULT_ORDER = 4
IIR_DEFAULT_TYPE = 'butter'

# 带通滤波器截止频率 (Hz)
BANDPASS_LOW_ASK = 600
BANDPASS_HIGH_ASK = 1400
BANDPASS_LOW_FSK = 500
BANDPASS_HIGH_FSK = 1500
BANDPASS_LOW_PSK = 600
BANDPASS_HIGH_PSK = 1400

# 迭代优化中截止频率的缩放系数扫描 (相对名义带宽)
CUTOFF_SCALES = [0.8, 1.0, 1.2]

# ====================== LMS/NLMS 自适应滤波参数 ======================
LMS_ORDER = 32          # LMS 滤波器阶数
LMS_MU = 0.01           # LMS 步长 (学习率)
LMS_MU_RANGE = [0.001, 0.005, 0.01, 0.05, 0.1]  # 步长扫描范围

# NLMS 归一化 LMS：步长与输入功率无关，μ∈(0,2]
NLMS_ORDER = 32         # NLMS 滤波器阶数
NLMS_MU = 0.02          # 默认归一化步长 (小步长降低稳态失调)
NLMS_MU_RANGE = [0.01, 0.05, 0.1, 0.5, 1.0]     # 归一化步长扫描范围
NLMS_SWEEP_MUS = [0.01, 0.05, 0.1, 0.5]         # SNR 扫描中按条件择优的候选步长

# ====================== SNR 扫描实验参数 ======================
SWEEP_TRIALS = 5        # 每个 SNR 点的蒙特卡洛试验次数

# ====================== 卷积平滑参数 ======================
SMOOTH_KERNELS = [5, 11, 21, 51]  # 平滑窗长列表

# ====================== 绘图与输出参数 ======================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
LOG_FILE = os.path.join(RESULTS_DIR, 'logs.txt')

# 图表样式
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'
FIGURE_SIZE = (14, 8)
FIGURE_SIZE_LARGE = (16, 12)

# 中文字体配置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 GUI 问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
matplotlib.rcParams['figure.dpi'] = FIGURE_DPI

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore', message='.*Badly conditioned filter coefficients.*')

# ====================== 随机种子 ======================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def ensure_dirs():
    """确保输出目录存在"""
    os.makedirs(PLOTS_DIR, exist_ok=True)


# 初始化时创建目录
ensure_dirs()
