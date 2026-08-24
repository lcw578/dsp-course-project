# -*- coding: utf-8 -*-
"""
多变换域联合的信号调制特征识别与滤波优化系统
主程序入口：串联完整 DSP 处理流程

Step 1: 生成 ASK/FSK/PSK 纯净信号 + 含噪信号
Step 2: 时域分析（波形对比 + 卷积平滑）
Step 3: 多变换域联合分析（DFT + DTFT + Z变换零极点 + 三种变换应用差异对比）
Step 4: 滤波器设计（FIR/IIR）+ 系统函数 + 幅频相频特性
Step 5: 滤波器迭代优化（阶数×窗函数×截止频率）+ 最优滤波器零极点分析
Step 6: NLMS 归一化自适应滤波（创新拓展）
Step 7: 信号特征识别（ASK/FSK/PSK 自动分类）
Step 8: 性能评估（SNR/保真度/误码率BER/运算复杂度）
Step 9: SNR 扫描实验（蒙特卡洛：滤波性能 + 识别准确率）
Step 10: 单一域 vs 多域联合识别消融实验
Step 11: 报告生成
"""

import numpy as np
import os
import sys

# 修复 Windows 控制台 GBK 编码问题
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import time

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
from utils import generator, analyzer, filter_design, recognizer


def separator(title):
    """打印分隔线"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    """主函数：执行完整的 DSP 处理流程"""
    
    start_time = time.time()
    print("=" * 70)
    print("  多变换域联合的信号调制特征识别与滤波优化系统")
    print("  Multi-Transform Domain Joint Signal Modulation Recognition System")
    print("=" * 70)
    
    # ================================================================
    # Step 1: 生成 ASK/FSK/PSK 信号
    # ================================================================
    separator("Step 1: 调制信号建模与生成")
    
    print(f"  采样率: {config.FS} Hz")
    print(f"  比特数: {config.N_BITS}")
    print(f"  每比特采样点数: {config.SAMPLES_PER_BIT}")
    print(f"  信噪比: {config.SNR_DB} dB")
    
    signals = generator.generate_all_signals(
        n_bits=config.N_BITS, snr_db=config.SNR_DB, seed=config.RANDOM_SEED
    )
    
    for mod_type, data in signals.items():
        print(f"  [{mod_type}] 信号长度: {len(data['clean'])} 样本, "
              f"时长: {len(data['clean'])/config.FS*1000:.1f} ms")
    
    print("  ✓ 信号生成完成")
    
    # ================================================================
    # Step 2: 时域分析
    # ================================================================
    separator("Step 2: 时域波形分析与卷积平滑")
    
    # 2.1 时域波形对比
    analyzer.plot_time_domain_signals(
        signals, 
        save_path=os.path.join(config.PLOTS_DIR, '01_time_domain_signals.png')
    )
    
    # 2.2 卷积平滑效果对比
    analyzer.plot_convolution_smooth(
        signals, 
        kernel_sizes=config.SMOOTH_KERNELS,
        save_path=os.path.join(config.PLOTS_DIR, '02_convolution_smooth.png')
    )
    
    # 时域特征提取
    print("\n  时域特征统计：")
    for mod_type, data in signals.items():
        features = analyzer.time_domain_features(data['clean'])
        print(f"  [{mod_type}] 均值={features['mean']:.4f}, "
              f"方差={features['variance']:.4f}, "
              f"RMS={features['rms']:.4f}, "
              f"包络方差={features['envelope_var']:.4f}, "
              f"过零数={features['zero_crossings']}")
    
    print("  ✓ 时域分析完成")
    
    # ================================================================
    # Step 3: 多域联合分析
    # ================================================================
    separator("Step 3: 多变换域联合特征分析")
    
    # 3.1 DFT 离散频谱
    analyzer.plot_dft_spectrum(
        signals,
        save_path=os.path.join(config.PLOTS_DIR, '03_dft_spectrum.png')
    )
    print("  ✓ DFT 离散频谱分析完成")
    
    # 3.2 DTFT 连续频谱
    analyzer.plot_dtft_spectrum(
        signals,
        save_path=os.path.join(config.PLOTS_DIR, '04_dtft_spectrum.png')
    )
    print("  ✓ DTFT 连续频谱分析完成")
    
    # 3.3 Z 变换零极点
    analyzer.plot_zero_pole_map(
        signals,
        save_path=os.path.join(config.PLOTS_DIR, '05_zero_pole_map.png')
    )
    print("  ✓ Z 变换零极点分析完成")
    
    # 3.4 多域联合综合对比
    analyzer.plot_multi_domain_comparison(
        signals,
        save_path=os.path.join(config.PLOTS_DIR, '13_multi_domain_comparison.png')
    )
    print("  ✓ 多域联合对比图生成完成")
    
    # 3.5 三种变换应用差异对比 (DFT ⊂ DTFT 采样关系 + Z域谐振特征)
    analyzer.plot_dft_dtft_comparison(
        signals,
        save_path=os.path.join(config.PLOTS_DIR, '15_transform_comparison.png')
    )
    print("\n  三种变换分工：")
    print("    • DTFT(连续谱) → 观察谱形与带宽，零填充FFT逼近")
    print("    • DFT(离散谱线) → 频率分辨率 FS/N，支撑峰值检测与功率分布统计")
    print("    • Z变换(零极点) → 极点角位置对应谐振频率，揭示系统级特征")
    
    # ================================================================
    # Step 4: 滤波器设计与特性分析
    # ================================================================
    separator("Step 4: FIR/IIR 滤波器设计与频率响应特性")
    
    # 定义各调制方式的带通参数
    bandpass_params = {
        'ASK': (config.BANDPASS_LOW_ASK, config.BANDPASS_HIGH_ASK),
        'FSK': (config.BANDPASS_LOW_FSK, config.BANDPASS_HIGH_FSK),
        'PSK': (config.BANDPASS_LOW_PSK, config.BANDPASS_HIGH_PSK),
    }
    
    # 4.1 设计并绘制 FIR 滤波器
    print("\n  FIR 滤波器设计 (窗函数法):")
    fir_filters = {}
    for mod_type in signals:
        low, high = bandpass_params[mod_type]
        b, a = filter_design.design_fir(
            config.FIR_DEFAULT_ORDER, low, high, config.FS, config.FIR_DEFAULT_WINDOW
        )
        fir_filters[mod_type] = (b, a)
        print(f"  [{mod_type}] FIR order={config.FIR_DEFAULT_ORDER}, "
              f"window={config.FIR_DEFAULT_WINDOW}, "
              f"cutoff=[{low}, {high}] Hz, 系数数量={len(b)}")
    
    # 绘制 FIR 频率响应 (以 FSK 为例展示完整特性)
    b_fsk, a_fsk = fir_filters['FSK']
    filter_design.plot_filter_response(
        b_fsk, a_fsk, config.FS,
        title=f'(FIR, order={config.FIR_DEFAULT_ORDER}, {config.FIR_DEFAULT_WINDOW})',
        save_path=os.path.join(config.PLOTS_DIR, '06_fir_response.png')
    )
    print("\n  [系统函数推导] FIR 带通 (FSK):")
    print(analyzer.format_system_function(b_fsk, a_fsk))
    
    # 4.2 设计并绘制 IIR 滤波器
    print("\n  IIR 滤波器设计:")
    iir_filters = {}
    for mod_type in signals:
        low, high = bandpass_params[mod_type]
        b, a = filter_design.design_iir(
            config.IIR_DEFAULT_ORDER, low, high, config.FS, config.IIR_DEFAULT_TYPE
        )
        iir_filters[mod_type] = (b, a)
        print(f"  [{mod_type}] IIR order={config.IIR_DEFAULT_ORDER}, "
              f"type={config.IIR_DEFAULT_TYPE}, "
              f"cutoff=[{low}, {high}] Hz")
    
    # 绘制 IIR 频率响应
    b_fsk_iir, a_fsk_iir = iir_filters['FSK']
    filter_design.plot_filter_response(
        b_fsk_iir, a_fsk_iir, config.FS,
        title=f'(IIR, order={config.IIR_DEFAULT_ORDER}, {config.IIR_DEFAULT_TYPE})',
        save_path=os.path.join(config.PLOTS_DIR, '07_iir_response.png')
    )
    print("\n  [系统函数推导] IIR 带通 (FSK):")
    print(analyzer.format_system_function(b_fsk_iir, a_fsk_iir))
    
    # 绘制滤波器零极点图
    filter_design.plot_filter_zero_pole(
        b_fsk_iir, a_fsk_iir,
        title=f'(IIR Butterworth, order={config.IIR_DEFAULT_ORDER})',
        save_path=os.path.join(config.PLOTS_DIR, '05b_filter_zero_pole.png')
    )
    
    print("  ✓ 滤波器设计完成")
    
    # 4.3 滤波并绘制对比
    print("\n  执行滤波操作...")
    filtered_results = {}  # {mod_type: {method: (signal, time)}}
    
    for mod_type in signals:
        data = signals[mod_type]
        filtered_results[mod_type] = {}
        
        # FIR 滤波
        b, a = fir_filters[mod_type]
        t0 = time.time()
        fir_filtered = filter_design.apply_filter(b, a, data['noisy'])
        t_fir = time.time() - t0
        filtered_results[mod_type]['FIR最优'] = (fir_filtered, t_fir)
        
        # IIR 滤波
        b, a = iir_filters[mod_type]
        t0 = time.time()
        iir_filtered = filter_design.apply_filter(b, a, data['noisy'])
        t_iir = time.time() - t0
        filtered_results[mod_type]['IIR最优'] = (iir_filtered, t_iir)
    
    # 绘制滤波前后对比 (每种调制方式各一张图)
    for mod_type in signals:
        data = signals[mod_type]
        fir_f = filtered_results[mod_type]['FIR最优'][0]
        iir_f = filtered_results[mod_type]['IIR最优'][0]
        
        filter_design.plot_filter_comparison(
            data['clean'], data['noisy'], fir_f, iir_f, data['t'],
            mod_type=mod_type,
            save_path=os.path.join(config.PLOTS_DIR, f'08_{mod_type}_filter_comparison.png')
        )
    
    print("  ✓ 滤波与对比图生成完成")
    
    # ================================================================
    # Step 5: 滤波器迭代优化
    # ================================================================
    separator("Step 5: 滤波器参数迭代优化")
    
    optimization_results = {}
    best_filters = {}
    
    for mod_type in signals:
        data = signals[mod_type]
        low, high = bandpass_params[mod_type]
        
        print(f"\n  [{mod_type}] 开始参数扫描...")
        results, best = filter_design.iterative_optimize(
            data['clean'], data['noisy'], config.FS,
            cutoff_low=low, cutoff_high=high
        )
        
        optimization_results[mod_type] = results
        best_filters[mod_type] = best
        
        if best:
            scale_info = f", 带宽×{best['cutoff_scale']:g}" if best.get('cutoff_scale', 1) != 1 else ""
            print(f"  [{mod_type}] 最优: {best['type']} order={best['order']} "
                  f"{best['window']}{scale_info}, "
                  f"SNR改善={best['snr_improvement']:.2f}dB, "
                  f"保真度={best['correlation']:.4f}")
            
            # 使用最优滤波器的结果更新
            filtered_results[mod_type][f'{best["type"]}迭代最优'] = (
                best['filtered'], best['time']
            )
            
            # 最优滤波器幅频特性 + Z域零极点分析
            filter_design.plot_best_filter_analysis(
                best['b'], best['a'], best['cutoff'], mod_type=mod_type,
                best_info=best,
                save_path=os.path.join(config.PLOTS_DIR, f'09b_{mod_type}_best_filter.png')
            )
        
        # 绘制优化结果
        filter_design.plot_optimization_results(
            results, mod_type=mod_type,
            save_path=os.path.join(config.PLOTS_DIR, f'09_{mod_type}_optimization.png')
        )
    
    print("  ✓ 迭代优化完成")
    
    # ================================================================
    # Step 6: NLMS 归一化自适应滤波（创新拓展）
    # ================================================================
    separator("Step 6: NLMS 归一化自适应滤波（创新拓展）")
    
    all_lms_results = {}
    
    for mod_type in signals:
        data = signals[mod_type]
        
        print(f"\n  [{mod_type}] NLMS 自适应滤波分析...")
        
        # 评估不同归一化步长 (步长与输入功率无关，收敛快于标准 LMS)
        lms_results = filter_design.evaluate_lms_mu(
            data['clean'], data['noisy'], data['noise'],
            mu_range=config.NLMS_MU_RANGE,
            order=config.NLMS_ORDER,
            algorithm='nlms'
        )
        all_lms_results[mod_type] = lms_results
        
        # 绘制 NLMS 收敛分析
        filter_design.plot_lms_convergence(
            lms_results, mod_type=mod_type,
            save_path=os.path.join(config.PLOTS_DIR, f'10_{mod_type}_lms_convergence.png')
        )
        
        # 找最优步长
        best_lms = max(lms_results, key=lambda x: x['snr_out'])
        print(f"  [{mod_type}] 最优步长: μ={best_lms['mu']:.3f}, "
              f"SNR={best_lms['snr_out']:.2f}dB "
              f"(改善{best_lms['snr_improvement']:+.2f}dB), "
              f"相关系数={best_lms['correlation']:.4f}, "
              f"收敛样本数={best_lms['convergence_samples']}")
        
        # 使用最优步长进行去噪 (计时并复用结果，避免重复计算)
        t0 = time.time()
        denoised, mse_hist, weights, wh = filter_design.lms_denoise(
            data['noisy'], data['noise'],
            mu=best_lms['mu'], order=config.NLMS_ORDER, algorithm='nlms'
        )
        t_lms = time.time() - t0
        
        # 绘制 NLMS 波形对比
        filter_design.plot_lms_detailed_waveform(
            data['clean'], data['noisy'], denoised, data['t'],
            mod_type=mod_type, mu=best_lms['mu'],
            save_path=os.path.join(config.PLOTS_DIR, f'10b_{mod_type}_lms_waveform.png')
        )
        
        filtered_results[mod_type]['NLMS自适应'] = (denoised, t_lms)
    
    print("  ✓ NLMS 自适应滤波分析完成")
    
    # ================================================================
    # Step 7: 信号特征识别
    # ================================================================
    separator("Step 7: ASK/FSK/PSK 自动识别")
    
    recognition_results = {}
    
    for mod_type in signals:
        data = signals[mod_type]
        
        # 使用最优滤波器的输出进行识别
        if best_filters[mod_type]:
            signal_for_recognition = best_filters[mod_type]['filtered']
        else:
            signal_for_recognition = data['noisy']
        
        result = recognizer.classify_signal(signal_for_recognition, config.FS)
        recognition_results[mod_type] = result
        
        is_correct = result['predicted'] == mod_type
        status = "✓ 正确" if is_correct else "✗ 错误"
        
        print(f"  [{mod_type}] → 识别为: {result['predicted']} "
              f"(置信度: {result['confidence']:.1%}) {status}")
        print(f"    依据: {', '.join(result['reasons'])}")
    
    # 绘制识别结果
    recognizer.plot_recognition_results(
        recognition_results,
        save_path=os.path.join(config.PLOTS_DIR, '11_recognition_result.png')
    )
    
    print("  ✓ 信号识别完成")
    
    # ================================================================
    # Step 8: 性能评估（含误码率 BER）
    # ================================================================
    separator("Step 8: 综合性能评估（SNR / 保真度 / 误码率 BER）")
    
    # 准备数据
    clean_signals = {mod: signals[mod]['clean'] for mod in signals}
    noisy_signals = {mod: signals[mod]['noisy'] for mod in signals}
    
    # 未滤波基线（对照）
    for mod_type in signals:
        filtered_results[mod_type]['未滤波'] = (signals[mod_type]['noisy'], 0.0)
    
    # 评估所有方法 (含解调级 BER)
    all_metrics = recognizer.evaluate_all_methods(
        clean_signals, noisy_signals, filtered_results, 
        {mod: signals[mod]['t'] for mod in signals},
        true_bits={mod: signals[mod]['bits'] for mod in signals}
    )
    
    # 运算复杂度: 每输出样本的乘加次数 (零相位滤波 filtfilt 双向 ×2)
    for mod_type in all_metrics:
        for method in all_metrics[mod_type]:
            if method == '未滤波':
                ops = 0.0
            elif method == 'NLMS自适应':
                ops = 4.0 * config.NLMS_ORDER
            elif 'FIR' in method:
                if method == 'FIR迭代最优':
                    b_len = len(best_filters[mod_type]['b']) if best_filters[mod_type] else 0
                    ops = 2.0 * b_len
                else:
                    ops = 2.0 * len(fir_filters[mod_type][0])
            else:
                b_iir, a_iir = iir_filters[mod_type]
                ops = 2.0 * (len(b_iir) + len(a_iir) - 1)
            all_metrics[mod_type][method]['ops_per_sample'] = ops
    
    for mod_type in all_metrics:
        print(f"\n  [{mod_type}]")
        for method, m in all_metrics[mod_type].items():
            ber_str = f", BER={m['ber']:.4f}" if m['ber'] is not None else ""
            print(f"    {method:<12} ΔSNR={m['snr_improvement']:+6.2f}dB, "
                  f"保真度={m['fidelity']:.4f}{ber_str}")
    
    # 绘制性能汇总
    recognizer.plot_performance_summary(
        all_metrics,
        save_path=os.path.join(config.PLOTS_DIR, '12_performance_summary.png')
    )
    
    print("\n  ✓ 性能评估完成")
    
    # ================================================================
    # Step 9: SNR 扫描实验
    # ================================================================
    separator("Step 9: SNR 扫描实验（滤波性能 + 调制识别准确率）")
    
    print(f"  扫描范围: {config.SNR_RANGE} dB, "
          f"每点 {config.SWEEP_TRIALS} 次蒙特卡洛试验\n")
    
    sweep_results = recognizer.run_snr_sweep(
        snr_range=config.SNR_RANGE, n_trials=config.SWEEP_TRIALS
    )
    
    recognizer.plot_snr_sweep_results(
        sweep_results,
        save_path=os.path.join(config.PLOTS_DIR, '14_snr_sweep.png')
    )
    
    print("\n  ✓ SNR 扫描实验完成")
    
    # ================================================================
    # Step 10: 单一域 vs 多域联合识别消融实验
    # ================================================================
    separator("Step 10: 单一域 vs 多域联合识别消融实验")
    
    print(f"  对比模式: 仅时域 / 仅频域 / 仅Z域 / 多域联合, "
          f"每点 {config.SWEEP_TRIALS}×3 次试验\n")
    
    ablation_results = recognizer.run_domain_ablation(
        snr_range=config.SNR_RANGE, n_trials=config.SWEEP_TRIALS
    )
    
    recognizer.plot_domain_ablation(
        ablation_results,
        save_path=os.path.join(config.PLOTS_DIR, '16_domain_ablation.png')
    )
    
    print("\n  ✓ 消融实验完成")
    
    # ================================================================
    # Step 11: 报告生成
    # ================================================================
    separator("Step 11: 生成评估报告")
    
    report = recognizer.generate_report(
        recognition_results, all_metrics, all_lms_results,
        sweep_results=sweep_results, ablation_results=ablation_results,
        log_file=config.LOG_FILE
    )
    
    # ================================================================
    # 完成
    # ================================================================
    total_time = time.time() - start_time
    
    separator("完成！")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  图表输出: {config.PLOTS_DIR}")
    print(f"  评估报告: {config.LOG_FILE}")
    
    # 列出所有生成的图表
    plots = sorted(os.listdir(config.PLOTS_DIR))
    print(f"\n  生成图表清单 ({len(plots)} 张):")
    for p in plots:
        print(f"    • {p}")
    
    print(f"\n{'='*70}")
    print("  程序运行结束")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
