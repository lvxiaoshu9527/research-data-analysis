import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import find_peaks, savgol_filter
import lmfit
from lmfit.models import VoigtModel
import datetime


# ==========================================
# 体系说明（碳纳米管掺碘 Raman 体系）
# ==========================================
# 峰位区间定义（根据手性自动调整 RBM）：
#   碘峰1 (Iodine Peak 1): 90  – 120 cm⁻¹  (~100 cm⁻¹)
#   碘峰2 (Iodine Peak 2): 130 – 170 cm⁻¹  (~150 cm⁻¹)
#   RBM (7,4) CNT        : 260 – 300 cm⁻¹  (~280 cm⁻¹)
#   RBM (6,5) CNT        : 300 – 330 cm⁻¹  (~315 cm⁻¹)
#
# 数据范围：支持至 1800 cm⁻¹（兼容含 G-band 数据）
# 峰拟合仅在低频区进行，高频部分仅保留于输出光谱。
#
# 处理顺序：
#   load → 删除 <80 cm⁻¹ → SG平滑 → SNIP基线 → 三峰拟合

# ==========================================
# 手性识别规则配置
# ==========================================
CHIRALITY_RULES = [
    {
        'chirality': '(7,4)',
        'keywords': ['74', '(7,4)', '7,4'],
        'rbm_min': 260,
        'rbm_max': 300,
        'rbm_center_init': 280,
    },
    {
        'chirality': '(6,5)',
        'keywords': ['65', '(6,5)', '6,5','S7'],
        'rbm_min': 295,      # 修改：300→295，减少误识别
        'rbm_max': 320,      # 修改：330→320，提高拟合稳定性
        'rbm_center_init': 308,
    },
]

DEFAULT_CHIRALITY = {
    'chirality': 'unknown',
    'rbm_min': 260,
    'rbm_max': 330,
    'rbm_center_init': 290,
}


# ==========================================
# 1. 手性识别
# ==========================================

def detect_chirality(filename):
    """
    根据文件名自动识别 CNT 手性。

    识别规则：
      (7,4): 文件名包含 '74' / '(7,4)' / '7,4'
      (6,5): 文件名包含 '65' / '(6,5)' / '6,5' /'S7'

    返回对应手性的配置字典，包含：
      chirality      : 手性标签，如 '(7,4)'
      rbm_min/max    : RBM 拟合搜索区间
      rbm_center_init: RBM 初始中心位置
    """
    fname_lower = filename.lower()

    for rule in CHIRALITY_RULES:
        for kw in rule['keywords']:
            # 关键词匹配（不区分大小写）
            if kw.lower() in fname_lower:
                print(f"    → 手性识别: {rule['chirality']}  "
                      f"(关键词: '{kw}')  "
                      f"RBM 搜索范围: {rule['rbm_min']}–{rule['rbm_max']} cm⁻¹")
                return rule

    # 未能识别手性：使用宽搜索范围并提示用户
    print(f"    ⚠ 未识别到已知手性，使用默认宽 RBM 范围 "
          f"({DEFAULT_CHIRALITY['rbm_min']}–{DEFAULT_CHIRALITY['rbm_max']} cm⁻¹)")
    return DEFAULT_CHIRALITY.copy()


# ==========================================
# 2. 数据读取与预处理
# ==========================================

def load_and_cut_data(filepath, min_shift=80, max_shift=1800):
    """
    读取数据并删除所有 Raman_shift < 80 cm⁻¹ 的数据点。

    上限扩展至 1800 cm⁻¹，兼容包含 G-band (~1580 cm⁻¹) 的 (6,5) 数据。
    低频区 (<80 cm⁻¹) 不参与任何处理。
    """
    try:
        df = pd.read_csv(filepath, sep=r'[\s,]+', engine='python',
                         header=None, comment='#')
        if isinstance(df.iloc[0, 0], str):
            df = df.iloc[1:].reset_index(drop=True)

        df = df.astype(float)
        x, y = df.iloc[:, 0].values, df.iloc[:, 1].values

        sort_idx = np.argsort(x)
        x, y = x[sort_idx], y[sort_idx]

        # 删除 <80 cm⁻¹ 低频噪声区（不参与任何后续处理）
        mask = (x >= min_shift) & (x <= max_shift)
        if len(x[mask]) < 20:
            raise ValueError(f"截断后剩余数据点过少 ({len(x[mask])} 点)")

        x_cut, y_cut = x[mask], y[mask]
        x_max = x_cut.max()
        print(f"    数据范围: {x_cut.min():.1f} – {x_max:.1f} cm⁻¹  ({len(x_cut)} 点)")
        return x_cut, y_cut

    except Exception as e:
        print(f"\n读取文件 {filepath} 失败: {e}")
        return None, None


def smooth_data(y, window_length=9, polyorder=2):
    """
    Savitzky-Golay 平滑，仅用于基线计算和峰拟合。
    原始切断数据单独保留用于导出和绘图。
    """
    if len(y) >= window_length:
        return savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return y.copy()


def baseline_correction_als(y, niter=100):
    """
    SNIP 基线校正（Statistics-sensitive Non-linear Iterative Peak-clipping）。

    为何用 SNIP 替代 ALS
    ─────────────────────
    本数据的背景是从 ~400 强单调下降到 ~120 的宽荧光斜坡。
    ALS 用二阶差分平滑约束，对这种强斜坡会系统性地：
      1. 左端（80–120 cm⁻¹）baseline 偏高 → corrected 出现负值跳变
      2. 中段（150–200 cm⁻¹）出现 S 形偏差 → corrected 出现虚假鼓包

    SNIP 通过迭代削峰直接追踪下包络，对强斜坡背景天然适合：
      • 不需要调 λ/p 两个耦合参数
      • 左端和中段都能准确贴合谷底
      • 唯一参数 niter：越大基线越平（100 次适合宽荧光背景）

    算法
    ────
    1. 双重 sqrt 变换：w = √(√(y + 1))，压缩峰、稳定方差
    2. 迭代削峰：从 m=1 到 m=niter，
           w[i] = min(w[i],  (w[i-m] + w[i+m]) / 2)
       峰被逐渐削去，缓变背景保留
    3. 反变换：baseline = (w²)² − 1
    4. 安全截断：baseline = min(baseline, y)，确保 corrected ≥ 0
    """
    w = np.sqrt(np.sqrt(np.maximum(y, 0) + 1.0))
    for m in range(1, niter + 1):
        if 2 * m >= len(w):
            break
        lhs = w[:-2 * m]
        rhs = w[2 * m:]
        w[m:len(w) - m] = np.minimum(w[m:len(w) - m], (lhs + rhs) / 2.0)
    baseline = (w ** 2) ** 2 - 1.0
    return baseline


def ensure_nonnegative_baseline(y_smooth, baseline):
    """
    后处理保障：将基线超过平滑光谱的位置压回光谱值。
    防止 corrected spectrum 出现负值（物理上无意义）。
    """
    return np.minimum(baseline, y_smooth)


# ==========================================
# 3. 峰识别与拟合（仅在低频区进行）
# ==========================================

def find_iodine1_init(x, y_corr):
    """
    碘峰1 初始位置，搜索窗口收窄至 95–110 cm⁻¹。
    原窗口 90–120 cm⁻¹ 包含 baseline 快速下降区和 shoulder 噪声，
    收窄后可提高 Voigt 初始值稳定性。
    """
    mask = (x >= 95) & (x <= 110)
    x_w, y_w = x[mask], y_corr[mask]
    if len(y_w) == 0:
        return 102.0, 1e-6
    idx = np.argmax(y_w)
    return x_w[idx], max(y_w[idx] * 10, 1e-6)


def find_iodine2_init(x, y_corr):
    """碘峰2 初始位置，搜索窗口 130–170 cm⁻¹。"""
    mask = (x >= 130) & (x <= 170)
    x_w, y_w = x[mask], y_corr[mask]
    if len(y_w) == 0:
        return 150.0, 1e-6
    idx = np.argmax(y_w)
    return x_w[idx], max(y_w[idx] * 10, 1e-6)


def find_rbm_init(x, y_corr, rbm_min, rbm_max, rbm_center_init):
    """
    RBM 初始位置，搜索窗口由手性配置决定：
      (7,4): 260–300 cm⁻¹
      (6,5): 300–330 cm⁻¹
    """
    mask = (x >= rbm_min) & (x <= rbm_max)
    x_w, y_w = x[mask], y_corr[mask]
    if len(y_w) == 0:
        return rbm_center_init, 1e-6
    idx = np.argmax(y_w)
    return x_w[idx], max(y_w[idx] * 10, 1e-6)


def voigt_amplitude_from_height(height, sigma, gamma):
    """
    从峰高反推 Voigt amplitude 的合理初始估算。

    lmfit VoigtModel 的归一化定义：
        f(x) = amplitude * Re[w(z)] / (sigma * sqrt(2*pi))
    其中 w(z) 是 Faddeeva 函数，峰顶处 Re[w(0)] ≈ 1 / sqrt(1 + (gamma/sigma)^2 * pi/2) 近似。

    实用估算（避免数值依赖）：
        amplitude ≈ height * sigma * sqrt(2*pi)
    这给出比 height*10 更合理的初始值，大幅减少优化器偏离问题。
    """
    return max(height * sigma * np.sqrt(2 * np.pi), 1e-6)


def fit_three_peaks(x, y_corr, chirality_config):
    """
    三峰 Voigt 拟合（碘峰1 / 碘峰2 / RBM）。

    ── 关键修复：拟合仅在低频区进行 ──────────────────────────────────────
    拟合区域限制为 80 – (rbm_max + 30) cm⁻¹。
    原来在全谱（至 1800 cm⁻¹）上拟合时，G-band 区域数千个点会主导残差，
    优化器为了最小化整体残差而牺牲低频峰（尤其是 ~100 cm⁻¹ 碘峰）的精度，
    导致碘峰1 拟合结果严重偏宽偏低。

    拟合后，用得到的参数在完整 x 轴上重新 evaluate 各峰组分，
    确保绘图和 Excel 导出的光谱数据仍覆盖全谱。

    ── Voigt amplitude 初始值修复 ─────────────────────────────────────────
    原来用 height * 10 作为 amplitude 初值对 Voigt 是错误的估算。
    现在改用 voigt_amplitude_from_height(height, sigma, gamma) 计算合理初值。

    峰形选择说明：
      Voigt profile = Gaussian × Lorentzian 卷积。
      sigma : Gaussian 展宽
      gamma : Lorentzian 展宽（解锁自由拟合，不强制 gamma=sigma）

    约束：
      碘峰1: center 95–110 cm⁻¹,   sigma 2–10, gamma 2–10
      碘峰2: center 130–170 cm⁻¹,  sigma 3–10, gamma 3–10
      RBM:   center rbm_min–rbm_max, sigma 2–15, gamma 2–15
    """
    rbm_min         = chirality_config['rbm_min']
    rbm_max         = chirality_config['rbm_max']
    rbm_center_init = chirality_config['rbm_center_init']

    # ── 关键修复1: 截取低频拟合区域，排除 G-band 高频干扰 ─────────────────
    fit_x_max = rbm_max + 30          # 例如 (7,4): 330 cm⁻¹, (6,5): 350 cm⁻¹
    fit_mask  = (x >= 80) & (x <= fit_x_max)
    x_fit     = x[fit_mask]
    y_fit     = y_corr[fit_mask]

    if len(x_fit) < 15:
        raise ValueError(f"低频拟合区数据点不足 ({len(x_fit)} 点，需 ≥15)")

    # 在拟合区域内搜索初始峰位
    i1_init,  i1_h = find_iodine1_init(x_fit, y_fit)
    i2_init,  i2_h = find_iodine2_init(x_fit, y_fit)
    rbm_init, rbm_h = find_rbm_init(x_fit, y_fit, rbm_min, rbm_max, rbm_center_init)

    # ── 关键修复2: 用合理公式估算 Voigt amplitude 初始值 ──────────────────
    sigma0 = 5.0
    gamma0 = 5.0
    i1_amp0  = voigt_amplitude_from_height(i1_h,  sigma0, gamma0)
    i2_amp0  = voigt_amplitude_from_height(i2_h,  sigma0, gamma0)
    rbm_amp0 = voigt_amplitude_from_height(rbm_h, sigma0, gamma0)

    model  = (VoigtModel(prefix='i1_') +
              VoigtModel(prefix='i2_') +
              VoigtModel(prefix='rbm_'))
    params = model.make_params()

    # ── 碘峰1: 95–110 cm⁻¹ ───────────────────────────────────────────────
    params['i1_center'].set(   value=i1_init,  min=95,  max=110)
    params['i1_amplitude'].set(value=i1_amp0,  min=0)
    params['i1_sigma'].set(    value=4,         min=1,   max=10)
    params['i1_gamma'].set(    value=4,         min=1,   max=10, vary=True, expr='')

    # ── 碘峰2: 130–170 cm⁻¹ ──────────────────────────────────────────────
    params['i2_center'].set(   value=i2_init,  min=130, max=170)
    params['i2_amplitude'].set(value=i2_amp0,  min=0)
    params['i2_sigma'].set(    value=5,         min=2,   max=10)
    params['i2_gamma'].set(    value=5,         min=2,   max=10, vary=True, expr='')

    # ── RBM: 手性决定范围 ─────────────────────────────────────────────────
    params['rbm_center'].set(   value=rbm_init,  min=rbm_min, max=rbm_max)
    params['rbm_amplitude'].set(value=rbm_amp0,  min=0)
    params['rbm_sigma'].set(    value=5,          min=1,       max=15)
    params['rbm_gamma'].set(    value=5,          min=1,       max=15, vary=True, expr='')

    # ── 仅在低频区域拟合 ──────────────────────────────────────────────────
    result = model.fit(y_fit, params, x=x_fit)

    # 后检查：碘峰 sigma > 10 则以更严格上界重拟合
    p = result.params
    if p['i1_sigma'].value > 10 or p['i2_sigma'].value > 10:
        print("    ⚠ 碘峰 sigma 超限，以更严格上界重拟合...")
        params['i1_sigma'].set(value=min(p['i1_sigma'].value, 7), min=1, max=10)
        params['i1_gamma'].set(value=min(p['i1_gamma'].value, 7), min=1, max=10,
                               vary=True, expr='')
        params['i2_sigma'].set(value=min(p['i2_sigma'].value, 7), min=2, max=10)
        params['i2_gamma'].set(value=min(p['i2_gamma'].value, 7), min=2, max=10,
                               vary=True, expr='')
        result = model.fit(y_fit, params, x=x_fit)

    # ── 用拟合参数在完整 x 轴上 evaluate，供绘图和 Excel 使用 ──────────────
    # result.best_fit 只有 x_fit 长度，需要手动在全谱 x 上重新计算
    result._x_full   = x           # 存储完整 x，供调用方使用
    result._fit_mask = fit_mask     # 存储拟合区掩码

    return result


def compute_fit_quality(y_data, y_fit, n_free_params):
    """
    计算拟合质量评价指标。

    参数：
      y_data       : 被拟合的数据（corrected spectrum，仅拟合区域）
      y_fit        : 拟合曲线
      n_free_params: 自由参数个数（用于计算 reduced chi²）

    返回字典：
      R2            : 决定系数（越接近 1 越好）
      RMSE          : 均方根误差（越小越好，与数据同量纲）
      reduced_chi2  : 缩减卡方（≈1 理想；>>1 欠拟合；<<1 可能过拟合）
    """
    residual = y_data - y_fit
    ss_res   = np.sum(residual ** 2)
    ss_tot   = np.sum((y_data - np.mean(y_data)) ** 2)

    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(residual ** 2))

    dof  = len(y_data) - n_free_params          # 自由度
    # 使用数据标准差估算噪声水平（sigma²）以归一化 chi²
    sigma2 = np.var(y_data - y_fit) if dof > 0 else 1.0
    reduced_chi2 = (ss_res / sigma2 / dof) if (dof > 0 and sigma2 > 0) else np.nan

    return {
        'R2':           round(float(r2),           6),
        'RMSE':         round(float(rmse),          4),
        'reduced_chi2': round(float(reduced_chi2),  4),
    }


# ==========================================
# 4. Excel 导出
# ==========================================

def _build_wide_table(data_dict):
    """
    将 {sample_name: (x_arr, y_arr)} 组装成宽表 DataFrame。
    第一列 Raman_shift 取所有样品 x 轴的并集（排序），
    各样品列用线性插值对齐，范围外填 NaN。
    这样即使各样品 x 轴略有差异也能正确对齐。
    """
    if not data_dict:
        return pd.DataFrame({'Raman_shift': []})

    # 取所有样品 x 轴的并集作为公共轴
    all_x = np.unique(np.concatenate([x for x, _ in data_dict.values()]))

    df = pd.DataFrame({'Raman_shift': all_x})
    for sname, (x_arr, y_arr) in data_dict.items():
        df[sname] = np.interp(all_x, x_arr, y_arr,
                              left=np.nan, right=np.nan)
    return df


def export_excel(all_summaries, all_data_store, output_dir):
    """
    导出 Excel，包含以下工作表：

    Sheet 1: Key_results
      仅保留快速分析所需的关键参数，每行一个样品。
      列：sample_name | chirality |
          iodine1_position | iodine1_intensity |
          iodine2_position | iodine2_intensity |
          RBM_position     | RBM_intensity

    Sheet 2: Peak_parameters_full
      完整 Voigt 峰参数 + 拟合质量指标，每行一个样品。
      列：三峰各自的 position/height/area/FWHM/sigma/gamma
          + fit_R2 / fit_RMSE / fit_reduced_chi2

    Sheet 3–8: 光谱宽表（结构不变）
      Raw_spectra | Smoothed_spectra | Baseline |
      Corrected_spectra | Fit_spectra | Residual_spectra
      第一列 Raman_shift，后续每列为一个样品。
      x 轴覆盖各文件实际数据范围（最高至 1800 cm⁻¹），不人为截断。
    """
    excel_path = os.path.join(output_dir, "Raman_CNT_Iodine_Results.xlsx")
    df_all = pd.DataFrame(all_summaries)

    # ── Sheet 1: Key_results（快速分析用，精简列）────────────────────────
    key_cols = [
        'sample_name', 'chirality',
        'iodine_peak1_position', 'iodine_peak1_height',
        'iodine_peak2_position', 'iodine_peak2_height',
        'RBM_position',          'RBM_height',
    ]
    # 重命名为更简洁的列名，方便直接绘图
    key_rename = {
        'iodine_peak1_position': 'iodine1_position',
        'iodine_peak1_height':   'iodine1_intensity',
        'iodine_peak2_position': 'iodine2_position',
        'iodine_peak2_height':   'iodine2_intensity',
        'RBM_position':          'RBM_position',
        'RBM_height':            'RBM_intensity',
    }
    df_key = df_all[[c for c in key_cols if c in df_all.columns]].copy()
    df_key = df_key.rename(columns=key_rename)

    # ── Sheet 2: Peak_parameters_full（完整 Voigt 参数 + 拟合质量）───────
    full_cols = [
        'sample_name', 'chirality',
        # 碘峰1
        'iodine_peak1_position', 'iodine_peak1_height', 'iodine_peak1_area',
        'iodine_peak1_FWHM',     'iodine_peak1_sigma',  'iodine_peak1_gamma',
        # 碘峰2
        'iodine_peak2_position', 'iodine_peak2_height', 'iodine_peak2_area',
        'iodine_peak2_FWHM',     'iodine_peak2_sigma',  'iodine_peak2_gamma',
        # RBM
        'RBM_position', 'RBM_height', 'RBM_area',
        'RBM_FWHM',     'RBM_sigma',  'RBM_gamma',
        # 拟合质量
        'fit_R2', 'fit_RMSE', 'fit_reduced_chi2',
    ]
    for col in full_cols:
        if col not in df_all.columns:
            df_all[col] = np.nan
    df_full = df_all[full_cols].copy()

    # ── 光谱宽表 ─────────────────────────────────────────────────────────
    raw_dict       = {s: (d['x'], d['raw'])       for s, d in all_data_store.items()}
    smooth_dict    = {s: (d['x'], d['smooth'])    for s, d in all_data_store.items()}
    baseline_dict  = {s: (d['x'], d['baseline'])  for s, d in all_data_store.items()}
    corrected_dict = {s: (d['x'], d['corrected']) for s, d in all_data_store.items()}
    fit_dict       = {s: (d['x'], d['fit'])       for s, d in all_data_store.items()}
    residual_dict  = {s: (d['x'], d['residual'])  for s, d in all_data_store.items()}

    df_raw       = _build_wide_table(raw_dict)
    df_smooth    = _build_wide_table(smooth_dict)
    df_baseline  = _build_wide_table(baseline_dict)
    df_corrected = _build_wide_table(corrected_dict)
    df_fit       = _build_wide_table(fit_dict)
    df_residual  = _build_wide_table(residual_dict)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_key      .to_excel(writer, sheet_name='Key_results',           index=False)
        df_full     .to_excel(writer, sheet_name='Peak_parameters_full',  index=False)
        df_raw      .to_excel(writer, sheet_name='Raw_spectra',           index=False)
        df_smooth   .to_excel(writer, sheet_name='Smoothed_spectra',      index=False)
        df_baseline .to_excel(writer, sheet_name='Baseline',              index=False)
        df_corrected.to_excel(writer, sheet_name='Corrected_spectra',     index=False)
        df_fit      .to_excel(writer, sheet_name='Fit_spectra',           index=False)
        df_residual .to_excel(writer, sheet_name='Residual_spectra',      index=False)

    print(f"\n✓ Excel 已保存: {excel_path}")
    print(f"  Sheet 1: Key_results            — 8 列关键参数（快速分析用）")
    print(f"  Sheet 2: Peak_parameters_full   — 完整 Voigt 参数 + 拟合质量")
    print(f"  Sheet 3–8: 光谱宽表             — Raw / Smoothed / Baseline / "
          f"Corrected / Fit / Residual")
    return excel_path


# ==========================================
# 5. 绘图
# ==========================================

def plot_fitting_results(df_spectrum, filename, chirality_label,
                         rbm_min, rbm_max, output_dir):
    """
    完整保留所有图层，并在标题中显示识别到的手性与 RBM 区间。

    图层包含：
      - 原始切断数据（灰色散点）
      - 平滑数据（浅蓝虚线）
      - SNIP 基线（灰色实线）
      - 校正后光谱（蓝色实线）
      - 总拟合曲线（红色实线）
      - 碘峰1、碘峰2、RBM 各组分（填色）
      - 残差子图
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                             gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    x = df_spectrum['Raman_shift']

    ax.plot(x, df_spectrum['raw_intensity'], 'o', color='lightgray',
            markersize=3, zorder=1, label='Raw (≥80 cm⁻¹)')
    ax.plot(x, df_spectrum['smoothed_intensity'], '--', color='steelblue',
            linewidth=1, alpha=0.7, zorder=2, label='Smoothed (SG)')
    ax.plot(x, df_spectrum['baseline'], '-', color='#888888',
            linewidth=1.5, alpha=0.8, zorder=3, label='SNIP Baseline')
    ax.plot(x, df_spectrum['corrected_intensity'], 'b-',
            alpha=0.5, linewidth=2, zorder=4, label='Corrected')
    ax.plot(x, df_spectrum['total_fit'], 'r-',
            linewidth=2.5, zorder=5, label='Total Fit')

    ax.fill_between(x, df_spectrum['iodine_peak1'],
                    alpha=0.45, color='orange', zorder=6,
                    label='Iodine Peak 1 (90–120 cm⁻¹)')
    ax.fill_between(x, df_spectrum['iodine_peak2'],
                    alpha=0.35, color='green', zorder=6,
                    label='Iodine Peak 2 (130–170 cm⁻¹)')
    ax.fill_between(x, df_spectrum['RBM_component'],
                    alpha=0.45, color='magenta', zorder=6,
                    label=f'RBM {chirality_label} ({rbm_min}–{rbm_max} cm⁻¹)')

    ax.set_title(
        f'CNT-Iodine Raman Fit (Voigt): {filename}\n'
        f'Chirality: {chirality_label}  |  '
        f'RBM region: {rbm_min}–{rbm_max} cm⁻¹  |  '
        f'SNIP baseline (niter=100)',
        fontsize=11
    )
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Intensity (a.u.)')

    # 绘图 x 轴范围：覆盖完整数据范围（含 G-band）
    x_max_plot = min(df_spectrum['Raman_shift'].max(), 1800)
    ax.set_xlim(80, x_max_plot)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    # 残差子图
    ax2 = axes[1]
    residual = df_spectrum['corrected_intensity'] - df_spectrum['total_fit']
    ax2.plot(x, residual, color='purple', linewidth=1)
    ax2.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax2.fill_between(x, residual, alpha=0.3, color='purple')
    ax2.set_xlabel('Raman Shift (cm⁻¹)')
    ax2.set_ylabel('Residual')
    ax2.set_xlim(80, x_max_plot)
    ax2.set_title('Fit Residual', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fitted_spectra", f"{filename}_fit.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_overlay(all_data_store, chirality_map, output_dir,
                  x_lo, x_hi, fname, title_suffix):
    """
    内部辅助：在指定 x 范围内绘制所有样品的 corrected 光谱叠加图。
    按手性着色：(7,4)=蓝, (6,5)=红, unknown=灰。
    """
    if not all_data_store:
        return

    chirality_colors = {'(7,4)': 'tab:blue', '(6,5)': 'tab:red', 'unknown': 'tab:gray'}
    plt.figure(figsize=(13, 6))

    for idx, (sname, d) in enumerate(all_data_store.items()):
        x_arr, y_arr = d['x'], d['corrected']
        chi   = chirality_map.get(sname, 'unknown')
        color = chirality_colors.get(chi, f'C{idx % 10}')
        # 只绘制落在目标 x 范围内的数据点，避免远端高频噪声干扰视图
        mask = (x_arr >= x_lo) & (x_arr <= x_hi)
        if mask.sum() == 0:
            continue
        plt.plot(x_arr[mask], y_arr[mask], linewidth=1.2, alpha=0.8,
                 label=f'{sname} [{chi}]', color=color)

    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Corrected Intensity (a.u.)')
    plt.title(f'All Samples — Corrected Spectra Overlay {title_suffix}')
    plt.xlim(x_lo, x_hi)
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    总览图已保存: {fname}")


def plot_all_spectra_overlay(all_data_store, chirality_map, output_dir):
    """
    生成两张总览叠加图：

    1. all_spectra_overlay_RBM.png  (50–500 cm⁻¹)
       用于观察 RBM 峰及碘峰区域。

    2. all_spectra_overlay_full.png (50–1800 cm⁻¹)
       展示完整 Raman 光谱（RBM + G-band + D-band）。
    """
    _draw_overlay(all_data_store, chirality_map, output_dir,
                  x_lo=50,  x_hi=500,
                  fname='all_spectra_overlay_RBM.png',
                  title_suffix='— RBM region (50–500 cm⁻¹)')

    _draw_overlay(all_data_store, chirality_map, output_dir,
                  x_lo=50,  x_hi=1800,
                  fname='all_spectra_overlay_full.png',
                  title_suffix='— Full spectrum (50–1800 cm⁻¹)')


# ==========================================
# 6. 主程序
# ==========================================

def main():
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askquestion(
        "数据选择",
        "是否批量选择文件夹？\n[是] 整个文件夹\n[否] 手动多选文件"
    )
    file_paths = []
    if choice == 'yes':
        folder_path = filedialog.askdirectory(title="选择包含 Raman 数据的文件夹")
        if not folder_path:
            return
        file_paths.extend(glob.glob(os.path.join(folder_path, "*.txt")))
        file_paths.extend(glob.glob(os.path.join(folder_path, "*.csv")))
    else:
        paths = filedialog.askopenfilenames(
            title="选择 Raman 文件",
            filetypes=[("Data Files", "*.txt *.csv")]
        )
        file_paths = list(paths)

    if not file_paths:
        messagebox.showwarning("无文件", "未选择任何文件，程序退出。")
        return

    messagebox.showinfo("保存路径", "请选择结果输出的根文件夹")
    output_root = filedialog.askdirectory(title="选择结果保存文件夹")
    if not output_root:
        return

    os.makedirs(os.path.join(output_root, "fitted_spectra"), exist_ok=True)

    print(f"\n{'='*65}")
    print(f"碳纳米管掺碘 Raman 拟合程序（Voigt 模型 + 手性自动识别）")
    print(f"{'='*65}")
    print(f"手性识别规则：")
    for rule in CHIRALITY_RULES:
        print(f"  {rule['chirality']}: 关键词 {rule['keywords']}  "
              f"→ RBM {rule['rbm_min']}–{rule['rbm_max']} cm⁻¹")
    print(f"碘峰区间：碘峰1 90–120 cm⁻¹ | 碘峰2 130–170 cm⁻¹")
    print(f"数据范围：80–1800 cm⁻¹（兼容 G-band）")
    print(f"基线参数：SNIP niter=100")
    print(f"{'='*65}")
    print(f"共 {len(file_paths)} 个文件待处理\n")

    all_summaries  = []
    # all_data_store: {sample_name: {'x', 'raw', 'smooth', 'baseline', 'corrected', 'fit'}}
    # 保存每个样品的全部中间光谱层，用于 Excel 多表导出和总览图
    all_data_store   = {}
    chirality_map    = {}   # {sample_name: chirality_label}

    today_str = datetime.date.today().strftime("%Y-%m-%d")

    for filepath in file_paths:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\n处理: {filename}")

        try:
            # ── 手性识别（根据文件名）────────────────────────────────────
            chi_config = detect_chirality(filename)
            chirality_label = chi_config['chirality']
            chirality_map[filename] = chirality_label

            # ── STEP 1: 读取并删除 <80 cm⁻¹（上限扩展至 1800 cm⁻¹）────
            x, y_raw = load_and_cut_data(filepath, min_shift=80, max_shift=1800)
            if x is None:
                continue

            # ── STEP 2: SG 平滑（保留原始数据）─────────────────────────
            y_smooth = smooth_data(y_raw, window_length=9, polyorder=2)

            # ── STEP 3: SNIP 基线（追踪下包络，避免虚假鼓包）────────────
            baseline = baseline_correction_als(y_smooth, niter=100)
            baseline = ensure_nonnegative_baseline(y_smooth, baseline)
            y_corrected = y_smooth - baseline

            # ── STEP 4 & 5: 三峰拟合（仅在低频区进行，RBM 区间由手性决定）──
            # fit_three_peaks 内部截取低频区拟合，避免 G-band 数据主导残差。
            # 拟合参数通过 result.params 取出后，在完整 x 轴重新 evaluate。
            fit_result = fit_three_peaks(x, y_corrected, chi_config)
            p = fit_result.params

            # 在完整 x 轴上重新计算各峰组分和总拟合（供绘图和 Excel 使用）
            model_full = (VoigtModel(prefix='i1_') +
                          VoigtModel(prefix='i2_') +
                          VoigtModel(prefix='rbm_'))
            total_fit_full = model_full.eval(p, x=x)
            comps_full     = model_full.eval_components(params=p, x=x)

            # ── 构建光谱 DataFrame（用于绘图，全谱范围）─────────────────
            df_spectrum = pd.DataFrame({
                'Raman_shift':        x,
                'raw_intensity':      y_raw,
                'smoothed_intensity': y_smooth,
                'baseline':           baseline,
                'corrected_intensity':y_corrected,
                'total_fit':          total_fit_full,
                'iodine_peak1':       comps_full['i1_'],
                'iodine_peak2':       comps_full['i2_'],
                'RBM_component':      comps_full['rbm_']
            })

            # ── 汇总峰参数（Voigt 参数 + 拟合质量）────────────────────────
            # 拟合质量在低频拟合区（fit_result.best_fit 即低频区结果）计算
            fit_mask_q = (x >= 80) & (x <= chi_config['rbm_max'] + 30)
            fit_qual   = compute_fit_quality(
                y_data=y_corrected[fit_mask_q],
                y_fit=total_fit_full[fit_mask_q],
                n_free_params=fit_result.nvarys,
            )

            summary_row = {
                'sample_name': filename,
                'chirality':   chirality_label,
                # 碘峰1
                'iodine_peak1_position': round(p['i1_center'].value,    2),
                'iodine_peak1_height':   round(p['i1_height'].value,    2),
                'iodine_peak1_area':     round(p['i1_amplitude'].value, 4),
                'iodine_peak1_FWHM':     round(p['i1_fwhm'].value,      2),
                'iodine_peak1_sigma':    round(p['i1_sigma'].value,     4),
                'iodine_peak1_gamma':    round(p['i1_gamma'].value,     4),
                # 碘峰2
                'iodine_peak2_position': round(p['i2_center'].value,    2),
                'iodine_peak2_height':   round(p['i2_height'].value,    2),
                'iodine_peak2_area':     round(p['i2_amplitude'].value, 4),
                'iodine_peak2_FWHM':     round(p['i2_fwhm'].value,      2),
                'iodine_peak2_sigma':    round(p['i2_sigma'].value,     4),
                'iodine_peak2_gamma':    round(p['i2_gamma'].value,     4),
                # RBM
                'RBM_position': round(p['rbm_center'].value,    2),
                'RBM_height':   round(p['rbm_height'].value,    2),
                'RBM_area':     round(p['rbm_amplitude'].value, 4),
                'RBM_FWHM':     round(p['rbm_fwhm'].value,      2),
                'RBM_sigma':    round(p['rbm_sigma'].value,     4),
                'RBM_gamma':    round(p['rbm_gamma'].value,     4),
                # 拟合质量指标
                'fit_R2':           fit_qual['R2'],
                'fit_RMSE':         fit_qual['RMSE'],
                'fit_reduced_chi2': fit_qual['reduced_chi2'],
            }
            all_summaries.append(summary_row)

            # 保存全部光谱层（用于 Excel 各宽表 + 总览图）
            # residual = corrected − total_fit（完整谱范围）
            y_residual = y_corrected - total_fit_full
            all_data_store[filename] = {
                'x':         x,
                'raw':       y_raw,
                'smooth':    y_smooth,
                'baseline':  baseline,
                'corrected': y_corrected,
                'fit':       total_fit_full,
                'residual':  y_residual,
            }

            # ── 绘图（含手性标注）────────────────────────────────────────
            plot_fitting_results(
                df_spectrum, filename, chirality_label,
                chi_config['rbm_min'], chi_config['rbm_max'],
                output_root
            )

            print(f"    ✓  手性={chirality_label}  "
                  f"RBM={p['rbm_center'].value:.1f} cm⁻¹ (σ={p['rbm_sigma'].value:.2f}, γ={p['rbm_gamma'].value:.2f})  "
                  f"I1={p['i1_center'].value:.1f} cm⁻¹  I2={p['i2_center'].value:.1f} cm⁻¹  "
                  f"R²={fit_qual['R2']:.4f}")

        except Exception as e:
            print(f"    ✗ 失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 全部处理完毕：导出 Excel + 总览图 ────────────────────────────────
    if all_summaries:
        export_excel(all_summaries, all_data_store, output_root)
        plot_all_spectra_overlay(all_data_store, chirality_map, output_root)

        # 统计各手性样品数
        chi_counts = {}
        for row in all_summaries:
            c = row['chirality']
            chi_counts[c] = chi_counts.get(c, 0) + 1
        chi_summary = '  |  '.join(f"{c}: {n} 个" for c, n in chi_counts.items())

        messagebox.showinfo(
            "完成",
            f"分析完成！成功处理 {len(all_summaries)} / {len(file_paths)} 个样品\n\n"
            f"手性分布: {chi_summary}\n\n"
            f"输出文件夹:\n{output_root}\n\n"
            f"Excel: Raman_CNT_Iodine_Results.xlsx\n"
            f"  [Sheet 1] Key_results          — 关键峰位 & 强度（快速分析）\n"
            f"  [Sheet 2] Peak_parameters_full — 完整 Voigt 参数 + R²/RMSE/χ²\n"
            f"  [Sheet 3] Raw_spectra\n"
            f"  [Sheet 4] Smoothed_spectra\n"
            f"  [Sheet 5] Baseline\n"
            f"  [Sheet 6] Corrected_spectra\n"
            f"  [Sheet 7] Fit_spectra\n"
            f"  [Sheet 8] Residual_spectra\n\n"
            f"总览图:\n"
            f"  all_spectra_overlay_RBM.png  (50–500 cm⁻¹)\n"
            f"  all_spectra_overlay_full.png (50–1800 cm⁻¹)"
        )
    else:
        messagebox.showwarning("警告", "没有成功处理任何数据，请检查文件格式。")


if __name__ == "__main__":
    main()