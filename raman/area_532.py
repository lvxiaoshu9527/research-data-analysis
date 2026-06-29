import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import lmfit
from lmfit.models import PseudoVoigtModel
import warnings

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════╗
# ║          532nm 专属配置表 —— 融合 785nm 单峰逻辑与严格物理锁           ║
# ╠══════════════════════════════════════════════════════════════════╣
CHIRALITY_DB = {
    "65": {
        "label": "(6,5)",
        "rbm_center": 312,
        "rbm_range": [300, 325],
        "rbm_sigma": [1.0, 4.5],  # 【物理锁】防鼓包：严禁峰变宽
    },
    "91": {
        "label": "(9,1)",
        "rbm_center": 312,
        "rbm_range": [300, 325],
        "rbm_sigma": [1.0, 4.5],  # 【物理锁】防鼓包：严禁峰变宽
    },
    "74": {
        "label": "(7,4)",
        "rbm_center": 278,
        "rbm_range": [260, 290],
        "rbm_sigma": [1.0, 4.5],  # 【物理锁】防鼓包：严禁峰变宽
    },
    "66": {  # <--- 【新增】(6,6) 手性配置，参数与 74 基本一致
        "label": "(6,6)",
        "rbm_center": 276,        # 如果 6,6 实际峰位有微小差异，可以修改这里的中心值
        "rbm_range": [260, 290],
        "rbm_sigma": [1.0, 4.5],  # 【物理锁】防鼓包：严禁峰变宽
    },
    "S7": {
        "label": "(6,5) Dirty",
        "rbm_center": 311,
        "rbm_range": [305, 316],  # S7 的 RBM 区间收窄，避开右侧强杂质
        "rbm_sigma": [1.0, 3.5],  # 夹缝中求生，极细的展宽限制
    }
}

# 碘峰约束（大道至简单峰模型，785nm纯净逻辑）
IODINE_CONFIG = {
    "i1": {"center": 104, "range": [90, 125], "sigma": [2.0, 8.0]},
    "i2": {"center": 140, "range": [125, 170], "sigma": [3.0, 10.0]},
}


def safe_div(num, den):
    return round(float(num) / float(den), 4) if abs(den) > 1e-3 else np.nan


def peak_area_numerical(x, y):
    return float(trapezoid(np.maximum(y, 0), x))


def parse_chirality(filename):
    fname_lower = filename.lower()
    if 's7' in fname_lower: return 'S7'
    if '65' in fname_lower or '6,5' in fname_lower: return '65'
    if '74' in fname_lower or '7,4' in fname_lower: return '74'
    if '66' in fname_lower or '6,6' in fname_lower: return '66' # <--- 【新增】识别 66 或 6,6
    if '91' in fname_lower or '9,1' in fname_lower: return '91'
    return None


def load_and_cut_data(filepath, x_min=70, x_max=500):
    """截断 80 cm⁻¹ 以下的无效滤光片截断区"""
    try:
        df = pd.read_csv(filepath, sep=r'[\s,]+', engine='python', header=None, comment='#')
        if isinstance(df.iloc[0, 0], str): df = df.iloc[1:].reset_index(drop=True)
        df = df.astype(float)
        x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        mask = (x >= x_min) & (x <= x_max)
        return x[mask], y[mask]
    except:
        return None, None


def baseline_correction_snip(y, niter=100):
    """【保留旧版精髓】SNIP 基线校正：完美追踪 532nm 下包络谷底"""
    w = np.sqrt(np.sqrt(np.maximum(y, 0) + 1.0))
    for m in range(1, niter + 1):
        if 2 * m >= len(w): break
        lhs, rhs = w[:-2 * m], w[2 * m:]
        w[m:len(w) - m] = np.minimum(w[m:len(w) - m], (lhs + rhs) / 2.0)
    baseline = (w ** 2) ** 2 - 1.0
    return baseline


def estimate_local_snr(x, y_corr, center, half_width):
    """基于扣除 SNIP 基线后的信号进行 SNR 评估"""
    peak_mask = (x >= center - half_width) & (x <= center + half_width)
    noise_mask = (x >= center - half_width * 3) & (x <= center + half_width * 3) & ~peak_mask
    if peak_mask.sum() < 3: return 0.0
    peak_h = max(y_corr[peak_mask].max(), 0.0)
    noise_std = y_corr[noise_mask].std() if noise_mask.sum() >= 3 else 1.0
    return peak_h / max(noise_std, 1.0)


def initialize_parameters(x, y_corr, chirality_key):
    """【自适应寻峰】完全由代码根据数据真实高度猜测初始值"""

    def local_init(center, half_w):
        mask = (x >= center - half_w) & (x <= center + half_w)
        if mask.sum() > 3:
            y_local = y_corr[mask]
            ci = np.argmax(y_local)
            # 根据真实高度和粗略半宽自动计算 Amplitude 的完美初始值
            init_amp = max(y_local[ci], 1.0) * half_w * 1.5
            return x[mask][ci], init_amp
        return center, 100

    cfg = CHIRALITY_DB[chirality_key]
    rbm_c, rbm_a = local_init(cfg["rbm_center"], 10)
    i1_c, i1_a = local_init(IODINE_CONFIG["i1"]["center"], 15)
    i2_c, i2_a = local_init(IODINE_CONFIG["i2"]["center"], 15)

    is_dirty = (chirality_key == 'S7')
    m1_target = 278 if is_dirty else 245
    m2_target = 324 if is_dirty else 280

    m1_c, m1_a = local_init(m1_target, 12)
    m2_c, m2_a = local_init(m2_target, 12)

    snr_rbm = estimate_local_snr(x, y_corr, cfg["rbm_center"], 10)
    snr_m1 = estimate_local_snr(x, y_corr, m1_target, 12)
    snr_m2 = estimate_local_snr(x, y_corr, m2_target, 12)

    # 动态 M 带触发：SNR >= 2.0 自动开启，S7强制开启排雷
    flags = {
        "rbm": snr_rbm >= 1.2 or is_dirty,
        "i1": True,
        "i2": True,
        "m1": snr_m1 >= 2.0 or is_dirty,
        "m2": snr_m2 >= 2.0 or is_dirty
    }

    inits = dict(rbm_c=rbm_c, rbm_a=rbm_a, i1_c=i1_c, i1_a=i1_a, i2_c=i2_c, i2_a=i2_a,
                 m1_c=m1_c, m1_a=m1_a, m2_c=m2_c, m2_a=m2_a)
    return inits, flags, {"SNR_RBM": round(snr_rbm, 2)}


def build_model(inits, flags, chirality_key):
    """构建纯粹的 PseudoVoigt 峰组（因为 SNIP 已经把基线处理掉了）"""
    cfg = CHIRALITY_DB[chirality_key]
    is_dirty = (chirality_key == 'S7')

    model = PseudoVoigtModel(prefix='i1_') + PseudoVoigtModel(prefix='i2_')

    add_m1 = flags["m1"] and (is_dirty or abs(cfg["rbm_center"] - inits['m1_c']) > 15)
    add_m2 = flags["m2"] and (is_dirty or abs(cfg["rbm_center"] - inits['m2_c']) > 15)

    if add_m1: model = model + PseudoVoigtModel(prefix='m1_')
    if add_m2: model = model + PseudoVoigtModel(prefix='m2_')

    has_rbm = flags["rbm"]
    if has_rbm: model = model + PseudoVoigtModel(prefix='rbm_')

    params = model.make_params()

    if add_m1:
        params['m1_center'].set(value=inits['m1_c'], min=inits['m1_c'] - 15, max=inits['m1_c'] + 15)
        params['m1_sigma'].set(value=4, min=1.0, max=6.0)  # 杂峰上限防鼓包
        params['m1_amplitude'].set(value=inits['m1_a'], min=0)
        params['m1_fraction'].set(value=0.5, min=0, max=1)
    if add_m2:
        params['m2_center'].set(value=inits['m2_c'], min=inits['m2_c'] - 15, max=inits['m2_c'] + 15)
        params['m2_sigma'].set(value=4, min=1.0, max=5.0)  # 杂峰上限防鼓包
        params['m2_amplitude'].set(value=inits['m2_a'], min=0)
        params['m2_fraction'].set(value=0.5, min=0, max=1)

    for p_fix in ['i1', 'i2']:
        c = IODINE_CONFIG[p_fix]
        params[f'{p_fix}_center'].set(value=inits[f'{p_fix}_c'], min=c["range"][0], max=c["range"][1])
        params[f'{p_fix}_sigma'].set(value=c["sigma"][0], min=1.0, max=c["sigma"][1])
        params[f'{p_fix}_amplitude'].set(value=inits[f'{p_fix}_a'], min=0)
        params[f'{p_fix}_fraction'].set(value=0.5, min=0, max=1)

    if has_rbm:
        params['rbm_center'].set(value=inits['rbm_c'], min=cfg["rbm_range"][0], max=cfg["rbm_range"][1])
        params['rbm_sigma'].set(value=cfg["rbm_sigma"][0], min=0.5, max=cfg["rbm_sigma"][1])  # 核心防鼓包锁
        params['rbm_amplitude'].set(value=inits['rbm_a'], min=0)
        params['rbm_fraction'].set(value=0.5, min=0, max=1)

    return model, params, has_rbm, {"add_m1": add_m1, "add_m2": add_m2}


def run_fit(x, y_corr, y_raw, chirality_key):
    inits, flags, snrs = initialize_parameters(x, y_corr, chirality_key)
    model, params, has_rbm, extra_flags = build_model(inits, flags, chirality_key)

    # 权重依然依赖于真实 y_raw 噪声统计
    weights = 1.0 / np.sqrt(np.maximum(y_raw, 1.0))

    # 执行真实自动拟合（对扣除基线后的 y_corr 拟合）
    fit_coarse = model.fit(y_corr, params, x=x, weights=weights, method='leastsq', fit_kws={'maxfev': 800})
    fit_result = model.fit(y_corr, fit_coarse.params, x=x, weights=weights, method='leastsq', fit_kws={'maxfev': 2000})

    fit_result._flags = flags
    fit_result._snrs = snrs
    fit_result._has_rbm = has_rbm
    fit_result._extra = extra_flags
    return fit_result


def extract_results(x, fit_result, filename, chirality_key):
    p = fit_result.params
    comps = fit_result.eval_components(x=x)
    cfg = CHIRALITY_DB[chirality_key]
    has_rbm = fit_result._has_rbm

    areas = {k: peak_area_numerical(x, v) for k, v in comps.items()}
    area_i1 = areas.get('i1_', 0.0)
    area_i2 = areas.get('i2_', 0.0)
    area_rbm = areas.get('rbm_', 0.0)

    ratio_row = {
        "Sample": filename,
        "Chirality": cfg["label"],
        "Reduced_Chi2": round(fit_result.redchi, 4),
        "RBM_center": round(p['rbm_center'].value, 2) if has_rbm and 'rbm_center' in p else np.nan,
        "I1_Area": round(area_i1, 2),
        "I2_Area": round(area_i2, 2),
        "RBM_Area": round(area_rbm, 2),
        "I1/RBM": safe_div(area_i1, area_rbm),
        "I2/RBM": safe_div(area_i2, area_rbm),
        "I1/I2": safe_div(area_i1, area_i2),
    }

    # 【新增功能】：独立杂峰统计报表
    extra_rows = []
    for m_prefix, m_label in [('m1_', 'M1/Metal Band'), ('m2_', 'M2/Impurity Band')]:
        if fit_result._extra.get(f"add_{m_prefix[:2]}") and m_prefix in p:
            extra_rows.append({
                "Sample": filename,
                "Chirality": cfg["label"],
                "Peak_Type": m_label,
                "Center": round(p[f'{m_prefix}center'].value, 2),
                "Height": round(p[f'{m_prefix}height'].value, 2) if f'{m_prefix}height' in p else np.nan,
                "Area": round(areas.get(m_prefix, 0.0), 2),
                "Sigma_Width": round(p[f'{m_prefix}sigma'].value, 2)
            })

    return ratio_row, extra_rows, comps


def plot_result(x, y_raw, y_corr, fit_result, comps, bkg, filename, chirality_key, output_dir):
    cfg = CHIRALITY_DB[chirality_key]
    has_rbm = fit_result._has_rbm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    ax1.plot(x, y_raw, '.', color='dimgray', ms=4, label='Raw Data')

    # 真正的 Total Fit = SNIP 背景 + 各自的拟合组分
    total_fit_absolute = fit_result.best_fit + bkg
    ax1.plot(x, total_fit_absolute, 'r-', lw=2.2, label=f'Total Auto-Fit (χ²ᵣ={fit_result.redchi:.2f})')
    ax1.plot(x, bkg, '--', color='gray', lw=1.5, label='SNIP Baseline')

    def fill(prefix, color, label):
        if prefix in comps:
            ax1.fill_between(x, bkg, bkg + comps[prefix], alpha=0.45, color=color, label=label)

    fill('i1_', 'orange', 'I₁ Peak (~104 cm⁻¹)')
    fill('i2_', 'forestgreen', 'I₂ Peak (~140 cm⁻¹)')
    fill('m1_', 'cyan', 'Extra Band 1')
    fill('m2_', 'dodgerblue', 'Extra Band 2')

    if has_rbm:
        fill('rbm_', 'magenta', f'{cfg["label"]} RBM')

    ax1.set_xlim(80, 420)
    ax1.set_ylabel('Raman Intensity', fontsize=12)
    ax1.set_title(f'532nm: {filename}  [{cfg["label"]}] (SNIP + Auto-Fit)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    norm_res = fit_result.residual / np.sqrt(np.maximum(y_corr, 1))
    ax2.plot(x, norm_res, '.', color='steelblue', ms=3)
    ax2.axhline(0, color='k', ls='--')
    ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Norm. Residual', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitted_spectra", f"{filename}_532nm_Fit.png"), dpi=200)
    plt.close()


def main():
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askquestion("数据选择", "是否批量选择文件夹？\n[是] 整个文件夹\n[否] 手动多选文件")
    paths = []
    if choice == 'yes':
        folder_path = filedialog.askdirectory(title="选择包含 532nm Raman 数据的文件夹")
        if not folder_path: return
        paths.extend(glob.glob(os.path.join(folder_path, "*.txt")))
        paths.extend(glob.glob(os.path.join(folder_path, "*.csv")))
    else:
        selected_files = filedialog.askopenfilenames(title="选择 532nm Raman 文件",
                                                     filetypes=[("Data Files", "*.txt *.csv")])
        paths = list(selected_files)

    if not paths: return

    output_root = filedialog.askdirectory(title="选择保存分析结果的文件夹")
    if not output_root: return
    os.makedirs(os.path.join(output_root, "fitted_spectra"), exist_ok=True)

    all_ratios, all_extra_peaks = [], []
    list_raw, list_fit, list_baseline, list_comps = [], [], [], []

    print("\n🚀 启动 532nm 智能自动化拟合 (SNIP 基线 + 严格防鼓包机制)...\n")

    for filepath in paths:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        chirality_key = parse_chirality(filename)
        if not chirality_key:
            print(f"⚠ 跳过: {filename} (未识别手性)")
            continue

        print(f"▶ 处理: {filename} ", end="", flush=True)
        try:
            x, y_raw = load_and_cut_data(filepath)
            if x is None or len(x) < 20:
                print(" ❌ 失败! (有效数据不足)")
                continue

            # 1. 计算 SNIP 基线
            wl = min(15, len(y_raw) - (1 - len(y_raw) % 2))
            wl = max(wl if wl % 2 == 1 else wl - 1, 5)
            y_smooth = savgol_filter(y_raw, window_length=wl, polyorder=2)
            bkg = baseline_correction_snip(y_smooth, niter=100)
            bkg = np.minimum(bkg, y_smooth)  # 确保基线不超调

            # 2. 对扣除基线后的光谱进行自动拟合
            y_corr = y_raw - bkg
            fit_result = run_fit(x, y_corr, y_raw, chirality_key)

            # 3. 提取特征
            ratio_row, extra_rows, comps = extract_results(x, fit_result, filename, chirality_key)
            all_ratios.append(ratio_row)
            all_extra_peaks.extend(extra_rows)

            list_raw.append(pd.DataFrame({f'{filename}_X': x, f'{filename}_Raw': y_raw}))
            list_fit.append(pd.DataFrame({f'{filename}_X': x, f'{filename}_Fit': fit_result.best_fit + bkg}))
            list_baseline.append(pd.DataFrame({f'{filename}_X': x, f'{filename}_Baseline': bkg}))
            comp_dict = {f'{filename}_X': x}
            for tag, pf in [('I1', 'i1_'), ('I2', 'i2_'), ('RBM', 'rbm_'), ('M1', 'm1_'), ('M2', 'm2_')]:
                comp_dict[f'{filename}_{tag}'] = comps.get(pf, np.zeros_like(x))
            list_comps.append(pd.DataFrame(comp_dict))

            plot_result(x, y_raw, y_corr, fit_result, comps, bkg, filename, chirality_key, output_root)
            print(" ✔️")
        except Exception as e:
            print(f" ❌ 失败! ({e})")
            import traceback
            traceback.print_exc()

    if all_ratios:
        excel_path = os.path.join(output_root, "Raman_532nm_Results_Final.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(all_ratios).to_excel(writer, sheet_name='Core_Ratios', index=False)
            if all_extra_peaks:
                pd.DataFrame(all_extra_peaks).to_excel(writer, sheet_name='Extra_Peaks_Info', index=False)
            else:
                pd.DataFrame([{"Message": "No extra impurity peaks detected"}]).to_excel(writer,
                                                                                         sheet_name='Extra_Peaks_Info',
                                                                                         index=False)
            pd.concat(list_raw, axis=1).to_excel(writer, sheet_name='Raw_Data', index=False)
            pd.concat(list_fit, axis=1).to_excel(writer, sheet_name='Total_Fit', index=False)
            pd.concat(list_baseline, axis=1).to_excel(writer, sheet_name='Baseline', index=False)
            pd.concat(list_comps, axis=1).to_excel(writer, sheet_name='Components', index=False)

        print(f"\n🎉 532nm 全自动拟合完毕！报表和图表已输出至：{output_root}")
        messagebox.showinfo("完成", f"智能提取完毕！\n新增了 Extra_Peaks_Info 杂峰工作表。\n结果请查看：\n{output_root}")


if __name__ == "__main__":
    main()