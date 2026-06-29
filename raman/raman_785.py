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
# ║          手性配置表 —— 极严格的物理边界，防串岗专用              ║
# ╠══════════════════════════════════════════════════════════════════╣
CHIRALITY_DB = {
    "65": {
        "label": "(6,5)",
        "rbm_center": 313,
        "rbm_range": [305, 320],
        "rbm_sigma": [3, 8.0],
    },
    "74": {
        "label": "(7,4)",
        "rbm_center": 270,
        "rbm_range": [263, 274],  # [核心修复] 严格卡死在274以下，绝不准触碰285的杂质带
        "rbm_sigma": [3, 8.0],
    },
    "91": {
        "label": "(9,1)",
        "rbm_center": 310,
        "rbm_range": [303, 318],
        "rbm_sigma": [3, 8.0],
    },
}

# 碘峰约束（单峰模型，允许动态漂移）
IODINE_CONFIG = {
    "i1": {"center": 104, "range": [90, 125], "sigma": [4, 20]},
    "i2": {"center": 145, "range": [135, 155], "sigma": [5, 20]},
}

# 降低 SNR 阈值，挽救弱信号的 RBM 峰
SNR_THRESHOLD = 1.5


def safe_div(num, den):
    return round(float(num) / float(den), 4) if abs(den) > 1e-3 else np.nan


def peak_area_numerical(x, y):
    return float(trapezoid(np.maximum(y, 0), x))


def snip_baseline(y, iter_max):
    """
    SNIP 基线算法 —— 采用“极近端切线外推”解决边缘夹角问题
    """
    # [核心修复] 只取最边缘的 3 个点算斜率，获取最陡峭的切线，避免形成内凹夹角
    left_slope = max((y[0] - y[3]) / 3.0, 0)
    left_pad = y[0] + left_slope * np.arange(iter_max, 0, -1)

    right_slope = max((y[-4] - y[-1]) / 3.0, 0)
    right_pad = y[-1] - right_slope * np.arange(1, iter_max + 1)

    z = np.concatenate([left_pad, y, right_pad])
    for p in range(1, iter_max + 1):
        left = z[:-2 * p]
        right = z[2 * p:]
        avg = (left + right) / 2.0
        z[p:-p] = np.minimum(z[p:-p], avg)

    return z[iter_max:-iter_max]


def estimate_local_snr(x, y_corr, center, half_width):
    peak_mask = (x >= center - half_width) & (x <= center + half_width)
    noise_mask = (x >= center - half_width * 3) & (x <= center + half_width * 3) & ~peak_mask
    if peak_mask.sum() < 3: return 0.0
    peak_h = max(y_corr[peak_mask].max(), 0.0)
    noise_std = y_corr[noise_mask].std() if noise_mask.sum() >= 3 else 1.0
    return peak_h / max(noise_std, 1.0)


def parse_chirality(filename):
    m = re.search(r'I@(\d+)', filename, re.IGNORECASE)
    if m:
        key = m.group(1)
        if key in CHIRALITY_DB: return key
    return None


def load_and_cut_data(filepath, x_min=50, x_max=450):
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


def initialize_parameters(x, y_raw, chirality_key):
    wl = min(21, len(y_raw) - (1 - len(y_raw) % 2))
    wl = max(wl if wl % 2 == 1 else wl - 1, 5)
    y_smooth = savgol_filter(y_raw, window_length=wl, polyorder=2) if len(y_raw) > 9 else y_raw.copy()

    dx = np.median(np.diff(x))
    iter_max = max(int(40 / dx), 10)

    baseline_est = snip_baseline(y_smooth, iter_max)
    y_corr = np.maximum(y_smooth - baseline_est, 0)

    def local_init(center, half_w, default_amp):
        mask = (x >= center - half_w) & (x <= center + half_w)
        if mask.sum() > 3:
            ci = np.argmax(y_corr[mask])
            return x[mask][ci], max(y_corr[mask][ci], 1.0) * half_w * 1.5
        return center, default_amp

    cfg = CHIRALITY_DB[chirality_key]

    # [核心修复] RBM 的寻峰半径缩紧到 5，防止一不小心抓到远处的杂质包
    rbm_c, rbm_a = local_init(cfg["rbm_center"], 5, 0)
    i1_c, i1_a = local_init(IODINE_CONFIG["i1"]["center"], 15, 100)
    i2_c, i2_a = local_init(IODINE_CONFIG["i2"]["center"], 15, 200)

    m1_c, m1_a = local_init(245, 10, 80)
    m2_c, m2_a = local_init(285, 10, 80)

    snr_rbm = estimate_local_snr(x, y_corr, cfg["rbm_center"], 5)
    snr_m1 = estimate_local_snr(x, y_corr, 245, 10)
    snr_m2 = estimate_local_snr(x, y_corr, 285, 10)

    flags = {
        "rbm": snr_rbm >= SNR_THRESHOLD,
        "i1": True,
        "i2": True,
        "m1": snr_m1 >= 1.5,
        "m2": snr_m2 >= 1.5
    }

    inits = dict(rbm_c=rbm_c, rbm_a=rbm_a, i1_c=i1_c, i1_a=i1_a, i2_c=i2_c, i2_a=i2_a,
                 m1_c=m1_c, m1_a=m1_a, m2_c=m2_c, m2_a=m2_a)

    return inits, flags, {"SNR_RBM": round(snr_rbm, 2)}, baseline_est


def build_model(inits, flags, chirality_key):
    cfg = CHIRALITY_DB[chirality_key]

    add_m1 = flags["m1"]
    add_m2 = flags["m2"]

    model = PseudoVoigtModel(prefix='i1_') + PseudoVoigtModel(prefix='i2_')

    if add_m1: model = model + PseudoVoigtModel(prefix='m1_')
    if add_m2: model = model + PseudoVoigtModel(prefix='m2_')

    has_rbm = flags["rbm"]
    if has_rbm: model = model + PseudoVoigtModel(prefix='rbm_')

    params = model.make_params()

    if add_m1:
        params['m1_center'].set(value=max(inits['m1_c'], 220), min=220, max=260)
        params['m1_sigma'].set(value=6, min=2, max=15)
        params['m1_amplitude'].set(value=inits['m1_a'], min=0)
    if add_m2:
        # [核心修复] 动态划定 M2 杂质带的底线，确保它绝对不会侵入 RBM 的领地
        m2_lower_bound = max(275, cfg["rbm_range"][1] + 2)
        params['m2_center'].set(value=max(inits['m2_c'], m2_lower_bound + 1), min=m2_lower_bound, max=305)
        params['m2_sigma'].set(value=6, min=2, max=15)
        params['m2_amplitude'].set(value=inits['m2_a'], min=0)

    for p_fix in ['i1', 'i2']:
        c = IODINE_CONFIG[p_fix]
        params[f'{p_fix}_center'].set(value=inits[f'{p_fix}_c'], min=c["range"][0], max=c["range"][1])
        params[f'{p_fix}_sigma'].set(value=c["sigma"][0], min=2, max=c["sigma"][1])
        params[f'{p_fix}_amplitude'].set(value=inits[f'{p_fix}_a'], min=0)
        params[f'{p_fix}_fraction'].set(value=0.5, min=0, max=1)

    if has_rbm:
        params['rbm_center'].set(value=inits['rbm_c'], min=cfg["rbm_range"][0], max=cfg["rbm_range"][1])
        params['rbm_sigma'].set(value=cfg["rbm_sigma"][0], min=1.5, max=cfg["rbm_sigma"][1])
        params['rbm_amplitude'].set(value=inits['rbm_a'], min=0)
        params['rbm_fraction'].set(value=0.5, min=0, max=1)

    return model, params, has_rbm


def run_fit(x, y_raw, chirality_key):
    inits, flags, snrs, bkg = initialize_parameters(x, y_raw, chirality_key)
    y_corr = np.maximum(y_raw - bkg, 0)

    model, params, has_rbm = build_model(inits, flags, chirality_key)
    weights = 1.0 / np.sqrt(np.maximum(y_raw, 1.0))

    fit_coarse = model.fit(y_corr, params, x=x, weights=weights, method='leastsq', fit_kws={'maxfev': 800})
    fit_result = model.fit(y_corr, fit_coarse.params, x=x, weights=weights, method='leastsq', fit_kws={'maxfev': 2000})

    fit_result._flags = flags
    fit_result._snrs = snrs
    fit_result._has_rbm = has_rbm
    fit_result._bkg = bkg

    return fit_result


def extract_results(x, y_raw, fit_result, filename, chirality_key):
    p = fit_result.params
    comps = fit_result.eval_components(x=x)
    bkg = fit_result._bkg
    cfg = CHIRALITY_DB[chirality_key]
    has_rbm = fit_result._has_rbm

    areas = {k: peak_area_numerical(x, v) for k, v in comps.items()}

    def get_area(prefix):
        return areas.get(prefix, 0.0)

    area_i1 = get_area('i1_')
    area_i2 = get_area('i2_')
    area_rbm = get_area('rbm_')

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

    all_peaks_data = []
    for tag, pf in [('I1', 'i1_'), ('I2', 'i2_'), ('M1', 'm1_'), ('M2', 'm2_'), ('RBM', 'rbm_')]:
        if pf in comps and f'{pf}height' in p:
            area = get_area(pf)
            ratio_to_rbm = safe_div(area, area_rbm) if area_rbm > 0 else np.nan
            all_peaks_data.append({
                'Sample': filename,
                'Chirality': cfg["label"],
                'Peak_Name': tag,
                'Center': round(p[f'{pf}center'].value, 2),
                'Height': round(p[f'{pf}height'].value, 2),
                'Area': round(area, 2),
                'FWHM': round(p[f'{pf}fwhm'].value, 2),
                'Area_Ratio_to_RBM': ratio_to_rbm
            })

    return ratio_row, comps, bkg, all_peaks_data


def plot_result(x, y_raw, fit_result, comps, bkg, filename, chirality_key, output_dir):
    cfg = CHIRALITY_DB[chirality_key]
    has_rbm = fit_result._has_rbm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    total_fit_with_bkg = bkg + fit_result.best_fit

    ax1.plot(x, y_raw, '.', color='dimgray', ms=4, label='Raw Data')
    ax1.plot(x, total_fit_with_bkg, 'r-', lw=2.2, label=f'Total Fit (χ²ᵣ={fit_result.redchi:.2f})')
    ax1.plot(x, bkg, '--', color='gray', lw=1.5, label='SNIP Baseline')

    def fill(prefix, color, label):
        if prefix in comps:
            ax1.fill_between(x, bkg, bkg + comps[prefix], alpha=0.45, color=color, label=label)

    fill('i1_', 'orange', 'I₁ Peak (~104 cm⁻¹)')
    fill('i2_', 'forestgreen', 'I₂ Peak (~145 cm⁻¹)')
    fill('m1_', 'cyan', 'M1 Band')
    fill('m2_', 'dodgerblue', 'M2 Band')

    if has_rbm:
        fill('rbm_', 'magenta', f'{cfg["label"]} RBM')

    ax1.set_xlim(50, 400)
    ax1.set_ylabel('Raman Intensity', fontsize=12)
    ax1.set_title(f'{filename}  [{cfg["label"]}] (SNIP Auto Adaptive Mode)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    norm_res = (y_raw - total_fit_with_bkg) / np.sqrt(np.maximum(y_raw, 1))
    ax2.plot(x, norm_res, '.', color='steelblue', ms=3)
    ax2.axhline(0, color='k', ls='--')
    ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Norm. Residual', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitted_spectra", f"{filename}_Fit.png"), dpi=200)
    plt.close()


def main():
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askquestion("数据选择", "是否批量选择文件夹？\n[是] 整个文件夹\n[否] 手动多选文件")
    paths = []
    if choice == 'yes':
        folder_path = filedialog.askdirectory(title="选择包含需要自动处理的 Raman 数据的文件夹")
        if not folder_path: return
        paths.extend(glob.glob(os.path.join(folder_path, "*.txt")))
        paths.extend(glob.glob(os.path.join(folder_path, "*.csv")))
    else:
        selected_files = filedialog.askopenfilenames(title="选择需要自动处理的 Raman 文件",
                                                     filetypes=[("Data Files", "*.txt *.csv")])
        paths = list(selected_files)

    if not paths: return

    output_root = filedialog.askdirectory(title="选择保存分析结果的文件夹")
    if not output_root: return
    os.makedirs(os.path.join(output_root, "fitted_spectra"), exist_ok=True)

    all_ratios = []
    all_peaks_list = []
    list_raw, list_fit, list_baseline, list_comps = [], [], [], []

    print("\n🚀 启动全自动智能拟合 (优化外推版 SNIP)...\n")

    for filepath in paths:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        chirality_key = parse_chirality(filename)
        if not chirality_key: continue
        print(f"▶ 处理: {filename} ", end="", flush=True)
        try:
            x, y_raw = load_and_cut_data(filepath)
            fit_result = run_fit(x, y_raw, chirality_key)
            ratio_row, comps, bkg, peaks_data = extract_results(x, y_raw, fit_result, filename, chirality_key)

            all_ratios.append(ratio_row)
            all_peaks_list.extend(peaks_data)

            list_raw.append(pd.DataFrame({f'{filename}_X': x, f'{filename}_Raw': y_raw}))
            list_fit.append(pd.DataFrame({f'{filename}_X': x, f'{filename}_Fit': bkg + fit_result.best_fit}))
            list_baseline.append(pd.DataFrame({f'{filename}_X': x, f'{filename}_Baseline': bkg}))
            comp_dict = {f'{filename}_X': x}
            for tag, pf in [('I1', 'i1_'), ('I2', 'i2_'), ('RBM', 'rbm_'), ('M1', 'm1_'), ('M2', 'm2_')]:
                comp_dict[f'{filename}_{tag}'] = comps.get(pf, np.zeros_like(x))
            list_comps.append(pd.DataFrame(comp_dict))

            plot_result(x, y_raw, fit_result, comps, bkg, filename, chirality_key, output_root)
            print(" ✔️")
        except Exception as e:
            print(f" ❌ 失败! ({e})")

    if all_ratios:
        excel_path = os.path.join(output_root, "Raman_Results_AutoSNIP.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(all_ratios).to_excel(writer, sheet_name='Core_Ratios', index=False)

            if all_peaks_list:
                df_all_peaks = pd.DataFrame(all_peaks_list)
                df_all_peaks.to_excel(writer, sheet_name='All_Peaks_Summary', index=False)

            pd.concat(list_raw, axis=1).to_excel(writer, sheet_name='Raw_Data', index=False)
            pd.concat(list_fit, axis=1).to_excel(writer, sheet_name='Total_Fit', index=False)
            pd.concat(list_baseline, axis=1).to_excel(writer, sheet_name='Baseline', index=False)
            pd.concat(list_comps, axis=1).to_excel(writer, sheet_name='Components', index=False)

        print(f"\n🎉 处理完毕！报表和图表已输出至：{output_root}")
        messagebox.showinfo("完成", f"智能提取完毕！\n结果请查看：\n{output_root}")


if __name__ == "__main__":
    main()