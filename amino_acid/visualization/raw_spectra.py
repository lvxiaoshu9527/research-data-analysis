# -*- coding: utf-8 -*-
"""
功能: 终极版原始光谱数据批量处理、可视化与导出脚本 (最终分类版)
描述:
此版本根据用户最终澄清的需求进行了彻底重构。核心逻辑变更：
1.  **按文件夹名称分类**：脚本不再通过浓度值来划分数据集。它会直接读取数据所在的文件夹名称，
    如果名称包含 "train"，则将其归为训练组；如果包含 "conc"，则归为预测组。
2.  **分离输出**：为训练组和预测组分别生成图表，并保存在两个独立的文件夹中：
    - `Spectra_Train_Plots`
    - `Spectra_Prediction_Plots`
3.  **数据对齐修复 (v2)**：采用更稳健的 `pandas.concat` 和 `groupby` 方法来合并和计算重复实验的
    均值/标准差。这彻底解决了因不同文件间波长或浓度列不匹配导致的
    `ValueError: all input arrays must have the same shape` 错误。
4.  **保留关键修复**：之前版本中关于X轴范围显示的修复被保留并应用。

本脚本旨在完全自动化地根据源文件结构对数据进行分类、绘图和保存。

注意: 需要安装 openpyxl 和 numpy 库 (pip install openpyxl numpy)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

# --- 1. 全局设置 ---
WAVELENGTH_RANGES = {
    '(6,5)': (900, 1250),
    'S7-(6,5)': (900, 1250),
    '(7,5)': (950, 1300),
    '(8,3)': (900, 1200),
    'default': (900, 1300)
}


# --- 2. 核心样式应用函数 ---
def apply_publication_style(ax, xlabel, ylabel, title, legend=True):
    """将指定的出版物风格应用于一个matplotlib Axes对象。"""
    try:
        plt.rcParams['font.family'] = 'Arial'
    except RuntimeError:
        print("警告: 未找到 Arial 字体。图表将使用默认字体。")

    ax.grid(False)
    ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')
    if legend and ax.get_legend() is not None:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=12, frameon=False, loc='best', ncol=2)


# --- 3. 数据查找与分类解析 ---
def find_and_classify_data(parent_dir):
    """
    扫描父目录，根据文件夹名称将数据分为训练组和预测组。
    """
    train_data = defaultdict(list)
    prediction_data = defaultdict(list)
    print("\n--- 开始扫描并分类数据文件 ---")

    for root, dirs, files in os.walk(parent_dir):
        folder_name = os.path.basename(root).lower()

        current_dict = None
        if 'train' in folder_name:
            current_dict = train_data
        elif 'conc' in folder_name:
            current_dict = prediction_data
        else:
            continue

        for file in files:
            if file.endswith(".xlsx") and "output_data-" in file:
                try:
                    base_name = file.replace(".xlsx", "")
                    parts = base_name.split("output_data-")
                    info_part = parts[-1]
                    replicate_str = info_part[-1]
                    aa_name = info_part[:-1]

                    if not replicate_str.isdigit(): continue

                    cnt_type = os.path.basename(root)
                    file_path = os.path.join(root, file)
                    df = pd.read_excel(file_path, sheet_name='All')

                    if df.shape[1] > 1:
                        df.rename(columns={df.columns[0]: 'Wavelength'}, inplace=True)
                        current_dict[(cnt_type, aa_name)].append(df)

                except Exception as e:
                    print(f"  - 警告: 处理文件失败 {file}, 错误: {e}")

    print("\n--- 扫描分类完成 ---")
    print(f"找到 {sum(len(v) for v in train_data.values())} 个训练组数据文件。")
    print(f"找到 {sum(len(v) for v in prediction_data.values())} 个预测组数据文件。")
    return train_data, prediction_data


# --- 4. 通用绘图与数据导出函数 ---
def generate_plots_and_excel(data_dict, output_dir, folder_name, plot_suffix):
    """
    一个通用的函数，为给定的数据集（训练或预测）生成图表和Excel。
    """
    if not data_dict:
        print(f"\n未找到 {plot_suffix} 数据，跳过绘图。")
        return

    plots_dir = os.path.join(output_dir, folder_name)
    os.makedirs(plots_dir, exist_ok=True)

    excel_path = os.path.join(output_dir, f'original_spectra_data_{plot_suffix.lower()}.xlsx')
    excel_writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    unique_aas = sorted(list(set([key[1] for key in data_dict.keys()])))

    for aa in unique_aas:
        print(f"  - 正在为 {plot_suffix} 组绘制氨基酸: {aa}")
        relevant_cnts = sorted(list(set([k[0] for k in data_dict if k[1] == aa])))
        if not relevant_cnts: continue

        fig, axes = plt.subplots(len(relevant_cnts), 1, figsize=(12, 8 * len(relevant_cnts)), sharex=False,
                                 squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f'Spectral Response for {aa} ({plot_suffix} Set)', fontsize=30, fontweight='bold')

        full_aa_data_for_excel = pd.DataFrame()

        for j, cnt in enumerate(relevant_cnts):
            ax = axes[j]
            replicates = data_dict.get((cnt, aa), [])
            if not replicates: continue

            # --- 最终错误修复：使用concat和groupby进行稳健的数据对齐和计算 ---
            indexed_dfs = []
            for i, df in enumerate(replicates):
                df_indexed = df.set_index('Wavelength')
                df_indexed.columns = pd.to_numeric(df_indexed.columns, errors='coerce')
                indexed_dfs.append(df_indexed)

            try:
                # 1. 智能合并所有重复实验，自动对齐波长轴
                combined_df = pd.concat(indexed_dfs, axis=1, keys=range(len(indexed_dfs)), join='outer')
                # 2. 对整个合并后的数据进行插值，填充所有缺失值
                combined_df.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')
                # 3. 按浓度列（列索引的第二层）分组，计算均值和标准差
                mean_df = combined_df.groupby(level=1, axis=1).mean()
                std_df = combined_df.groupby(level=1, axis=1).std()
                wavelength_col = mean_df.index
            except Exception as e:
                print(f"    - 错误: 无法处理 {cnt} - {aa} 的数据。可能是列名问题。错误: {e}")
                ax.text(0.5, 0.5, 'Data Processing Error', ha='center', va='center', fontsize=15, color='red')
                continue
            # --- 修复结束 ---

            # 绘图
            palette = sns.color_palette("viridis", n_colors=len(mean_df.columns))
            wl_range = WAVELENGTH_RANGES.get(cnt.split('-')[0], WAVELENGTH_RANGES['default'])
            mask = (wavelength_col >= wl_range[0]) & (wavelength_col <= wl_range[1])

            for k, conc in enumerate(sorted(mean_df.columns)):
                ax.plot(wavelength_col[mask], mean_df[conc][mask], label=f'{conc} uM', color=palette[k], linewidth=2)
                ax.fill_between(wavelength_col[mask],
                                (mean_df[conc] - std_df[conc])[mask],
                                (mean_df[conc] + std_df[conc])[mask],
                                color=palette[k], alpha=0.2)

            ax.set_xlim(wl_range)
            apply_publication_style(ax, "" if j < len(relevant_cnts) - 1 else "Wavelength (nm)", "Intensity (a.u.)",
                                    cnt)

            # 准备Excel数据
            excel_sheet_df = pd.DataFrame(index=wavelength_col)
            for conc_col in sorted(mean_df.columns):
                excel_sheet_df[f'{cnt}_{conc_col}_mean'] = mean_df[conc_col]
                excel_sheet_df[f'{cnt}_{conc_col}_std'] = std_df[conc_col]
            if full_aa_data_for_excel.empty:
                full_aa_data_for_excel = excel_sheet_df.reset_index().rename(columns={'index': 'Wavelength'})
            else:
                current_excel_df = excel_sheet_df.reset_index().rename(columns={'index': 'Wavelength'})
                full_aa_data_for_excel = pd.merge(full_aa_data_for_excel, current_excel_df, on='Wavelength',
                                                  how='outer')

        if not full_aa_data_for_excel.empty:
            full_aa_data_for_excel.to_excel(excel_writer, sheet_name=aa, index=False)

        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_path = os.path.join(plots_dir, f'Spectra_{plot_suffix}_{aa}.png')
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    excel_writer.close()
    print(f"  - {plot_suffix} 组图表已保存至 '{plots_dir}'")
    print(f"  - {plot_suffix} 组数据已导出至 '{excel_path}'")


# --- 5. 主程序执行 ---
if __name__ == '__main__':
    root = tk.Tk();
    root.withdraw()
    print("程序启动，请根据弹窗提示操作。")
    parent_dir = filedialog.askdirectory(title="请选择包含所有碳管数据文件夹的“父目录”")
    if not parent_dir: exit("未选择父目录，程序退出。")
    output_dir = filedialog.askdirectory(title="请选择一个文件夹用于保存输出")
    if not output_dir: exit("未选择输出文件夹，程序退出。")

    train_data, prediction_data = find_and_classify_data(parent_dir)

    print("\n--- 开始生成训练组图表 ---")
    generate_plots_and_excel(train_data, output_dir, 'Spectra_Train_Plots', 'Train')

    print("\n--- 开始生成预测组图表 ---")
    generate_plots_and_excel(prediction_data, output_dir, 'Spectra_Prediction_Plots', 'Prediction')

    print("\n--- 所有任务已完成 ---")
