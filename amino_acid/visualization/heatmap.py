# -*- coding: utf-8 -*-
"""
功能: 最终发布版的热图与聚类图生成脚本 (含交错批次校正)
描述:
本脚本为最终的科学发布版本。包含：
1. 原有的标准化分析（总图、独立浓度图、树状图）。
2. 新增：基于交错梯度的批次校正（Interleaved Batch Correction）。
3. 新增：生成“校正后原始长图”，展示真实的浓度-强度梯度关系。
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# --- 1. 定义氨基酸化学分类与颜色 ---
AMINO_ACID_CLASSIFICATION = {
    'Gly': {'class': 'Aliphatic', 'color': '#1f77b4'}, 'Ala': {'class': 'Aliphatic', 'color': '#1f77b4'},
    'Val': {'class': 'Aliphatic', 'color': '#1f77b4'}, 'Leu': {'class': 'Aliphatic', 'color': '#1f77b4'},
    'Ile': {'class': 'Aliphatic', 'color': '#1f77b4'},
    'Met': {'class': 'Aliphatic', 'color': '#1f77b4'},
    'Pro': {'class': 'Aliphatic', 'color': '#1f77b4'},
    'Phe': {'class': 'Aromatic', 'color': '#ff7f0e'}, 'Tyr': {'class': 'Aromatic', 'color': '#ff7f0e'},
    'Trp': {'class': 'Aromatic', 'color': '#ff7f0e'},
    'Ser': {'class': 'Polar', 'color': '#2ca02c'}, 'Thr': {'class': 'Polar', 'color': '#2ca02c'},
    'Cys': {'class': 'Polar', 'color': '#2ca02c'},
    'Lys': {'class': 'Basic', 'color': '#d62728'}, 'Arg': {'class': 'Basic', 'color': '#d62728'},
    'His': {'class': 'Basic', 'color': '#d62728'},
    'Asp': {'class': 'Acidic', 'color': '#9467bd'}, 'Glu': {'class': 'Acidic', 'color': '#9467bd'},
}


# --- 2. 核心绘图与样式函数 ---
def get_custom_cmap():
    red = (235 / 255, 31 / 255, 35 / 255)
    white = (1.0, 1.0, 1.0)
    teal = (19 / 255, 132 / 255, 154 / 255)
    colors = [teal, "#F0F8FF", white, "#FFF0F5", red]
    nodes = [0.0, 0.49, 0.5, 0.51, 1.0]
    return LinearSegmentedColormap.from_list("custom_heatmap", list(zip(nodes, colors)))


def apply_publication_style(ax, xlabel, ylabel, title, border_width=3, font_size_scale=1.0):
    try:
        plt.rcParams['font.family'] = 'Arial'
    except RuntimeError:
        pass
    ax.set_xlabel(xlabel, fontsize=30 * font_size_scale, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=30 * font_size_scale, fontweight='bold')
    ax.set_title(title, fontsize=32 * font_size_scale, fontweight='bold', pad=25 * font_size_scale)
    ax.tick_params(axis='both', which='major', labelsize=26 * font_size_scale)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)


# --- 3. 核心算法: 交错批次校正 ---
def correct_interleaved_data(train_df, test_df):
    """
    功能: 利用两批数据的浓度交错特性 (10,30... vs 20,40...)，
    通过邻近插值计算并消除系统性偏差 (Batch Effect)。
    """
    print("  - 正在执行交错梯度平滑校正...")
    feature_cols = [c for c in train_df.columns if c not in ['AA', '浓度/uM']]

    # 复制数据，避免修改原始输入
    train_corrected = train_df.copy()
    test_corrected = test_df.copy()

    # 临时合并用于计算均值
    combined = pd.concat([train_df, test_df])
    means = combined.groupby(['AA', '浓度/uM'])[feature_cols].mean()

    for aa in combined['AA'].unique():
        if aa not in means.index.get_level_values(0): continue

        aa_means = means.loc[aa]
        concs = sorted(aa_means.index)

        # 识别两批次各自包含的浓度
        train_concs = [c for c in concs if c in train_df['浓度/uM'].unique()]

        if not train_concs: continue

        # 计算该氨基酸下，Train批次相对于Test批次趋势的整体偏移量
        # 逻辑：Train的某浓度值 vs (Test相邻浓度的平均值)
        for col in feature_cols:
            diffs = []
            for c_train in train_concs:
                c_prev = c_train - 10
                c_next = c_train + 10

                val_train = aa_means.loc[c_train, col]
                val_prev = aa_means.loc[c_prev, col] if c_prev in aa_means.index else None
                val_next = aa_means.loc[c_next, col] if c_next in aa_means.index else None

                if val_prev is not None and val_next is not None:
                    # 线性插值：预测该位置理论上应该是多少
                    est = (val_prev + val_next) / 2
                    diffs.append(val_train - est)

            if diffs:
                # 计算平均偏差
                avg_shift = np.mean(diffs)
                # 从Train数据中扣除这个偏差 (校正Train向Test对齐)
                mask = (train_corrected['AA'] == aa)
                train_corrected.loc[mask, col] -= avg_shift

    return pd.concat([train_corrected, test_corrected], ignore_index=True)


# --- 4. 图表绘制函数 ---

def plot_corrected_long_heatmap(df_corrected, output_dir, excel_writer, cmap):
    """生成校正后的原始强度长图（合并平行组）"""
    print("\n[新增阶段: 正在生成“校正后原始长图”...]")
    feature_cols = [c for c in df_corrected.columns if c not in ['AA', '浓度/uM']]

    # 合并平行组：取平均
    df_averaged = df_corrected.groupby(['AA', '浓度/uM'])[feature_cols].mean().reset_index()

    # 排序：AA -> 浓度
    df_sorted = df_averaged.sort_values(by=['AA', '浓度/uM']).reset_index(drop=True)

    # 准备绘图数据
    heatmap_data = df_sorted[feature_cols].transpose()
    clean_labels = [col.replace('"', '') for col in feature_cols]
    heatmap_data.index = clean_labels

    # 保存数据
    heatmap_data.to_excel(excel_writer, sheet_name='Corrected_Raw_Long_Map')

    # 绘图
    fig, ax = plt.subplots(figsize=(30, 10))
    # 使用原始数据的绝对最大值作为色标范围
    vmax = np.abs(heatmap_data.values).max()

    sns.heatmap(heatmap_data, cmap=cmap, ax=ax, vmin=-vmax, vmax=vmax, center=0, xticklabels=False,
                cbar_kws={'label': 'Corrected Intensity (Sample - Baseline)'})

    # 绘制特征分割白线
    ax.hlines(np.arange(1, len(clean_labels)), *ax.get_xlim(), colors='white', linewidth=2)

    # 绘制氨基酸分割黑线和标签
    group_boundaries = df_sorted['AA'].ne(df_sorted['AA'].shift()).to_numpy().nonzero()[0]
    group_labels = df_sorted['AA'].unique()

    tick_positions = []
    for i, boundary in enumerate(group_boundaries):
        if i > 0:
            ax.axvline(x=boundary, color='black', linewidth=2.5, linestyle='-')
        start_pos = boundary
        end_pos = group_boundaries[i + 1] if i + 1 < len(group_boundaries) else len(df_sorted)
        tick_pos = start_pos + (end_pos - start_pos) / 2
        tick_positions.append(tick_pos)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels, rotation=45, ha='center', fontsize=18)
    ax.tick_params(axis='x', which='major', length=0, pad=10)

    apply_publication_style(ax, "", "Features", "Interleaved Corrected Fingerprint (Smooth Gradient)", border_width=2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'Interleaved_Corrected_Fingerprint.png'), dpi=300)
    plt.close()
    print(f"  - 已生成: Interleaved_Corrected_Fingerprint.png (推荐使用此图)")


# (保留原有函数: plot_total_averaged_heatmap, plot_individual..., plot_dendrogram 等)
def plot_total_averaged_heatmap(df_scaled, output_dir, excel_writer, cmap):
    print("\n[阶段一: 正在生成“总图”(平均指纹)...]")
    feature_cols = [c for c in df_scaled.columns if c not in ['AA', '浓度/uM']]
    amino_acids = sorted(df_scaled['AA'].unique())
    heatmap_data = df_scaled.groupby('AA')[feature_cols].mean().reindex(amino_acids)
    heatmap_data.to_excel(excel_writer, sheet_name='Total_Averaged_Fingerprint')
    heatmap_data.columns = [c.replace('"', '') for c in feature_cols]
    fig, ax = plt.subplots(figsize=(16, 14))
    vmax = heatmap_data.abs().max().max()
    sns.heatmap(heatmap_data, cmap=cmap, annot=False, linewidths=.5, ax=ax, vmin=-vmax, vmax=vmax, center=0)
    apply_publication_style(ax, "Features", "Amino Acid", "Averaged Fingerprint of All Amino Acids")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Total_Averaged_Fingerprint.png'), dpi=300)
    plt.close()
    return heatmap_data


def plot_individual_concentration_heatmaps(train_df_raw, test_df_raw, output_dir, excel_writer, cmap):
    print("\n[阶段二: 正在生成每种氨基酸的浓度-指纹热图...]")
    feature_cols = [col for col in train_df_raw.columns if col not in ['AA', '浓度/uM']]
    amino_acids = sorted(pd.concat([train_df_raw, test_df_raw])['AA'].unique())
    individual_dir = os.path.join(output_dir, 'Individual_Concentration_Fingerprints')
    os.makedirs(individual_dir, exist_ok=True)
    processed_items = []

    for aa in amino_acids:
        train_aa_df = train_df_raw[train_df_raw['AA'] == aa]
        test_aa_df = test_df_raw[test_df_raw['AA'] == aa]
        if train_aa_df.empty and test_aa_df.empty: continue
        original_concentration_order = pd.concat([train_aa_df, test_aa_df])['浓度/uM'].unique()

        # 保持原有的单独标准化逻辑，以便于观察相对变化模式
        scaler_train = StandardScaler()
        scaler_test = StandardScaler()
        train_aa_scaled = train_aa_df.copy()
        test_aa_scaled = test_aa_df.copy()
        if not train_aa_df.empty: train_aa_scaled[feature_cols] = scaler_train.fit_transform(train_aa_df[feature_cols])
        if not test_aa_df.empty: test_aa_scaled[feature_cols] = scaler_test.fit_transform(test_aa_df[feature_cols])
        combined_scaled_df = pd.concat([train_aa_scaled, test_aa_scaled])

        final_heatmap_data = combined_scaled_df.groupby('浓度/uM')[feature_cols].mean().reindex(
            original_concentration_order)
        final_excel_data = pd.concat([train_aa_df, test_aa_df]).groupby('浓度/uM')[feature_cols].mean().reindex(
            original_concentration_order)
        processed_items.append({'name': aa, 'heatmap_data': final_heatmap_data, 'excel_data': final_excel_data})

    if not processed_items: return
    full_heatmap_df = pd.concat([item['heatmap_data'] for item in processed_items])
    vmax_global = full_heatmap_df.abs().max().max()

    for item in processed_items:
        aa_name = item['name']
        heatmap_data = item['heatmap_data']
        item['excel_data'].to_excel(excel_writer, sheet_name=f'Data_{aa_name}')
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap=cmap, annot=False, ax=ax, linewidths=.5, vmin=-vmax_global, vmax=vmax_global,
                    center=0, yticklabels=True)
        clean_feature_labels = [col.replace('"', '') for col in feature_cols]
        ax.set_xticklabels(clean_feature_labels, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        apply_publication_style(ax, "Features", "Concentration / uM", f"Concentration-Dependent Fingerprint: {aa_name}",
                                border_width=2, font_size_scale=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f'Conc_Fingerprint_{aa_name}.png'), dpi=300)
        plt.close()


def plot_combined_long_heatmap_standardized(df_scaled, output_dir, excel_writer, cmap):
    # 原版标准化长图
    print("\n[阶段三: 正在生成“标准化长图”(作为对比)...]")
    feature_cols = [col for col in df_scaled.columns if col not in ['AA', '浓度/uM']]
    df_sorted = df_scaled.sort_values(by=['AA', '浓度/uM']).reset_index(drop=True)
    heatmap_data = df_sorted[feature_cols].transpose()
    heatmap_data.to_excel(excel_writer, sheet_name='Total_Combined_Standardized')
    clean_labels = [col.replace('"', '') for col in feature_cols]
    heatmap_data.index = clean_labels
    fig, ax = plt.subplots(figsize=(30, 10))
    vmax = np.abs(heatmap_data.values).max()
    sns.heatmap(heatmap_data, cmap=cmap, ax=ax, vmin=-vmax, vmax=vmax, center=0, xticklabels=False,
                cbar_kws={'label': 'Scaled Response (Z-score)'})
    ax.hlines(np.arange(1, len(clean_labels)), *ax.get_xlim(), colors='white', linewidth=2)
    group_boundaries = df_sorted['AA'].ne(df_sorted['AA'].shift()).to_numpy().nonzero()[0]
    group_labels = df_sorted['AA'].unique()
    tick_positions = []
    for i, boundary in enumerate(group_boundaries):
        if i > 0: ax.axvline(x=boundary, color='black', linewidth=2.5, linestyle='-')
        start_pos = boundary
        end_pos = group_boundaries[i + 1] if i + 1 < len(group_boundaries) else len(df_sorted)
        tick_pos = start_pos + (end_pos - start_pos) / 2
        tick_positions.append(tick_pos)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels, rotation=45, ha='center', fontsize=18)
    ax.tick_params(axis='x', which='major', length=0, pad=10)
    apply_publication_style(ax, "", "Features", "Combined Fingerprint (Standardized)", border_width=2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'Total_Combined_Standardized.png'), dpi=300)
    plt.close()


def plot_dendrogram(averaged_fingerprint_data, output_dir):
    print("\n[阶段四: 正在生成层次聚类树状图...]")
    data_matrix = averaged_fingerprint_data.values
    labels = averaged_fingerprint_data.index.tolist()
    linked = linkage(pdist(data_matrix, metric='euclidean'), method='ward')
    fig, ax = plt.subplots(figsize=(14, 20))
    dendrogram(linked, orientation='left', labels=labels, ax=ax, leaf_font_size=24, above_threshold_color='#808080',
               color_threshold=0)
    for c in ax.collections: c.set_linewidth(3)
    y_labels = ax.get_ymajorticklabels()
    for lbl in y_labels:
        clean_name = lbl.get_text().replace('L-', '').replace('D-', '')
        if clean_name in AMINO_ACID_CLASSIFICATION:
            lbl.set_color(AMINO_ACID_CLASSIFICATION[clean_name]['color'])
            lbl.set_fontweight('bold')
    apply_publication_style(ax, "Euclidean Distance", "", "Hierarchical Clustering")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(output_dir, 'Amino_Acid_Clustering_Dendrogram.png'), dpi=300)
    plt.close()


def create_legend_image(classification_dict, output_dir):
    print("\n[阶段五: 正在生成图例...]")
    legend_info = {}
    for aa_info in classification_dict.values():
        if aa_info['class'] not in legend_info: legend_info[aa_info['class']] = aa_info['color']
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    ax.axis('off')
    for i, (class_name, color) in enumerate(legend_info.items()):
        ax.text(0.1, 0.9 - i * 0.18, class_name, fontsize=24, fontweight='bold', ha='left', va='center', color=color)
    ax.set_title("Amino Acid Classes", fontsize=28, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Dendrogram_Legend.png'), dpi=300, transparent=True)
    plt.close()


# --- 主程序 ---
if __name__ == '__main__':
    try:
        root = tk.Tk()
        root.withdraw()
        print("程序启动，请根据弹窗提示操作。")
        train_file = filedialog.askopenfilename(title="请选择训练数据文件 (train_data.csv)")
        if not train_file: exit("未选择训练文件。")
        test_file = filedialog.askopenfilename(title="请选择测试数据文件 (test_data.csv)")
        if not test_file: exit("未选择测试文件。")
        output_dir = filedialog.askdirectory(title="请选择一个文件夹用于保存输出")
        if not output_dir: exit("未选择输出文件夹。")

        excel_path = os.path.join(output_dir, 'fingerprint_data_output.xlsx')
        print("正在读取数据文件...")
        train_df_raw = pd.read_csv(train_file)
        test_df_raw = pd.read_csv(test_file)

        # 修正拼写错误
        if 'L-Lle' in train_df_raw['AA'].unique(): train_df_raw['AA'] = train_df_raw['AA'].replace('L-Lle', 'L-Ile')
        if 'L-Lle' in test_df_raw['AA'].unique(): test_df_raw['AA'] = test_df_raw['AA'].replace('L-Lle', 'L-Ile')

        # 1. 执行交错校正，得到用于“原始长图”的数据
        df_corrected = correct_interleaved_data(train_df_raw, test_df_raw)

        # 2. 准备用于“标准化图”的数据 (保留原有逻辑: 分别标准化)
        scaler_train = StandardScaler()
        scaler_test = StandardScaler()
        train_scaled_copy = train_df_raw.copy()
        test_scaled_copy = test_df_raw.copy()
        feature_cols_main = [c for c in train_df_raw.columns if c not in ['AA', '浓度/uM']]
        train_scaled_copy[feature_cols_main] = scaler_train.fit_transform(train_df_raw[feature_cols_main])
        test_scaled_copy[feature_cols_main] = scaler_test.fit_transform(test_df_raw[feature_cols_main])
        scaled_df_for_others = pd.concat([train_scaled_copy, test_scaled_copy], ignore_index=True)

        with pd.ExcelWriter(excel_path, engine='openpyxl') as excel_writer:
            custom_cmap = get_custom_cmap()
            print("\n--- 开始生成分析图表 ---")

            # 绘制新功能: 校正后的原始长图
            plot_corrected_long_heatmap(df_corrected, output_dir, excel_writer, custom_cmap)

            # 绘制原有图表
            plot_individual_concentration_heatmaps(train_df_raw, test_df_raw, output_dir, excel_writer, custom_cmap)
            avg_fingerprint_data = plot_total_averaged_heatmap(scaled_df_for_others, output_dir, excel_writer,
                                                               custom_cmap)
            plot_combined_long_heatmap_standardized(scaled_df_for_others, output_dir, excel_writer, custom_cmap)

            if avg_fingerprint_data is not None:
                plot_dendrogram(avg_fingerprint_data, output_dir)
                create_legend_image(AMINO_ACID_CLASSIFICATION, output_dir)

        print("\n--- 所有任务已圆满完成 ---")
        print(f"推荐查看的新图表: 'Interleaved_Corrected_Fingerprint.png'")

    except Exception as e:
        print(f"\n--- 程序发生严重错误 ---\n错误详情: {e}")
        import traceback

        traceback.print_exc()
    finally:
        input("\n程序已运行完毕，按回车键退出...")