# -*- coding: utf-8 -*-
"""
功能: 氨基酸传感阵列特征数据可视化与数据导出脚本 (交互式 & 最终版)
描述:
本脚本会弹出对话框让用户选择输入/输出路径，生成一系列
符合出版物风格的图表，并同时将所有用于绘图的数据
导出到一个Excel文件中，方便在Origin等软件中进行后续处理。

注意: 需要安装 openpyxl 库才能导出为 .xlsx 文件 (pip install openpyxl)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog


# --- 1. 核心样式应用函数 (恢复到简洁专业版) ---

def apply_publication_style(ax, xlabel, ylabel, title):
    """将指定的出版物风格应用于一个matplotlib Axes对象。"""
    try:
        plt.rcParams['font.family'] = 'Arial'
    except RuntimeError:
        print("警告: 未找到 Arial 字体。图表将使用默认字体。")

    ax.grid(False)
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')


# --- 2. 绘图与数据导出函数 (已更新) ---

def plot_feature_stability(df, output_dir, excel_writer):
    """制作正文图 2A: 特征稳定性图，并导出数据"""
    target_aa = 'L-Asp'
    target_conc = df[df['AA'] == target_aa]['浓度/uM'].quantile(0.5, interpolation='nearest')
    subset = df[(df['AA'] == target_aa) & (df['浓度/uM'] == target_conc)]
    if subset.empty: return

    feature_cols = [col for col in df.columns if col not in ['AA', '浓度/uM']]
    means = subset[feature_cols].mean()
    stds = subset[feature_cols].std()

    # --- 数据导出 ---
    data_to_export = pd.DataFrame({'Mean': means, 'StdDev': stds})
    data_to_export.to_excel(excel_writer, sheet_name='Fig2A_Stability_Data')

    # --- 绘图 ---
    clean_labels = [col.replace('"', '').replace('_', '\n') for col in feature_cols]
    fig, ax = plt.subplots(figsize=(16, 9))
    means.plot(kind='bar', yerr=stds, capsize=5, color=sns.color_palette("viridis", len(feature_cols)), ax=ax, zorder=2)
    ax.set_xticklabels(clean_labels, rotation=0, fontsize=20, fontweight='bold')
    apply_publication_style(ax, "Features", "Response Value", f'Feature Stability: {target_aa} at {target_conc}uM')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'Fig2A_Feature_Stability.png')
    plt.savefig(save_path, dpi=300);
    plt.close()
    print(f"  - 已生成图表: Fig2A_Feature_Stability.png")
    print(f"  - 已导出数据到Excel工作表: Fig2A_Stability_Data")


def plot_global_heatmap(df, output_dir, excel_writer):
    """制作正文图 3: 全局响应指纹热力图，并导出数据"""
    feature_cols = [col for col in df.columns if col not in ['AA', '浓度/uM']]
    amino_acids = sorted(df['AA'].unique())
    heatmap_data = df.groupby('AA')[feature_cols].mean().reindex(amino_acids)

    # --- 数据导出 ---
    heatmap_data.to_excel(excel_writer, sheet_name='Fig3_Heatmap_Data')

    # --- 绘图 ---
    clean_labels = [col.replace('"', '') for col in feature_cols]
    heatmap_data.columns = clean_labels
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(heatmap_data, cmap='viridis', annot=False, linewidths=.5, ax=ax)
    ax.set_xlabel("Features", fontsize=24, fontweight='bold')
    ax.set_ylabel("Amino Acid", fontsize=24, fontweight='bold')
    ax.set_title("Response Fingerprint of 18 Amino Acids", fontsize=26, fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'Fig3_Global_Fingerprint_Heatmap.png')
    plt.savefig(save_path, dpi=300);
    plt.close()
    print(f"  - 已生成图表: Fig3_Global_Fingerprint_Heatmap.png")
    print(f"  - 已导出数据到Excel工作表: Fig3_Heatmap_Data")


def plot_pca_scatter(df_scaled, labels, output_dir, excel_writer):
    """制作新增核心图: PCA降维可视化，并导出数据"""
    feature_cols = df_scaled.columns
    amino_acids = sorted(labels.unique())
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Amino Acid'] = labels.values

    # --- 数据导出 ---
    pca_df.to_excel(excel_writer, sheet_name='Fig4_PCA_Data', index=False)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.scatterplot(x='PC1', y='PC2', hue='Amino Acid', palette='tab20', data=pca_df, legend='full', s=100, alpha=0.8,
                    ax=ax)
    apply_publication_style(ax, f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})',
                            f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', 'PCA of 8D Features')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = os.path.join(output_dir, 'Fig4_PCA_Scatter.png')
    plt.savefig(save_path, dpi=300);
    plt.close()
    print(f"  - 已生成图表: Fig4_PCA_Scatter.png")
    print(f"  - 已导出数据到Excel工作表: Fig4_PCA_Data")


def plot_concentration_response(df, output_dir, excel_writer):
    """制作SI图集S2: 所有特征的浓度响应总览，并导出数据"""
    feature_cols = [col for col in df.columns if col not in ['AA', '浓度/uM']]
    amino_acids = sorted(df['AA'].unique())

    for feature in feature_cols:
        clean_feature_name = feature.replace('"', '')

        # --- 数据导出 (整理成Origin友好格式) ---
        data_for_excel = df.groupby(['AA', '浓度/uM'])[feature].mean().reset_index()
        pivoted_data = data_for_excel.pivot(index='浓度/uM', columns='AA', values=feature)
        pivoted_data.to_excel(excel_writer, sheet_name=f'S2_ConcResponse_{clean_feature_name}')

        # --- 绘图 ---
        fig, ax = plt.subplots(figsize=(14, 9))
        sns.lineplot(data=df, x='浓度/uM', y=feature, hue='AA', palette='tab20', legend=False, errorbar=None, ax=ax)
        apply_publication_style(ax, "Concentration / uM", "Response Value",
                                f"Concentration Response: {clean_feature_name}")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'S2_ConcResponse_{clean_feature_name}.png')
        plt.savefig(save_path, dpi=300);
        plt.close()
    print(f"  - 已生成S2系列图表 (共{len(feature_cols)}张)")
    print(f"  - 已导出S2系列数据到Excel (共{len(feature_cols)}个工作表)")


def plot_channel_discrimination(df, output_dir, excel_writer):
    """制作SI图集S3: 单通道区分能力剖析，并导出数据"""
    feature_cols = [col for col in df.columns if col not in ['AA', '浓度/uM']]
    amino_acids = sorted(df['AA'].unique())
    target_conc = df['浓度/uM'].max()
    subset = df[df['浓度/uM'] == target_conc]
    if subset.empty: return

    for feature in feature_cols:
        clean_feature_name = feature.replace('"', '')

        # --- 数据导出 ---
        mean_responses = subset.groupby('AA')[feature].mean().reindex(amino_acids)
        mean_responses.to_excel(excel_writer, sheet_name=f'S3_Discrimination_{clean_feature_name}')

        # --- 绘图 ---
        fig, ax = plt.subplots(figsize=(16, 9))
        mean_responses.plot(kind='bar', color=sns.color_palette("viridis", len(amino_acids)), ax=ax, zorder=2)
        apply_publication_style(ax, "Amino Acid", "Mean Response Value",
                                f'Channel Discrimination ({target_conc}uM): {clean_feature_name}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'S3_ChannelDiscrimination_{clean_feature_name}.png')
        plt.savefig(save_path, dpi=300);
        plt.close()
    print(f"  - 已生成S3系列图表 (共{len(feature_cols)}张)")
    print(f"  - 已导出S3系列数据到Excel (共{len(feature_cols)}个工作表)")


# --- 3. 主程序执行 ---

if __name__ == '__main__':
    root = tk.Tk();
    root.withdraw()
    print("程序启动，请根据弹窗提示操作。")

    train_file = filedialog.askopenfilename(title="请选择训练数据文件 (train_data.csv)")
    if not train_file: exit("未选择训练文件，程序退出。")
    test_file = filedialog.askopenfilename(title="请选择测试数据文件 (test_data.csv)")
    if not test_file: exit("未选择测试文件，程序退出。")
    output_dir = filedialog.askdirectory(title="请选择一个文件夹用于保存输出")
    if not output_dir: exit("未选择输出文件夹，程序退出。")

    # 创建输出子目录
    MANUSCRIPT_DIR = os.path.join(output_dir, 'Manuscript_Figures')
    SI_DIR = os.path.join(output_dir, 'SI_Figures')
    os.makedirs(MANUSCRIPT_DIR, exist_ok=True)
    os.makedirs(SI_DIR, exist_ok=True)

    # 创建Excel写入对象
    excel_path = os.path.join(output_dir, 'plot_data_for_origin.xlsx')
    excel_writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    try:
        full_df = pd.concat([pd.read_csv(train_file), pd.read_csv(test_file)], ignore_index=True)
        feature_cols_main = [c for c in full_df.columns if c not in ['AA', '浓度/uM']]
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(full_df[feature_cols_main]), columns=feature_cols_main)
    except Exception as e:
        excel_writer.close()
        exit(f"错误: 读取或处理数据时出错。\n{e}")

    print("\n--- 开始生成图表与数据文件 ---")

    print("\n[阶段一: 正文图表与数据]")
    plot_feature_stability(full_df, MANUSCRIPT_DIR, excel_writer)
    plot_global_heatmap(full_df, MANUSCRIPT_DIR, excel_writer)
    plot_pca_scatter(df_scaled, full_df['AA'], MANUSCRIPT_DIR, excel_writer)

    print("\n[阶段二: 支持信息(SI)图表与数据]")
    plot_concentration_response(full_df, SI_DIR, excel_writer)
    plot_channel_discrimination(full_df, SI_DIR, excel_writer)

    # 保存并关闭Excel文件
    excel_writer.close()

    print("\n--- 所有任务已完成 ---")
    print(f"图表已保存在以下两个目录中:\n  - '{MANUSCRIPT_DIR}'\n  - '{SI_DIR}'")
    print(f"所有绘图数据已导出至Excel文件:\n  - '{excel_path}'")
