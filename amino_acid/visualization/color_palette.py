# -*- coding: utf-8 -*-
"""
功能: 根据用户最终的详细要求，生成一个高度定制化的颜色风格预览图。
描述:
本脚本使用一套经过多轮修改的最终配色方案，分别绘制PCA散点图、
柱状图和热力图，并整合到一张预览图中，供用户最终确认。
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog


# --- 1. 定义最终的专属配色方案 ---

def get_final_palettes():
    """返回一个包含最终定制化色板的字典"""

    # 热图: 翻转后的渐变色板 (玫瑰红/红 -> 白 -> 青绿)
    heatmap_colors_flipped = ["#C71585", "#FF4500", "white",
                              "#20B2AA"]  # MediumVioletRed, OrangeRed, white, LightSeaGreen
    heatmap_nodes_flipped = [0.0, 0.2, 0.5, 1.0]  # 调整节点让红色区域更集中
    final_cmap = LinearSegmentedColormap.from_list(
        "final_heatmap", list(zip(heatmap_nodes_flipped, heatmap_colors_flipped))
    )

    # 柱状图: 采用'tab20'专业色板
    barplot_palette = sns.color_palette('tab20', 18)

    # PCA图: 手工挑选的高对比度、非刺眼颜色
    pca_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7'
    ]  # 这是matplotlib的默认'tab20'色板，但去除了过于鲜亮的颜色并重新排序

    # 重新设计一个更柔和且无黄色的PCA色板
    pca_palette_final = [
        '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD',
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B',
        '#E377C2', '#7F7F7F', '#17BECF', '#AEC7E8', '#FFBB78', '#98DF8A'
    ]

    # 最终决定：使用seaborn的'colorblind'色板，它专业、清晰且无障碍
    pca_palette_professional = sns.color_palette('colorblind', 10) + sns.color_palette('muted', 8)

    return {
        'pca': pca_palette_professional,
        'barplot': barplot_palette,
        'heatmap': final_cmap
    }


# --- 2. 核心绘图函数 (采用最终风格) ---

def draw_pca(ax, df_scaled, labels, palette):
    """绘制PCA散点图"""
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Amino Acid'] = labels.values

    sns.scatterplot(
        x='PC1', y='PC2', hue='Amino Acid',
        palette=palette, data=pca_df,
        legend=False, s=60, alpha=0.9, ax=ax
    )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=16, fontweight='bold')


def draw_barplot(ax, df, palette):
    """绘制带透明度和边线的柱状图"""
    target_conc = df['浓度/uM'].max()
    feature = '(6,5)_intensity'
    subset = df[df['浓度/uM'] == target_conc]
    amino_acids = sorted(df['AA'].unique())
    mean_responses = subset.groupby('AA')[feature].mean().reindex(amino_acids)

    mean_responses.plot(
        kind='bar', color=palette, ax=ax, zorder=2,
        alpha=0.8, edgecolor='black', linewidth=1.5
    )
    ax.set_xlabel('')
    ax.set_ylabel('Response', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=90)


def draw_heatmap(ax, df_scaled, labels, cmap):
    """绘制热力图"""
    feature_cols = df_scaled.columns
    temp_df = df_scaled.copy()
    temp_df['AA'] = labels.values
    amino_acids = sorted(labels.unique())
    heatmap_data = temp_df.groupby('AA')[feature_cols].mean().reindex(amino_acids)

    # 计算颜色映射的中心点
    v_min, v_max = heatmap_data.min().min(), heatmap_data.max().max()
    center = 0 if v_min < 0 < v_max else None

    sns.heatmap(heatmap_data, cmap=cmap, annot=False, linewidths=0, ax=ax, center=center)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0)


# --- 3. 主预览生成函数 ---

def generate_final_preview(df_scaled, full_df, output_path):
    """生成包含所有图表和最终定制风格的预览图"""
    palettes = get_final_palettes()

    try:
        plt.rcParams['font.family'] = 'Arial'
    except RuntimeError:
        print("警告: 未找到 Arial 字体。")

    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    fig.suptitle('最终定制风格预览', fontsize=32, fontweight='bold')

    draw_pca(axes[0], df_scaled, full_df['AA'], palettes['pca'])
    draw_barplot(axes[1], full_df, palettes['barplot'])
    draw_heatmap(axes[2], df_scaled, full_df['AA'], palettes['heatmap'])

    axes[0].set_title('PCA Scatter Plot', fontsize=24, fontweight='bold')
    axes[1].set_title('Bar Plot (Tableau Style)', fontsize=24, fontweight='bold')
    axes[2].set_title('Fingerprint Heatmap (Flipped)', fontsize=24, fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=14)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n最终风格预览图已生成: {output_path}")


# --- 4. 主程序执行 ---

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    print("--- 最终风格预览生成器 ---")

    train_path = filedialog.askopenfilename(title="请选择训练数据文件 (train_data.csv)")
    if not train_path: exit("操作取消。")
    test_path = filedialog.askopenfilename(title="请选择测试数据文件 (test_data.csv)")
    if not test_path: exit("操作取消。")
    output_dir = filedialog.askdirectory(title="请选择一个文件夹用于保存预览图")
    if not output_dir: exit("操作取消。")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        full_df = pd.concat([train_df, test_df], ignore_index=True)
    except Exception as e:
        exit(f"错误: 读取数据文件时出错。\n{e}")

    feature_cols = [col for col in full_df.columns if col not in ['AA', '浓度/uM']]
    scaler = StandardScaler()
    df_scaled_np = scaler.fit_transform(full_df[feature_cols])
    df_scaled = pd.DataFrame(df_scaled_np, columns=feature_cols)

    output_file = os.path.join(output_dir, 'final_style_preview.png')

    generate_final_preview(df_scaled, full_df, output_file)
