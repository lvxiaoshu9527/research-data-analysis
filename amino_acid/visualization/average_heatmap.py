# -*- coding: utf-8 -*-
"""
功能: 生成训练集与测试集的总览热图 (按氨基酸-浓度分组)
修改:
1. 只输出两张图：训练集总图、测试集总图。
2. 结构：Y轴=特征，X轴=氨基酸(内含浓度梯度)。
3. 数据：相同浓度取平均。
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


# --- 1. 样式与颜色定义 ---
def get_custom_cmap():
    """定义红-白-蓝绿色的自定义色图"""
    red = (235 / 255, 31 / 255, 35 / 255)
    white = (1.0, 1.0, 1.0)
    teal = (19 / 255, 132 / 255, 154 / 255)
    colors = [teal, "#F0F8FF", white, "#FFF0F5", red]
    nodes = [0.0, 0.49, 0.5, 0.51, 1.0]
    return LinearSegmentedColormap.from_list("custom_heatmap", list(zip(nodes, colors)))


def apply_publication_style(ax, xlabel, ylabel, title, border_width=2):
    """应用出版级图表样式"""
    try:
        plt.rcParams['font.family'] = 'Arial'
    except RuntimeError:
        pass  # 如果没有Arial字体则使用默认

    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)

    # 加粗坐标轴线
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)


# --- 2. 核心绘图逻辑 ---
def plot_grouped_heatmap(df_raw, dataset_name, output_dir, cmap):
    """
    生成按 '氨基酸' 和 '浓度' 分组的平均热图。
    逻辑: 标准化 -> 按(AA, 浓度)分组求均值 -> 转置绘图
    """
    print(f"\n[正在处理: {dataset_name}] 生成分组热图...")

    # 1. 数据准备与标准化
    feature_cols = [c for c in df_raw.columns if c not in ['AA', '浓度/uM']]

    # 确保AA列没有多余空格
    df_raw['AA'] = df_raw['AA'].str.strip()

    # 标准化 (Z-score)
    scaler = StandardScaler()
    df_scaled = df_raw.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_raw[feature_cols])

    # 2. 排序与分组聚合
    # 获取所有氨基酸并排序 (可根据需要改为特定顺序)
    unique_aas = sorted(df_scaled['AA'].unique())

    # 将AA设为Categorical类型以保证排序正确
    df_scaled['AA'] = pd.Categorical(df_scaled['AA'], categories=unique_aas, ordered=True)

    # 按 氨基酸 -> 浓度 排序
    df_sorted = df_scaled.sort_values(by=['AA', '浓度/uM'])

    # 分组求均值 (合并平行样)
    # 结果索引将是 MultiIndex (AA, 浓度/uM)
    heatmap_data_grouped = df_sorted.groupby(['AA', '浓度/uM'], observed=True)[feature_cols].mean()

    # 3. 转换为热图格式 (Y轴=特征, X轴=样本)
    # 转置后: 行是特征, 列是 (AA, 浓度)
    heatmap_data = heatmap_data_grouped.transpose()

    # 清理特征名称 (去除可能存在的引号)
    heatmap_data.index = [c.replace('"', '') for c in heatmap_data.index]

    # 4. 绘图
    # 计算图表宽度: 氨基酸数量 * 浓度数量 * 缩放系数 (保证格子是方形或适中)
    n_cols = heatmap_data.shape[1]
    n_rows = heatmap_data.shape[0]
    fig_width = max(20, n_cols * 0.3)
    fig_height = max(10, n_rows * 0.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 设置颜色范围 (对称)
    vmax = np.abs(heatmap_data.values).max()

    sns.heatmap(heatmap_data, cmap=cmap, ax=ax,
                vmin=-vmax, vmax=vmax, center=0,
                xticklabels=False,  # 暂时隐藏X轴标签，后面手动添加
                yticklabels=True,
                cbar_kws={'label': 'Scaled Response (Z-score)'})

    # 5. 添加视觉分隔线和自定义X轴标签
    # 获取每个氨基酸包含的列数 (即浓度数量)
    # heatmap_data.columns 是 MultiIndex, level 0 是 AA
    aa_counts = heatmap_data.columns.get_level_values(0).value_counts().reindex(unique_aas)

    current_pos = 0
    tick_positions = []
    tick_labels = []

    for aa in unique_aas:
        count = aa_counts[aa]
        if pd.isna(count) or count == 0: continue

        # 画竖线分隔不同氨基酸
        if current_pos > 0:
            ax.axvline(x=current_pos, color='black', linewidth=3, linestyle='-')

        # 记录标签位置 (居中)
        tick_positions.append(current_pos + count / 2)
        tick_labels.append(aa)

        current_pos += count

    # 设置X轴
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', which='major', length=0, pad=10)

    # 设置Y轴标签水平显示
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16, fontweight='bold')

    # 应用样式
    title_text = f"{dataset_name} Group - Total Fingerprint"
    apply_publication_style(ax, "Amino Acid", "Features", title_text)

    # 6. 保存
    plt.tight_layout()
    filename = f"{dataset_name}_Total_Heatmap.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  - 已保存: {filename}")

    # 保存Excel数据备查
    excel_name = os.path.join(output_dir, f"{dataset_name}_Processed_Data.xlsx")
    heatmap_data.to_excel(excel_name)
    print(f"  - 数据已导出: {dataset_name}_Processed_Data.xlsx")


# --- 3. 主程序执行 ---
if __name__ == '__main__':
    try:
        # 初始化 Tkinter 窗口 (隐藏)
        root = tk.Tk()
        root.withdraw()

        print("=== 热图生成程序 (训练集/测试集 分组版) ===")
        print("请按提示选择文件...")

        # 1. 选择文件
        train_file = filedialog.askopenfilename(title="请选择 [训练集] 数据文件 (.csv)")
        if not train_file: exit("未选择训练文件。")

        test_file = filedialog.askopenfilename(title="请选择 [测试集] 数据文件 (.csv)")
        if not test_file: exit("未选择测试文件。")

        output_dir = filedialog.askdirectory(title="请选择 [结果保存文件夹]")
        if not output_dir: exit("未选择输出目录。")

        # 2. 读取数据
        print("\n正在读取数据...")
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # 简单的数据清洗 (修正拼写错误)
        for df in [train_df, test_df]:
            if 'AA' in df.columns:
                df['AA'] = df['AA'].replace('L-Lle', 'L-Ile')

        # 3. 生成热图
        custom_cmap = get_custom_cmap()

        # 生成训练集热图
        plot_grouped_heatmap(train_df, "Train", output_dir, custom_cmap)

        # 生成测试集热图
        plot_grouped_heatmap(test_df, "Test", output_dir, custom_cmap)

        print("\n=== 所有任务完成 ===")
        print(f"结果已保存在: {output_dir}")

    except Exception as e:
        print(f"\n!!! 发生错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        input("\n按回车键退出...")