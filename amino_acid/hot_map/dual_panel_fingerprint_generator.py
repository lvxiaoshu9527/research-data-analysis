import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, fcluster


def select_file(title):
    """打开文件选择对话框，用于选择单个CSV文件。"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("CSV files", "*.csv")])
    root.destroy()
    return file_path


def get_user_input(prompt, title):
    """通用弹窗让用户输入文本。"""
    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askstring(title, prompt, parent=root)
    root.destroy()
    return user_input


def process_csv_for_heatmap(file_path):
    """
    从CSV文件中提取所有数据，计算每个“氨基酸-传感器-浓度”组合的平均响应，
    构建“全景平均响应”矩阵，并进行行Z-score标准化。
    """
    print(f"正在从文件 '{os.path.basename(file_path)}' 中读取全量数据...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("文件读取错误", f"读取CSV文件时出错: {e}")
        return None

    label_col = 'AA' if 'AA' in df.columns else 'Label'
    conc_col = '浓度/uM' if '浓度/uM' in df.columns else 'Concentration'
    if label_col not in df.columns:
        messagebox.showerror("列名错误", "文件中必须包含氨基酸标签列 ('AA' 或 'Label')。")
        return None

    # --- 核心数据处理逻辑 ---
    # 1. 融化数据，让特征成为一列，方便后续分组
    feature_columns = [col for col in df.columns if col not in [label_col, conc_col]]
    df_melted = df.melt(id_vars=[label_col, conc_col], value_vars=feature_columns, var_name='sensor',
                        value_name='response')

    # 2. 计算每个“氨基酸-传感器-浓度”组合的平均响应值
    avg_response_df = df_melted.groupby([label_col, 'sensor', conc_col])['response'].mean().reset_index()

    # 3. 构建最终的“全景平均响应”矩阵
    # 行是氨基酸，列是传感器和浓度的多层级组合
    full_fingerprint_df = avg_response_df.pivot_table(
        index=label_col,
        columns=['sensor', conc_col],
        values='response'
    )

    full_fingerprint_df.fillna(0, inplace=True)

    # --- 行Z-score标准化 ---
    print("正在对'全景平均响应'矩阵进行行Z-score标准化...")
    if full_fingerprint_df.shape[1] > 1:
        zscore_matrix = full_fingerprint_df.apply(lambda row: zscore(row, nan_policy='omit'), axis=1,
                                                  result_type='expand')
        zscore_matrix.fillna(0, inplace=True)
        zscore_matrix.columns = full_fingerprint_df.columns
    else:
        zscore_matrix = pd.DataFrame(0, index=full_fingerprint_df.index, columns=full_fingerprint_df.columns)

    return zscore_matrix


def plot_publication_heatmap(data_matrix, num_clusters):
    """绘制出版级的、带聚类注释的“全景平均响应”热图。"""
    if data_matrix is None or data_matrix.empty:
        messagebox.showerror("错误", "没有足够的数据来生成热图。")
        return None

    print("\n正在生成最终版指纹热图...")

    try:
        row_linkage = linkage(data_matrix, method='average', metric='euclidean')
        cluster_labels = fcluster(row_linkage, t=num_clusters, criterion='maxclust')

        cluster_colors = sns.color_palette("Set2", num_clusters)
        row_color_map = {label: color for label, color in zip(np.unique(cluster_labels), cluster_colors)}
        row_colors = pd.Series(cluster_labels, index=data_matrix.index, name='Clusters').map(row_color_map)
    except Exception as e:
        messagebox.showerror("聚类错误", f"进行层次聚类时出错: {e}")
        return None

    g = sns.clustermap(
        data_matrix,
        row_linkage=row_linkage,
        row_colors=row_colors,
        col_cluster=False,  # 列代表传感器和浓度，保持原有顺序更具可读性
        cmap='bwr',
        center=0,
        linewidths=.5,
        figsize=(20, 14),  # 调整尺寸以适应48列
        annot=False,
        xticklabels=True,  # 显示X轴标签
        cbar_kws={'label': 'Row Z-score'},
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=90)  # 旋转并缩小X轴标签
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=12, rotation=0)

    g.fig.suptitle('Fingerprint of Amino Acids Across Sensors and Concentrations', fontsize=24, fontweight='bold')
    g.ax_heatmap.set_xlabel("Sensor Channels & Concentrations (µM)", fontsize=16, fontweight='bold')
    g.ax_heatmap.set_ylabel("Amino Acids", fontsize=16, fontweight='bold')
    g.ax_heatmap.tick_params(axis='y', which='major', length=0)
    plt.subplots_adjust(top=0.92, bottom=0.2)  # 为旋转后的X轴标签留出更多空间

    return g.fig


def main():
    """主函数。"""
    try:
        import matplotlib, seaborn, scipy
    except ImportError:
        messagebox.showerror("缺少库",
                             "需要库: matplotlib, seaborn, scipy\n请运行: pip install matplotlib seaborn scipy")
        return

    csv_file_path = select_file("请选择包含所有数据的预处理CSV文件")
    if not csv_file_path: return

    num_clusters_str = get_user_input("您希望将氨基酸分成几个类别 (例如: 5)？", "输入聚类数")
    if not num_clusters_str: return
    try:
        num_clusters = int(num_clusters_str)
        if num_clusters < 2: raise ValueError
    except ValueError:
        messagebox.showerror("输入错误", "请输入一个大于等于2的整数。")
        return

    fingerprint_df = process_csv_for_heatmap(csv_file_path)
    if fingerprint_df is None: return

    print("\n最终生成的'全景平均响应'矩阵 (部分数据预览):")
    print(fingerprint_df.iloc[:, :5].head())

    heatmap_fig = plot_publication_heatmap(fingerprint_df, num_clusters)

    if heatmap_fig:
        save_dir = os.path.dirname(csv_file_path)
        save_path = os.path.join(save_dir, "full_average_fingerprint_heatmap.png")
        try:
            heatmap_fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n热图已成功保存到: {save_path}")
            messagebox.showinfo("成功", f"“全景平均响应”热图已成功生成并保存。")
        except Exception as e:
            messagebox.showerror("保存错误", f"保存图形时出错: {e}")

        plt.show()


if __name__ == "__main__":
    main()
