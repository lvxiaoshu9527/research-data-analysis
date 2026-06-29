import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os


def select_file(title):
    """打开文件选择对话框。"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("CSV files", "*.csv")])
    root.destroy()
    return file_path


def plot_pca_overlay(ax, train_pca_df, test_pca_df, title, unique_labels, color_map, marker_map):
    """
    一个通用的绘图函数，用于绘制PCA叠加图。
    """
    # 绘制训练集椭圆和散点
    for label in unique_labels:
        train_subset = train_pca_df[train_pca_df['label'] == label]
        if train_subset.empty: continue

        ax.scatter(train_subset['PC 1'], train_subset['PC 2'],
                   marker=marker_map[label], color=color_map[label],
                   s=80, alpha=0.8, label=f'Train - {label}')

        if len(train_subset) >= 3:
            try:
                sns.kdeplot(data=train_subset, x='PC 1', y='PC 2', color=color_map[label],
                            alpha=0.1, fill=True, levels=4, thresh=0.1, ax=ax)
            except Exception as e:
                print(f"  - 无法为类别 '{label}' 绘制椭圆: {e}")

    # 绘制测试集散点
    if test_pca_df is not None and not test_pca_df.empty:
        for label in unique_labels:
            test_subset = test_pca_df[test_pca_df['label'] == label]
            if test_subset.empty: continue

            ax.scatter(test_subset['PC 1'], test_subset['PC 2'],
                       marker=marker_map[label], facecolors='none',
                       edgecolors=color_map[label], linewidths=1.5,
                       s=120, alpha=1.0, label=f'Test - {label}')

    # 创建自定义图例
    legend_handles = []
    for label in unique_labels:
        legend_handles.append(plt.Line2D([0], [0], marker=marker_map[label], color=color_map[label],
                                         linestyle='None', markersize=10, label=f'Train - {label}'))
        if test_pca_df is not None and label in test_pca_df['label'].unique():
            legend_handles.append(plt.Line2D([0], [0], marker=marker_map[label], color='w', markerfacecolor='none',
                                             markeredgecolor=color_map[label], markeredgewidth=1.5,
                                             linestyle='None', markersize=10, label=f'Test - {label}'))

    ax.legend(handles=legend_handles, title='图例', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.set_title(title, fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)


def main():
    """主函数，执行所有步骤。"""
    # --- 依赖库检查 ---
    try:
        import matplotlib, seaborn
        from sklearn.decomposition import PCA
    except ImportError:
        messagebox.showerror("缺少库",
                             "需要库: matplotlib, seaborn, scikit-learn\n请运行: pip install matplotlib seaborn scikit-learn")
        return

    # --- Step 1: 数据读取与预处理 ---
    train_file_path = select_file("请选择训练集 (Training Set) CSV 文件")
    if not train_file_path: return
    test_file_path = select_file("请选择测试集 (Test Set) CSV 文件")
    if not test_file_path: return

    try:
        train_df = pd.read_csv(train_file_path)
        test_df = pd.read_csv(test_file_path)
    except Exception as e:
        messagebox.showerror("文件读取错误", f"读取文件时出错: {e}")
        return

    label_col = 'AA' if 'AA' in train_df.columns else 'Label'
    feature_columns = [col for col in train_df.columns if
                       col not in [label_col, '浓度/uM'] and pd.api.types.is_numeric_dtype(train_df[col])]

    # --- 关键逻辑修复：找出在测试集中特征完整的氨基酸 ---
    # 1. 先从测试集中删除所有在特征列上存在缺失值的行
    test_df_complete_features = test_df.dropna(subset=feature_columns)
    # 2. 从这个“干净”的测试集中，获取氨基酸的唯一列表
    complete_test_labels = set(test_df_complete_features[label_col].unique())

    # 3. 找出训练集和这个“干净”测试集的交集
    train_labels_set = set(train_df[label_col].dropna().unique())
    common_labels = sorted(list(train_labels_set.intersection(complete_test_labels)))

    if not common_labels:
        messagebox.showinfo("提示", "训练集和测试集没有特征完整的共有氨基酸，无法生成对比图。")
        return

    print(f"检测到特征完整的共有氨基酸，将只显示以下类别: {', '.join(common_labels)}")

    # --- 使用共有氨基酸来筛选数据 ---
    train_df_focused = train_df[train_df[label_col].isin(common_labels)].copy()
    test_df_focused = test_df[test_df[label_col].isin(common_labels)].copy()

    # --- 清理筛选后的数据 ---
    X_train_focused_raw = train_df_focused[feature_columns]
    y_train_focused_raw = train_df_focused[label_col]
    valid_train_indices = X_train_focused_raw.dropna().index
    X_train_focused = X_train_focused_raw.loc[valid_train_indices]
    y_train_focused = y_train_focused_raw.loc[valid_train_indices]

    X_test_focused_raw = test_df_focused[feature_columns]
    y_test_focused_raw = test_df_focused[label_col]
    valid_test_indices = X_test_focused_raw.dropna().index
    X_test_focused = X_test_focused_raw.loc[valid_test_indices]
    y_test_focused = y_test_focused_raw.loc[valid_test_indices]

    if X_train_focused.empty:
        messagebox.showerror("数据错误", "在筛选共有氨基酸并清理后，没有足够的训练数据。")
        return

    # --- Step 2: 在筛选后的数据上训练模型 ---
    scaler = StandardScaler().fit(X_train_focused)
    pca = PCA(n_components=2).fit(scaler.transform(X_train_focused))

    X_train_pca = pca.transform(scaler.transform(X_train_focused))
    X_test_pca = pca.transform(scaler.transform(X_test_focused)) if not X_test_focused.empty else None

    # --- Step 3: 准备绘图数据并绘图 ---
    train_pca_df = pd.DataFrame(X_train_pca, columns=['PC 1', 'PC 2'])
    train_pca_df['label'] = y_train_focused.values

    test_pca_df = None
    if X_test_pca is not None:
        test_pca_df = pd.DataFrame(X_test_pca, columns=['PC 1', 'PC 2'])
        test_pca_df['label'] = y_test_focused.values

    # 设置颜色和形状映射
    colors = sns.color_palette("husl", len(common_labels))
    markers = ['o', 's', '^', 'P', 'D', 'v', '<', '>', '*', 'H', 'X', 'd', 'p', 'h', '+']
    marker_map = {label: markers[i % len(markers)] for i, label in enumerate(common_labels)}
    color_map = {label: colors[i] for i, label in enumerate(common_labels)}

    fig, ax = plt.subplots(figsize=(16, 12))
    plot_pca_overlay(ax, train_pca_df, test_pca_df,
                     'PCA Focused Comparison Plot',
                     common_labels, color_map, marker_map)

    explained_variance = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC 1 ({explained_variance[0] * 100:.2f}%)', fontsize=14)
    ax.set_ylabel(f'PC 2 ({explained_variance[1] * 100:.2f}%)', fontsize=14)
    fig.tight_layout()

    # --- Step 4: 保存图形 ---
    save_dir = os.path.dirname(train_file_path)
    path = os.path.join(save_dir, "pca_focused_comparison_plot.png")
    fig.savefig(path, dpi=300)
    print(f"核心对比图已保存到: {path}")
    plt.close(fig)

    messagebox.showinfo("成功", f"核心对比图已成功生成并保存到训练集文件所在目录。")
    print("\n所有任务完成。")


if __name__ == "__main__":
    main()
