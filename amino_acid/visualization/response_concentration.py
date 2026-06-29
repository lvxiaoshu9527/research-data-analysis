import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tkinter as tk
from tkinter import filedialog

# --- 全局样式配置 (根据您的要求) ---
STYLE_CONFIG = {
    'figure_facecolor': 'white',
    'axes_facecolor': 'white',
    'axes_linewidth': 2,
    'title_fontsize': 20,
    'title_fontweight': 'bold',
    'label_fontsize': 22,
    'label_fontweight': 'bold',
    'label_xlabel': 'Concentration / uM',  # 英文标签
    'label_ylabel': 'Response',  # 英文标签
    'tick_fontsize': 16  # 刻度字体大小
}


def check_dependencies():
    """检查所需的库是否已安装"""
    try:
        import pandas
        import matplotlib
    except ImportError as e:
        print(f"错误: 缺少必要的库 -> {e.name}")
        print(f"请在您的终端或命令提示符中运行: pip install {e.name}")
        return False
    return True


def set_ax_style(ax, title):
    """应用用户定义的样式到指定的 matplotlib axes"""
    ax.set_facecolor(STYLE_CONFIG['axes_facecolor'])

    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(STYLE_CONFIG['axes_linewidth'])

    # 设置标题
    ax.set_title(title, fontsize=STYLE_CONFIG['title_fontsize'],
                 fontweight=STYLE_CONFIG['title_fontweight'])

    # 设置坐标轴标签
    ax.set_xlabel(STYLE_CONFIG['label_xlabel'], fontsize=STYLE_CONFIG['label_fontsize'],
                  fontweight=STYLE_CONFIG['label_fontweight'])
    ax.set_ylabel(STYLE_CONFIG['label_ylabel'], fontsize=STYLE_CONFIG['label_fontsize'],
                  fontweight=STYLE_CONFIG['label_fontweight'])

    # 设置刻度
    ax.tick_params(axis='both', which='major',
                   labelsize=STYLE_CONFIG['tick_fontsize'],
                   width=STYLE_CONFIG['axes_linewidth'])

    # 移除网格线
    ax.grid(False)


def plot_part1_data_errorbar(df_aa_stats, aa, feature, ax):
    """绘图第一部分：数据点图 (无连线)"""

    # 准备数据
    df_train_plot = df_aa_stats[df_aa_stats['source'] == 'train']
    df_test_plot = df_aa_stats[df_aa_stats['source'] == 'test']

    x_train = df_train_plot['浓度/uM']
    y_train = df_train_plot[f'{feature}_mean']
    yerr_train = df_train_plot[f'{feature}_std']

    x_test = df_test_plot['浓度/uM']
    y_test = df_test_plot[f'{feature}_mean']
    yerr_test = df_test_plot[f'{feature}_std']

    # 绘制 训练集 (Train) 误差棒图 (无连线)
    ax.errorbar(x_train, y_train, yerr=yerr_train,
                fmt='o',  # 'o' 是点 (无连线)
                capsize=5,  # 误差棒顶部的横线宽度
                label='Train (Mean ± SD)',
                color='blue',
                alpha=0.8, markersize=8)

    # 绘制 测试集 (Test) 误差棒图 (无连线)
    ax.errorbar(x_test, y_test, yerr=yerr_test,
                fmt='s',  # 's' 是方块 (无连线)
                capsize=5,
                label='Test (Mean ± SD)',
                color='orange',
                alpha=0.8, markersize=8)

    set_ax_style(ax, title=feature)
    ax.legend()


def plot_part2_summary(df_aa_stats, aa, feature_cols, plot_type, output_dir):
    """绘图第二部分：总图（强度 或 位移）"""
    safe_aa_name = "".join([c if c.isalnum() else "_" for c in aa])
    plot_filename = os.path.join(output_dir, f'AA_{safe_aa_name}_{plot_type.capitalize()}_All.png')

    features_to_plot = [f for f in feature_cols if plot_type in f.lower()]
    colors = plt.cm.get_cmap('tab10', len(features_to_plot))

    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10),
                               facecolor=STYLE_CONFIG['figure_facecolor'],
                               constrained_layout=True)

        df_train_plot = df_aa_stats[df_aa_stats['source'] == 'train']
        df_test_plot = df_aa_stats[df_aa_stats['source'] == 'test']

        for i, feature in enumerate(features_to_plot):
            color = colors(i)

            # Train (实线)
            x_train = df_train_plot['浓度/uM']
            y_train = df_train_plot[f'{feature}_mean']
            yerr_train = df_train_plot[f'{feature}_std']
            ax.errorbar(x_train, y_train, yerr=yerr_train,
                        fmt='-o', capsize=5,
                        label=f'Train: {feature}', color=color, alpha=0.8,
                        linestyle='-', markersize=8)

            # Test (虚线)
            x_test = df_test_plot['浓度/uM']
            y_test = df_test_plot[f'{feature}_mean']
            yerr_test = df_test_plot[f'{feature}_std']
            ax.errorbar(x_test, y_test, yerr=yerr_test,
                        fmt='--s', capsize=5,
                        label=f'Test: {feature}', color=color, alpha=0.8,
                        linestyle='--', markersize=8)

        set_ax_style(ax, title=f'AA: {aa} - All {plot_type.capitalize()} Features')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)
        return plot_filename
    except Exception as e:
        print(f"    ERROR plotting Part 2 ({aa} - {plot_type}): {e}")
        if 'fig' in locals(): plt.close(fig)
        return None


def plot_part3_fit(df_aa_raw, aa, feature, ax):
    """绘图第三部分：拟合图（散点 + 拟合线）"""

    df_train_feat = df_aa_raw[df_aa_raw['source'] == 'train']
    df_test_feat = df_aa_raw[df_aa_raw['source'] == 'test']

    # 绘制散点
    ax.scatter(df_train_feat['浓度/uM'], df_train_feat[feature],
               label='Train Data', alpha=0.5, s=50, c='blue')
    ax.scatter(df_test_feat['浓度/uM'], df_test_feat[feature],
               label='Test Data', alpha=0.5, s=50, c='orange')

    x_min = df_aa_raw['浓度/uM'].min()
    x_max = df_aa_raw['浓度/uM'].max()
    x_range = np.linspace(x_min, x_max, 10)

    # 拟合和绘制回归线 (训练集)
    if not df_train_feat.empty and len(df_train_feat['浓度/uM'].unique()) > 1:
        df_train_clean = df_train_feat[['浓度/uM', feature]].dropna()
        if not df_train_clean.empty and len(df_train_clean['浓度/uM'].unique()) > 1:
            z_train = np.polyfit(df_train_clean['浓度/uM'], df_train_clean[feature], 1)
            p_train = np.poly1d(z_train)
            ax.plot(x_range, p_train(x_range), color='blue', linestyle='-',
                    label=f'Train Fit (Slope: {z_train[0]:.2e})', linewidth=2.5)

    # 拟合和绘制回归线 (测试集)
    if not df_test_feat.empty and len(df_test_feat['浓度/uM'].unique()) > 1:
        df_test_clean = df_test_feat[['浓度/uM', feature]].dropna()
        if not df_test_clean.empty and len(df_test_clean['浓度/uM'].unique()) > 1:
            z_test = np.polyfit(df_test_clean['浓度/uM'], df_test_clean[feature], 1)
            p_test = np.poly1d(z_test)
            ax.plot(x_range, p_test(x_range), color='orange', linestyle='--',
                    label=f'Test Fit (Slope: {z_test[0]:.2e})', linewidth=2.5)

    set_ax_style(ax, title=feature)
    ax.legend()


def run_analysis(train_path, test_path, output_dir):
    """
    运行完整的数据处理和绘图流程
    """

    # --- 1. 加载数据 ---
    try:
        print(f"正在加载训练数据: {train_path}")
        df_train = pd.read_csv(train_path)
        print(f"正在加载测试数据: {test_path}")
        df_test = pd.read_csv(test_path)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # --- 2. 预处理和计算统计量 ---
    print("正在计算均值和标准差...")
    df_train['source'] = 'train'
    df_test['source'] = 'test'
    df_full_raw = pd.concat([df_train, df_test], ignore_index=True)

    concentration_col = '浓度/uM'
    id_cols = ['AA', concentration_col, 'source']
    feature_cols = [col for col in df_train.columns if col not in ['AA', concentration_col, 'source']]

    try:
        df_stats = df_full_raw.groupby(id_cols)[feature_cols].agg(['mean', 'std']).reset_index()
    except Exception as e:
        print(f"计算统计数据时出错: {e}")
        print("请检查您的CSV文件列名是否正确 (例如 'AA', '浓度/uM' 和特征列)。")
        return

    # 展平多级索引的列名
    df_stats_flat = df_stats.copy()
    new_cols = []
    for col in df_stats.columns:
        if col[0] in id_cols:
            new_cols.append(col[0])  # 保留 id_cols
        else:
            new_cols.append(f"{col[0]}_{col[1]}")  # 合并 (feature, stat)
    df_stats_flat.columns = new_cols

    # --- 3. 保存Excel数据 ---
    excel_filename = os.path.join(output_dir, 'summary_stats.xlsx')
    try:
        with pd.ExcelWriter(excel_filename) as writer:
            df_stats_flat.to_excel(writer, sheet_name='Summary_Stats (Mean_Std)', index=False)
            df_full_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
        print(f"成功将统计数据保存到: {excel_filename}")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")

    # --- 4. 循环绘图 ---
    unique_aas = df_full_raw['AA'].unique()
    print(f"开始为 {len(unique_aas)} 种氨基酸生成图像...")

    for aa in unique_aas:
        print(f"  正在处理: {aa} ...")

        df_aa_stats = df_stats_flat[df_stats_flat['AA'] == aa]
        df_aa_raw = df_full_raw[df_full_raw['AA'] == aa]
        safe_aa_name = "".join([c if c.isalnum() else "_" for c in aa])

        # --- Part 1: 数据图 (Errorbar, 8个子图) ---
        print(f"    生成 Part 1: Data (Errorbar)...")
        try:
            fig1, axes1 = plt.subplots(nrows=4, ncols=2, figsize=(22, 28),
                                       facecolor=STYLE_CONFIG['figure_facecolor'],
                                       constrained_layout=True)
            fig1.suptitle(f'AA: {aa} - Data (Mean ± 1 SD)', fontsize=26,
                          fontweight='bold', y=1.03)
            for i, feature in enumerate(feature_cols):
                plot_part1_data_errorbar(df_aa_stats, aa, feature, axes1.flatten()[i])
            for j in range(i + 1, len(axes1.flatten())):
                fig1.delaxes(axes1.flatten()[j])

            plot_filename1 = os.path.join(output_dir, f'AA_{safe_aa_name}_Data.png')
            plt.savefig(plot_filename1)
            plt.close(fig1)
        except Exception as e:
            print(f"    ERROR plotting Part 1 ({aa}): {e}")
            if 'fig1' in locals(): plt.close(fig1)

        # --- Part 2: 总图 (Intensity & Shift) ---
        print(f"    生成 Part 2: Intensity Summary...")
        plot_part2_summary(df_aa_stats, aa, feature_cols, 'intensity', output_dir)

        print(f"    生成 Part 2: Shift Summary...")
        plot_part2_summary(df_aa_stats, aa, feature_cols, 'shift', output_dir)

        # --- Part 3: 拟合图 (Scatter + Fit, 8个子图) ---
        print(f"    生成 Part 3: Fit (Scatter)...")
        try:
            fig3, axes3 = plt.subplots(nrows=4, ncols=2, figsize=(22, 28),
                                       facecolor=STYLE_CONFIG['figure_facecolor'],
                                       constrained_layout=True)
            fig3.suptitle(f'AA: {aa} - Scatter & Linear Fit', fontsize=26,
                          fontweight='bold', y=1.03)
            for i, feature in enumerate(feature_cols):
                plot_part3_fit(df_aa_raw, aa, feature, axes3.flatten()[i])
            for j in range(i + 1, len(axes3.flatten())):
                fig3.delaxes(axes3.flatten()[j])

            plot_filename3 = os.path.join(output_dir, f'AA_{safe_aa_name}_Fit.png')
            plt.savefig(plot_filename3)
            plt.close(fig3)
        except Exception as e:
            print(f"    ERROR plotting Part 3 ({aa}): {e}")
            if 'fig3' in locals(): plt.close(fig3)

    print("\n--- 全部处理完毕 ---")


def main():
    """
    主函数：使用Tkinter获取路径并调用绘图函数
    """
    print("--- 启动氨基酸绘图脚本 (带文件选择) ---")

    # 隐藏主窗口
    root = tk.Tk()
    root.withdraw()

    # 1. 选择训练文件
    train_path = filedialog.askopenfilename(
        title="请选择 训练 (Train) 数据文件 (例如: train_data.csv)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not train_path:
        print("操作取消：未选择训练文件。")
        return

    # 2. 选择测试文件
    test_path = filedialog.askopenfilename(
        title="请选择 测试 (Test) 数据文件 (例如: test_data.csv)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not test_path:
        print("操作取消：未选择测试文件。")
        return

    # 3. 选择保存目录
    output_dir = filedialog.askdirectory(
        title="请选择一个文件夹用于 保存 所有生成的图像和Excel文件"
    )
    if not output_dir:
        print("操作取消：未选择保存文件夹。")
        return

    print("\n路径选择完毕:")
    print(f"  训练文件: {train_path}")
    print(f"  测试文件: {test_path}")
    print(f"  保存目录: {output_dir}\n")

    # 4. 执行绘图
    run_analysis(train_path, test_path, output_dir)


# --- 运行主程序 ---
if __name__ == "__main__":
    if not check_dependencies():
        try:
            input("按 Enter 键退出...")
        except EOFError:
            pass
        sys.exit()  # 缺少库，退出

    main()

    print("\n脚本执行完毕。")
    try:
        input("按 Enter 键退出...")  # 防止窗口自动关闭
    except EOFError:
        pass