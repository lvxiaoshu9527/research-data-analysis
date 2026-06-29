import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D


# ==========================================
# 0. 全局视觉配置 (PPT/Paper 标准)
# ==========================================
def set_pub_style():
    sns.set_style("white")
    sns.set_context("talk")
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'SimHei', 'Microsoft YaHei'],
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'figure.figsize': (8, 6),
        'axes.grid': False,
        'image.cmap': 'RdBu_r',
        'axes.unicode_minus': False,
        'axes.edgecolor': 'black'  # 显式保留边框
    })


set_pub_style()


class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==========================================
# 1. 物理化学性质数据库
# ==========================================
class AAProperties:
    def __init__(self):
        self.pka_data = {
            'Ala': [2.34, 9.69, None], 'Arg': [2.17, 9.04, 12.48],
            'Asn': [2.02, 8.80, None], 'Asp': [1.88, 9.60, 3.65],
            'Cys': [1.96, 10.28, 8.18], 'Gln': [2.17, 9.13, None],
            'Glu': [2.19, 9.67, 4.25], 'Gly': [2.34, 9.60, None],
            'His': [1.82, 9.17, 6.00], 'Ile': [2.36, 9.68, None],
            'Leu': [2.36, 9.60, None], 'Lys': [2.18, 8.95, 10.53],
            'Met': [2.28, 9.21, None], 'Phe': [1.83, 9.13, None],
            'Pro': [1.99, 10.60, None], 'Ser': [2.21, 9.15, None],
            'Thr': [2.09, 9.10, None], 'Trp': [2.38, 9.39, None],
            'Tyr': [2.20, 9.11, 10.07], 'Val': [2.32, 9.62, None]
        }
        self.phys_data = {
            'Ala': [89.1, 6.00, 1.8, 67], 'Arg': [174.2, 10.76, -4.5, 148],
            'Asn': [132.1, 5.41, -3.5, 96], 'Asp': [133.1, 2.77, -3.5, 91],
            'Cys': [121.2, 5.07, 2.5, 86], 'Gln': [146.2, 5.65, -3.5, 114],
            'Glu': [147.1, 3.22, -3.5, 109], 'Gly': [75.1, 5.97, -0.4, 48],
            'His': [155.2, 7.59, -3.2, 118], 'Ile': [131.2, 6.02, 4.5, 124],
            'Leu': [131.2, 5.98, 3.8, 124], 'Lys': [146.2, 9.74, -3.9, 135],
            'Met': [149.2, 5.74, 1.9, 124], 'Phe': [165.2, 5.48, 2.8, 135],
            'Pro': [115.1, 6.30, -1.6, 90], 'Ser': [105.1, 5.68, -0.8, 73],
            'Thr': [119.1, 5.60, -0.7, 93], 'Trp': [204.2, 5.89, -0.9, 163],
            'Tyr': [181.2, 5.66, -1.3, 141], 'Val': [117.1, 5.96, 4.2, 105]
        }

    def calculate_net_charge(self, aa, ph=4.0):
        clean = aa.replace('L-', '')
        if clean not in self.pka_data: return 0.0
        pkas = self.pka_data[clean]
        c = 1.0 / (1.0 + 10 ** (ph - pkas[1]))
        c -= 1.0 / (1.0 + 10 ** (pkas[0] - ph))
        if pkas[2]:
            if clean in ['Asp', 'Glu', 'Tyr', 'Cys']:
                c -= 1.0 / (1.0 + 10 ** (pkas[2] - ph))
            elif clean in ['Arg', 'Lys', 'His']:
                c += 1.0 / (1.0 + 10 ** (ph - pkas[2]))
        return c

    def get_features_df(self, aa_list, ph=4.0):
        data = []
        for aa in aa_list:
            clean = aa.replace('L-', '')
            if clean in self.phys_data:
                p = self.phys_data[clean]
                data.append({
                    'AA': aa,
                    'MW': p[0], 'pI': p[1], 'Hydropathy': p[2], 'VDW_Volume': p[3],
                    f'Net_Charge_pH{ph}': self.calculate_net_charge(clean, ph)
                })
        return pd.DataFrame(data)


# ==========================================
# 2. 数据处理与清洗 (恢复原逻辑：多文件读取 -> 合并 -> 生成A/B表)
# ==========================================
def process_data(file_paths):
    print(">>> [Data] 正在读取并合并数据...")
    dfs = []
    for f in file_paths:
        try:
            print(f"    Reading: {os.path.basename(f)}")
            d = pd.read_csv(f)
            if 'AA' in d.columns:
                d['AA'] = d['AA'].replace({'L-Lle': 'L-Ile', 'Lle': 'L-Ile'})
            dfs.append(d)
        except Exception as e:
            print(f"Error: {e}")

    if not dfs: return None, None, None
    raw = pd.concat(dfs, ignore_index=True)

    ignore = ['AA', '浓度/uM', 'Date', 'Sample_ID', 'Replicate']
    feats = [c for c in raw.columns if c not in ignore and np.issubdtype(raw[c].dtype, np.number)]
    print(f"    特征列: {len(feats)} 个")
    print(f"    总样本数: {len(raw)}")

    # Dataset A (Samples): Mean + Std (样本级，用于 ML)
    print("    -> 生成 Dataset A (Mean+Std, 样本级, 用于训练)...")
    df_A = raw.groupby(['AA', '浓度/uM'])[feats].agg(['mean', 'std'])
    df_A.columns = [f"{c}_{s}" for c, s in df_A.columns]
    df_A = df_A.reset_index()

    # Dataset B (Fingerprints): Mean of Means (指纹级, 用于机理可视化)
    print("    -> 生成 Dataset B (Mean, 指纹级, 用于绘图)...")
    mean_cols = [c for c in df_A.columns if '_mean' in c]
    df_B = df_A.groupby('AA')[mean_cols].mean().reset_index()
    df_B.columns = [c.replace('_mean', '') for c in df_B.columns]

    return df_A, df_B, feats


# ==========================================
# 3. 绘图辅助函数：5分类颜色 & 堆叠图
# ==========================================
def get_aa_category_color(aa_name):
    clean = aa_name.replace('L-', '')
    if clean in ['Phe', 'Tyr', 'Trp']:
        return '#9467bd', 'Aromatic ($\pi$-stacking)'
    elif clean in ['Arg', 'Lys', 'His']:
        return '#d62728', 'Positive (+)'
    elif clean in ['Asp', 'Glu']:
        return '#1f77b4', 'Negative (-)'
    elif clean in ['Ser', 'Thr', 'Asn', 'Gln', 'Cys', 'Gly', 'Pro']:
        return '#2ca02c', 'Polar (Uncharged)'
    else:
        return '#7f7f7f', 'Non-polar (Hydrophobic)'


def get_legend_handles():
    legend_order = ['Positive (+)', 'Negative (-)', 'Aromatic ($\pi$-stacking)', 'Polar (Uncharged)',
                    'Non-polar (Hydrophobic)']
    legend_colors = {'Positive (+)': '#d62728', 'Negative (-)': '#1f77b4',
                     'Aromatic ($\pi$-stacking)': '#9467bd', 'Polar (Uncharged)': '#2ca02c',
                     'Non-polar (Hydrophobic)': '#7f7f7f'}
    return [
        Line2D([0], [0], marker='o', color='w', label=cat,
               markerfacecolor=legend_colors[cat], markersize=11)
        for cat in legend_order
    ]


def plot_aggregated_importance(perm_df, save_dir):
    """Chart 4b: 通道聚合堆叠图 (恢复你想要的功能)"""
    print(">>> [4b] 绘制堆叠特征重要性图 (Aggregated Stacked)...")

    def get_base_name(feat_name):
        for suffix in ['_mean', '_std', '_Mean', '_Std']:
            if feat_name.endswith(suffix):
                return feat_name.rsplit(suffix, 1)[0], suffix.replace('_', '').lower()
        return feat_name, 'other'

    parsed = perm_df['Feature'].apply(get_base_name)
    perm_df['Base_Feature'] = [p[0] for p in parsed]
    perm_df['Type'] = [p[1] for p in parsed]

    df_pivot = perm_df.pivot_table(index='Base_Feature', columns='Type', values='Importance', aggfunc='sum').fillna(0)
    if 'mean' not in df_pivot.columns: df_pivot['mean'] = 0
    if 'std' not in df_pivot.columns: df_pivot['std'] = 0

    df_pivot['Total'] = df_pivot['mean'] + df_pivot['std']
    df_pivot = df_pivot.sort_values(by='Total', ascending=True)

    # 导出
    csv_path = os.path.join(save_dir, 'Feature_Importance_Aggregated.csv')
    df_pivot.sort_values(by='Total', ascending=False).to_csv(csv_path)

    # 绘图
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(df_pivot))
    names = df_pivot.index.values
    means = df_pivot['mean'].values
    stds = df_pivot['std'].values

    # 蓝色系 vs 橙色系
    p1 = plt.barh(y_pos, means, align='center', color='#4c72b0', alpha=0.9, height=0.7, label='Intensity (Mean)')
    p2 = plt.barh(y_pos, stds, left=means, align='center', color='#dd8452', alpha=0.9, height=0.7,
                  label='Stability (Std)')

    # 标注 Total 值
    for i, t in enumerate(df_pivot['Total'].values):
        if t > max(df_pivot['Total'].values) * 0.05:
            plt.text(t + max(df_pivot['Total']) * 0.01, i, f'{t:.4f}', va='center', fontsize=12, color='#333333')

    plt.yticks(y_pos, names)
    plt.xlabel("Total Importance (Split 7:3)")
    plt.title("Channel-Level Importance Ranking")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4b_Feature_Importance_Stacked.png'), dpi=300)
    plt.close()


# ==========================================
# 4. 核心分析流程
# ==========================================
def run_analysis(df_A, df_B, raw_feats, save_dir):
    # 准备机理数据
    aa_tool = AAProperties()
    df_props = aa_tool.get_features_df(df_B['AA'].unique())
    df_mech = pd.merge(df_B, df_props, on='AA')
    valid_feats = [f for f in raw_feats if f in df_mech.columns]

    # --- Chart 1: 机理热图 ---
    print(">>> [1/4] 绘制机理热图...")
    prop_cols = ['MW', 'pI', 'Hydropathy', 'VDW_Volume', 'Net_Charge_pH4.0']
    prop_labels = ['MW', 'pI', 'Hydrophobicity', 'Volume', 'Net Charge']

    corr_matrix = df_mech[valid_feats + prop_cols].corr(method='spearman').loc[valid_feats, prop_cols]
    corr_matrix.columns = prop_labels
    corr_matrix.to_csv(os.path.join(save_dir, '1_Correlation_Matrix.csv'))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=1.5, linecolor='white', square=True, annot_kws={"size": 14})
    plt.title("Mechanism Correlation Map (Spearman)", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_Mechanism_Heatmap.png'), dpi=300)
    plt.close()

    # --- Chart 2: 自动相关性挖掘 (5分类颜色) ---
    print(">>> [2/4] 绘制相关性散点图 (5分类)...")
    df_mech.to_csv(os.path.join(save_dir, '2_Scatter_Plot_Data.csv'), index=False)

    df_mech['Color'] = df_mech['AA'].apply(lambda x: get_aa_category_color(x)[0])
    custom_legend = get_legend_handles()

    target_config = [
        ('Net_Charge_pH4.0', 'Net charge (pH 4.0)', 'Charge'),
        ('pI', 'Isoelectric Point (pI)', 'pI'),
        ('VDW_Volume', 'Van der Waals Volume', 'Volume'),
        ('Hydropathy', 'Hydropathy Index', 'Hydrophobicity')
    ]

    for prop, label_name, file_suffix in target_config:
        corrs = df_mech[valid_feats].corrwith(df_mech[prop], method='spearman')
        best_sensor = corrs.abs().idxmax()
        best_r = corrs[best_sensor]

        plt.figure(figsize=(8, 6.5))
        sns.regplot(data=df_mech, x=prop, y=best_sensor, scatter=False, color='#333333',
                    line_kws={"linewidth": 2, "linestyle": "--", "alpha": 0.4})
        plt.scatter(x=df_mech[prop], y=df_mech[best_sensor], c=df_mech['Color'],
                    s=180, alpha=0.9, edgecolor='white', linewidth=1.5, zorder=10)
        plt.legend(handles=custom_legend, loc='best', frameon=True, framealpha=0.9,
                   edgecolor='white', fontsize=12, handletextpad=0.2, labelspacing=0.4)
        plt.title(f"Channel {best_sensor} vs. {label_name}", pad=15, fontweight='bold', fontsize=20)
        plt.xlabel(label_name)
        plt.ylabel("Normalized Sensor Response")
        plt.text(0.04, 0.96, f"Spearman $R$ = {best_r:.2f}", transform=plt.gca().transAxes,
                 fontsize=18, color='#444444', va='top', fontweight='medium')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'2_Scatter_{file_suffix}.png'), dpi=300)
        plt.close()

    # --- Chart 3: 聚类树状图 ---
    print(">>> [3/4] 绘制聚类树状图...")
    df_B.to_csv(os.path.join(save_dir, '3_Clustering_Data.csv'), index=False)
    X = StandardScaler().fit_transform(df_B[valid_feats])
    Z = linkage(X, method='ward')
    plt.figure(figsize=(12, 7))
    dendrogram(Z, labels=[x.replace('L-', '') for x in df_B['AA']], leaf_font_size=18, leaf_rotation=0,
               color_threshold=0.7 * max(Z[:, 2]), above_threshold_color='#555555')
    ax = plt.gca();
    for line in ax.lines: line.set_linewidth(2.5)
    plt.title("Hierarchical Clustering of Sensors", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_Clustering.png'), dpi=300)
    plt.close()

    # --- Chart 4: 特征重要性 (按照要求：Split 7/3, Train on 70%, Score on 30%) ---
    print(">>> [4/4] 计算特征重要性 (70% Train / 30% Test)...")

    feat_cols_A = [c for c in df_A.columns if c not in ['AA', '浓度/uM']]
    X = df_A[feat_cols_A]
    y = df_A['AA']

    # 核心修改：一次性随机切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 训练 RF
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # 在 Test Set 上计算 Permutation Importance (泛化重要性)
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    imp_mean = result.importances_mean
    imp_std = result.importances_std

    # Chart 4a: Top 15 条形图
    sorted_idx = imp_mean.argsort()[-15:]  # Top 15
    plt.figure(figsize=(10, 8))
    bar_colors = ['#4c72b0' if '_mean' in feat_cols_A[i] else '#dd8452' for i in sorted_idx]
    plt.barh(range(len(sorted_idx)), imp_mean[sorted_idx], xerr=imp_std[sorted_idx],
             color=bar_colors, capsize=5, align='center', alpha=0.9, height=0.7)
    plt.yticks(range(len(sorted_idx)), [feat_cols_A[i] for i in sorted_idx])
    plt.xlabel("Importance (Permutation on 30% Test Set)")
    plt.title("Top 15 Discriminative Features (Split 7:3)", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4a_Feature_Importance_Top15.png'), dpi=300)
    plt.close()

    # Chart 4b: 聚合堆叠图 (调用辅助函数)
    perm_df = pd.DataFrame({'Feature': feat_cols_A, 'Importance': imp_mean, 'Std': imp_std})
    csv_imp = os.path.join(save_dir, 'Feature_Importance_Split73.csv')
    perm_df.sort_values('Importance', ascending=False).to_csv(csv_imp, index=False)

    plot_aggregated_importance(perm_df, save_dir)


# ==========================================
# 主程序
# ==========================================
def main():
    root = tk.Tk()
    root.withdraw()
    print("=== DNA-SWCNT 机理与特征全能分析 (Merge & Split 7:3) ===")

    # 1. 选择文件 (支持多选)
    print("请选择数据文件 (Train & Test csv)...")
    files = filedialog.askopenfilenames(title="选择数据文件 (支持多选 Train/Test)", filetypes=[("CSV", "*.csv")])
    if not files:
        print("未选择文件，退出。")
        return

    # 2. 选择保存路径
    print("请选择结果保存目录...")
    save_dir = filedialog.askdirectory(title="选择保存目录")
    if not save_dir:
        print("未选择保存目录，退出。")
        return

    # 开启日志
    sys.stdout = DualLogger(os.path.join(save_dir, 'analysis_log.txt'))

    # 3. 处理数据
    df_A, df_B, raw_feats = process_data(list(files))

    if df_A is not None:
        # 4. 运行分析
        run_analysis(df_A, df_B, raw_feats, save_dir)
        print("\n=== 分析全部完成 ===")
        print(f"所有结果已保存至: {save_dir}")
        messagebox.showinfo("成功", f"绘图及分析完成！\n文件已保存至: {save_dir}")

    sys.stdout = sys.stdout.terminal
    root.destroy()


if __name__ == "__main__":
    main()