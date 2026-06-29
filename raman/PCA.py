"""
PCA Analysis Script for Amino Acid Sensing Data
================================================
功能：
  - 弹出文件选择框，加载 Excel 或 CSV 文件
  - 自动识别第一列为样本名，第二列为分组标签，其余列为特征
  - 数据标准化 → PCA 降维 → Score Plot 可视化
  - 绘制 95% / 68% 置信椭圆（透明填充）
  - 轴标签标注 PC1/PC2 贡献率
  - 自动弹出保存对话框（TIFF / PDF，300 DPI）

依赖：pandas, scikit-learn, matplotlib, scipy, openpyxl, tkinter
安装：pip install pandas scikit-learn matplotlib scipy openpyxl
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# ── GUI（tkinter）────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

matplotlib.use("TkAgg")   # 确保 tkinter 与 matplotlib 使用同一后端


# ────────────────────────────────────────────────────────────────────────────
# 1. 文件选择
# ────────────────────────────────────────────────────────────────────────────
def select_input_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath = filedialog.askopenfilename(
        title="选择数据文件（Excel 或 CSV）",
        filetypes=[
            ("Excel 文件", "*.xlsx *.xls *.xlsm"),
            ("CSV 文件",   "*.csv"),
            ("所有文件",   "*.*"),
        ],
    )
    root.destroy()
    if not filepath:
        print("未选择文件，程序退出。")
        sys.exit(0)
    return filepath


# ────────────────────────────────────────────────────────────────────────────
# 2. 数据读取与解析
# ────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str):
    """
    返回:
      sample_names : list[str]   第一列，样本名
      groups       : list[str]   第二列，分组标签
      features     : np.ndarray  其余列，特征矩阵 (n_samples × n_features)
      feature_names: list[str]   特征列名
    """
    if filepath.lower().endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(filepath, header=0)
    else:
        # 自动检测编码与分隔符
        try:
            df = pd.read_csv(filepath, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, sep=None, engine="python", encoding="gbk")

    if df.shape[1] < 3:
        raise ValueError("数据至少需要 3 列：样本名 | 分组 | 特征…")

    sample_names  = df.iloc[:, 0].astype(str).tolist()
    groups        = df.iloc[:, 1].astype(str).tolist()
    feature_df    = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

    # 删除全 NaN 列，填充剩余 NaN 为列均值
    feature_df.dropna(axis=1, how="all", inplace=True)
    feature_df.fillna(feature_df.mean(), inplace=True)

    feature_names = feature_df.columns.tolist()
    features      = feature_df.values

    print(f"✅ 读取成功：{df.shape[0]} 个样本，{len(feature_names)} 个特征，"
          f"{len(set(groups))} 个分组（{sorted(set(groups))}）")
    return sample_names, groups, features, feature_names


# ────────────────────────────────────────────────────────────────────────────
# 3. 标准化 + PCA
# ────────────────────────────────────────────────────────────────────────────
def run_pca(features: np.ndarray, n_components: int = 2):
    scaler      = StandardScaler()
    X_scaled    = scaler.fit_transform(features)
    pca         = PCA(n_components=n_components)
    scores      = pca.fit_transform(X_scaled)
    var_ratio   = pca.explained_variance_ratio_ * 100   # 百分比
    return scores, var_ratio, pca, X_scaled


# ────────────────────────────────────────────────────────────────────────────
# 4. 置信椭圆绘制
# ────────────────────────────────────────────────────────────────────────────
def confidence_ellipse(x, y, ax, n_std: float = 2.0,
                       facecolor="none", **kwargs):
    """
    n_std=2.0 ≈ 95% 置信区间（二维正态）
    n_std=1.0 ≈ 68% 置信区间
    """
    if len(x) < 3:
        return None
    cov  = np.cov(x, y)
    mean = np.array([np.mean(x), np.mean(y)])

    # 协方差矩阵特征分解 → 椭圆参数
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        facecolor=facecolor, **kwargs
    )
    return ax.add_patch(ellipse)


# ────────────────────────────────────────────────────────────────────────────
# 5. 绘图
# ────────────────────────────────────────────────────────────────────────────
def plot_pca(scores, groups, var_ratio, ellipse_ci="95%"):
    """
    ellipse_ci: "95%" → n_std=2.0 ;  "68%" → n_std=1.0
    """
    n_std = 2.0 if ellipse_ci == "95%" else 1.0

    unique_groups = sorted(set(groups))
    n_groups      = len(unique_groups)

    # ── 专业科研配色 ──────────────────────────────────────────────────────
    # 优先使用 Seaborn deep 调色板；若未安装则回退至 Set1
    try:
        import seaborn as sns
        palette = sns.color_palette("deep", n_colors=n_groups)
    except ImportError:
        cmap    = plt.get_cmap("Set1")
        palette = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]

    color_map = {g: palette[i] for i, g in enumerate(unique_groups)}

    # ── 画布 ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    # ── 分组绘制散点 + 椭圆 ───────────────────────────────────────────────
    legend_handles = []
    for grp in unique_groups:
        mask  = np.array([g == grp for g in groups])
        x_grp = scores[mask, 0]
        y_grp = scores[mask, 1]
        color = color_map[grp]

        # 散点
        ax.scatter(
            x_grp, y_grp,
            color=color, s=80, zorder=3,
            edgecolors="white", linewidths=0.6,
            label=grp
        )

        # 置信椭圆（边框 + 半透明填充）
        if len(x_grp) >= 3:
            confidence_ellipse(
                x_grp, y_grp, ax, n_std=n_std,
                facecolor=(*color[:3], 0.15),   # RGBA：透明填充
                edgecolor=color, linewidth=1.8,
                linestyle="--", zorder=2
            )

        # 图例句柄
        patch = mpatches.Patch(color=color, label=grp)
        legend_handles.append(patch)

    # ── 零线 ──────────────────────────────────────────────────────────────
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    # ── 轴标签（含贡献率）────────────────────────────────────────────────
    ax.set_xlabel(
        f"PC1 ({var_ratio[0]:.1f}%)",
        fontsize=13, fontweight="bold", labelpad=8
    )
    ax.set_ylabel(
        f"PC2 ({var_ratio[1]:.1f}%)",
        fontsize=13, fontweight="bold", labelpad=8
    )
    ax.set_title(
        f"PCA Score Plot  —  {ellipse_ci} Confidence Ellipse",
        fontsize=14, fontweight="bold", pad=12
    )

    # ── 图例（图外右上角）────────────────────────────────────────────────
    ax.legend(
        handles=legend_handles,
        title="Group",
        title_fontsize=10,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    # ── 边框美化 ──────────────────────────────────────────────────────────
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#AAAAAA")

    ax.tick_params(axis="both", labelsize=10, color="#AAAAAA")

    plt.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────
# 6. 保存对话框
# ────────────────────────────────────────────────────────────────────────────
def save_figure(fig):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    save_path = filedialog.asksaveasfilename(
        title="保存 PCA 图片",
        defaultextension=".tiff",
        filetypes=[
            ("TIFF 高分辨率", "*.tiff *.tif"),
            ("PDF 矢量图",    "*.pdf"),
            ("PNG 图片",      "*.png"),
        ],
    )
    root.destroy()

    if save_path:
        fmt = save_path.rsplit(".", 1)[-1].lower()
        fig.savefig(save_path, dpi=300, format=fmt, bbox_inches="tight")
        print(f"✅ 图片已保存至：{save_path}")
        messagebox.showinfo("保存成功", f"图片已保存至：\n{save_path}")
    else:
        print("⚠️  未保存文件。")


# ────────────────────────────────────────────────────────────────────────────
# 7. 置信椭圆选择弹窗
# ────────────────────────────────────────────────────────────────────────────
def ask_ellipse_ci() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    choice = simpledialog.askstring(
        "置信椭圆",
        "请输入置信区间（输入 95 或 68）：",
        initialvalue="95",
        parent=root,
    )
    root.destroy()
    if choice and choice.strip() == "68":
        return "68%"
    return "95%"


# ────────────────────────────────────────────────────────────────────────────
# 主程序
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  氨基酸传感数据 PCA 分析脚本")
    print("=" * 55)

    # Step 1：选择文件
    filepath = select_input_file()
    print(f"📂 已选择：{filepath}")

    # Step 2：加载数据
    try:
        sample_names, groups, features, feature_names = load_data(filepath)
    except Exception as e:
        messagebox.showerror("数据读取失败", str(e))
        sys.exit(1)

    # Step 3：PCA
    scores, var_ratio, pca_model, X_scaled = run_pca(features)
    print(f"📊 PC1 贡献率：{var_ratio[0]:.1f}%，PC2 贡献率：{var_ratio[1]:.1f}%")

    # Step 4：选择置信区间
    ellipse_ci = ask_ellipse_ci()
    print(f"🔵 置信椭圆：{ellipse_ci}")

    # Step 5：绘图
    fig = plot_pca(scores, groups, var_ratio, ellipse_ci=ellipse_ci)
    plt.show(block=False)

    # Step 6：保存
    save_figure(fig)

    print("✅ 完成！")


if __name__ == "__main__":
    main()