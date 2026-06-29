import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# 确保tkinter可用，这是GUI弹窗的基础
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk, Checkbutton, IntVar, Radiobutton, StringVar, Toplevel
except ImportError:
    print("Error: Tkinter module not found. GUI cannot be created.")
    sys.exit(1)

# 确保UMAP可用
try:
    import umap.umap_ as umap
except ImportError:
    print("Error: UMAP library not installed. Please install it: pip install umap-learn")
    sys.exit(1)

# --- 全局字体设置 ---
mpl.rcParams['font.family'] = ['Arial', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


# --- GUI配置弹窗类 (英文化) ---
class AnalysisConfigDialog(Toplevel):
    def __init__(self, parent, amino_acids):
        super().__init__(parent)
        self.transient(parent)
        self.title("Analysis Configuration")
        self.parent = parent
        self.result = None
        self.amino_acids = amino_acids
        self.method_var = StringVar(value="PCA")
        self.acid_vars = {acid: IntVar(value=1) for acid in self.amino_acids}
        self.create_widgets()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_window(self)

    def create_widgets(self):
        acid_frame = ttk.LabelFrame(self, text="1. Select Amino Acids for Analysis", padding=(10, 5))
        acid_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        canvas = tk.Canvas(acid_frame, borderwidth=0, width=300, height=300)
        scrollbar = ttk.Scrollbar(acid_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        for acid in self.amino_acids:
            cb = ttk.Checkbutton(scrollable_frame, text=acid, variable=self.acid_vars[acid])
            cb.pack(anchor='w', padx=5)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        btn_frame = ttk.Frame(acid_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Select All", command=self._select_all).pack(side='left', expand=True, padx=5)
        ttk.Button(btn_frame, text="Deselect All", command=self._deselect_all).pack(side='left', expand=True, padx=5)
        method_frame = ttk.LabelFrame(self, text="2. Select Reduction Method", padding=(10, 5))
        method_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ttk.Radiobutton(method_frame, text="PCA", variable=self.method_var, value="PCA").pack(anchor='w')
        ttk.Radiobutton(method_frame, text="t-SNE", variable=self.method_var, value="t-SNE").pack(anchor='w')
        ttk.Radiobutton(method_frame, text="UMAP", variable=self.method_var, value="UMAP").pack(anchor='w')
        control_frame = ttk.Frame(self)
        control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        ttk.Button(control_frame, text="Cancel", command=self._on_cancel).pack(side="right", padx=5)
        ttk.Button(control_frame, text="Start Analysis", command=self._on_ok).pack(side="right")

    def _select_all(self):
        for var in self.acid_vars.values(): var.set(1)

    def _deselect_all(self):
        for var in self.acid_vars.values(): var.set(0)

    def _on_ok(self):
        selected_acids = [acid for acid, var in self.acid_vars.items() if var.get() == 1]
        if not selected_acids:
            messagebox.showwarning("Selection Error", "Please select at least one amino acid.", parent=self)
            return
        if len(selected_acids) < 2:
            messagebox.showwarning("Selection Info",
                                   "Warning: Selecting only one amino acid is not ideal for comparison.", parent=self)
        self.result = {"selected_acids": selected_acids, "method": self.method_var.get()}
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


# --- 数据保存函数 ---
def save_scatter_data(data, labels_aa, filename, output_dir):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        df_to_save = pd.DataFrame({'Component_1': data[:, 0], 'Component_2': data[:, 1], 'Amino_Acid': labels_aa})
        df_to_save.to_csv(file_path, index=False, sep='\t')
        print(f"Scatter plot data saved to: {file_path}")


def save_pca_components(pca_components, feature_names, explained_variance_ratio, filename, output_dir):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        pca_loadings_df = pd.DataFrame(pca_components.T, index=feature_names,
                                       columns=[f'PC{i + 1}' for i in range(pca_components.shape[0])])
        explained_variance_row = pd.Series(explained_variance_ratio,
                                           index=[f'PC{i + 1}' for i in range(pca_components.shape[0])],
                                           name='Explained Variance Ratio')
        pca_loadings_df = pd.concat([pca_loadings_df, pd.DataFrame(explained_variance_row).T])
        pca_loadings_df.to_csv(file_path, index=True)
        print(f"PCA components saved to: {file_path}")


# --- 可视化函数 (精修版) ---
def plot_clusters_final(X, labels_aa, title_prefix, filename_suffix, output_dir, params_text="",
                        explained_variance_ratio=None):
    fig_height = 8
    fig_width = fig_height * 1.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)

    unique_amino_acids = sorted(np.unique(labels_aa))

    if len(unique_amino_acids) <= 10:
        palette = sns.color_palette("tab10", len(unique_amino_acids))
    elif len(unique_amino_acids) <= 20:
        palette = sns.color_palette("tab20", len(unique_amino_acids))
    else:
        palette = sns.color_palette("husl", len(unique_amino_acids))
    aa_to_color = {aa: palette[i] for i, aa in enumerate(unique_amino_acids)}

    for aa in unique_amino_acids:
        indices = (labels_aa == aa)
        if np.sum(indices) > 0:
            # 绘制散点
            ax.scatter(X[indices, 0], X[indices, 1], c=[aa_to_color[aa]], alpha=0.7, s=50, label=aa)

            # 绘制置信椭圆
            if np.sum(indices) > 1:  # 至少需要两个点来计算协方差
                points = X[indices, :]
                mean = np.mean(points, axis=0)
                cov = np.cov(points, rowvar=False)

                # 计算特征值和特征向量
                eig_vals, eig_vecs = np.linalg.eigh(cov)
                order = eig_vals.argsort()[::-1]
                eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]

                # 获取椭圆参数
                angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
                # 95% 置信区间对应于2自由度的卡方分布的5.991
                width, height = 2 * np.sqrt(5.991 * eig_vals)

                ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                                  facecolor=aa_to_color[aa], alpha=0.2)
                ax.add_patch(ellipse)

    # 调整坐标轴
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    range_x, range_y = x_max - x_min, y_max - y_min
    max_range = max(range_x, range_y) if range_x > 0 and range_y > 0 else 0.1
    padding = max_range * 0.1
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    ax.set_xlim(center_x - (max_range / 2) - padding, center_x + (max_range / 2) + padding)
    ax.set_ylim(center_y - (max_range / 2) - padding, center_y + (max_range / 2) + padding)
    ax.set_aspect('equal', adjustable='box')

    # 设置标题和轴标签 (英文化)
    if title_prefix == 'PCA' and explained_variance_ratio is not None:
        ax.set_xlabel(f"Principal Component 1 ({explained_variance_ratio[0] * 100:.1f}%)", fontsize=18)
        ax.set_ylabel(f"Principal Component 2 ({explained_variance_ratio[1] * 100:.1f}%)", fontsize=18)
    else:
        ax.set_xlabel(f'{title_prefix} Component 1', fontsize=18)
        ax.set_ylabel(f'{title_prefix} Component 2', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    full_title = f'{title_prefix} Dimensionality Reduction Analysis\n{params_text}' if params_text else f'{title_prefix} Dimensionality Reduction Analysis'
    plt.title(full_title, fontsize=20, fontweight='bold')

    # 设置图例 (英文化)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=14, title="Amino Acid", title_fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_filepath_png = os.path.join(output_dir, f'{filename_suffix}_plot.png')
        plt.savefig(plot_filepath_png, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_filepath_png}")
    plt.close(fig)


# --- 主应用类 ---
class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dimensionality Reduction Tool")
        self.root.geometry("300x150")
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(expand=True, fill="both")
        self.start_button = ttk.Button(self.main_frame, text="Start Analysis", command=self.run_analysis_workflow)
        self.start_button.pack(expand=True)

    def run_analysis_workflow(self):
        try:
            print("Popping up file selection dialog...")
            input_file_path = filedialog.askopenfilename(parent=self.root, title="Select Training Data CSV File",
                                                         filetypes=[("CSV files", "*.csv")])
            if not input_file_path:
                print("File selection cancelled.")
                return

            print(f"Reading file: {input_file_path}")
            df_original = pd.read_csv(input_file_path)
            print("File read successfully.")

            concentration_column = '浓度/uM'
            if 'AA' not in df_original.columns or concentration_column not in df_original.columns:
                messagebox.showerror("Column Missing",
                                     f"CSV file must contain 'AA' and '{concentration_column}' columns.",
                                     parent=self.root)
                return

            unique_amino_acids = sorted(df_original['AA'].unique())
            print("Key columns check passed.")

            print("Waiting for user to configure analysis...")
            config_dialog = AnalysisConfigDialog(self.root, unique_amino_acids)
            config = config_dialog.result
            if not config:
                print("Configuration cancelled.")
                return

            selected_acids = config["selected_acids"]
            selected_method = config["method"]
            print(f"Selected Amino Acids: {selected_acids}")
            print(f"Selected Method: {selected_method}")

            print("Preparing data for analysis...")
            df_filtered = df_original[df_original['AA'].isin(selected_acids)].copy()
            chiralities = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']
            feature_columns_flat = [f'{c}_{t}' for c in chiralities for t in ['intensity', 'shift']]

            if any(col not in df_filtered.columns for col in feature_columns_flat):
                messagebox.showerror("Column Missing", "CSV file is missing some feature columns.", parent=self.root)
                return

            X = df_filtered[feature_columns_flat].fillna(0).values
            y_labels_aa = df_filtered['AA'].values
            print(f"Data prepared. Total data points for analysis: {len(X)}")

            save_base_dir = filedialog.askdirectory(parent=self.root, title="Select Directory to Save Results")
            if not save_base_dir:
                print("Save directory selection cancelled.")
                return

            print("\n===== Starting Dimensionality Reduction and Visualization =====")

            scaler = StandardScaler()
            print("--- Scaling data using StandardScaler ---")
            X_scaled = scaler.fit_transform(X)

            output_dir = os.path.join(save_base_dir, selected_method)
            os.makedirs(output_dir, exist_ok=True)

            if selected_method == 'PCA':
                print("Executing PCA analysis...")
                pca = PCA(n_components=2, random_state=42)
                X_reduced = pca.fit_transform(X_scaled)
                params_text = f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.4f}"
                filename_suffix = 'pca_analysis'
                plot_clusters_final(X_reduced, y_labels_aa, 'PCA', filename_suffix, output_dir, params_text,
                                    pca.explained_variance_ratio_)
                save_pca_components(pca.components_, feature_columns_flat, pca.explained_variance_ratio_,
                                    f'{filename_suffix}_components.csv', output_dir)
                save_scatter_data(X_reduced, y_labels_aa, f'{filename_suffix}_data.tsv', output_dir)

            elif selected_method == 't-SNE':
                print("Executing t-SNE analysis (this may take a moment)...")
                tsne_params = {'perplexity': 30, 'max_iter': 1000, 'learning_rate': 'auto'}
                tsne = TSNE(n_components=2, random_state=42, **tsne_params)
                X_reduced = tsne.fit_transform(X_scaled)
                params_text = ', '.join([f"{k}: {v}" for k, v in tsne_params.items()])
                filename_suffix = f'tsne_p{tsne_params["perplexity"]}'
                plot_clusters_final(X_reduced, y_labels_aa, 't-SNE', filename_suffix, output_dir, params_text)
                save_scatter_data(X_reduced, y_labels_aa, f'{filename_suffix}_data.tsv', output_dir)

            elif selected_method == 'UMAP':
                print("Executing UMAP analysis...")
                umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'}
                reducer = umap.UMAP(n_components=2, random_state=42, **umap_params)
                X_reduced = reducer.fit_transform(X_scaled)
                params_text = f"n_neighbors: {umap_params['n_neighbors']}, min_dist: {umap_params['min_dist']}"
                filename_suffix = f'umap_nn{umap_params["n_neighbors"]}'
                plot_clusters_final(X_reduced, y_labels_aa, 'UMAP', filename_suffix, output_dir, params_text)
                save_scatter_data(X_reduced, y_labels_aa, f'{filename_suffix}_data.tsv', output_dir)

            print("\n===== Analysis and Visualization Complete =====")
            messagebox.showinfo("Success", "Analysis is complete! Results have been saved to the selected directory.",
                                parent=self.root)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            messagebox.showerror("Critical Error", f"An unexpected error occurred during the process:\n\n{e}",
                                 parent=self.root)


if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()
    print("Application closed.")
