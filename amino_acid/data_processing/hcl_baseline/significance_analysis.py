import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Button, Frame, Text, Scrollbar, END
import os
import matplotlib as mpl
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import traceback
import sys
import threading
import joblib
from scipy.stats import f, shapiro, levene, linregress, t
from statsmodels.stats.multitest import multipletests
from scipy.signal import savgol_filter

# --- 全局设置 ---
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 300
})


class AnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("合并学习与预测工具 (Pro Max: 带源数据导出)")
        master.geometry("950x700")

        self.control_frame = Frame(master, padx=10, pady=10)
        self.control_frame.pack(fill=tk.X)
        self.btn_frame = Frame(self.control_frame);
        self.btn_frame.pack(pady=5)

        self.start_button = Button(self.btn_frame, text="开始建模分析", font=('Arial', 12, 'bold'),
                                   command=self.start_analysis_thread, width=20)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.baseline_btn = Button(self.btn_frame, text="生成基线报告 + 源数据表", font=('Arial', 12, 'bold'),
                                   command=self.start_baseline_analysis_thread, width=30, fg="white", bg="#006400")
        self.baseline_btn.pack(side=tk.LEFT, padx=10)

        self.log_frame = Frame(master, padx=10, pady=10)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = Text(self.log_frame, wrap=tk.WORD, state='disabled', font=('Courier New', 10))
        self.scrollbar = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.log("工具已就绪。本次更新：")
        self.log("1. 自动生成 Fig1, Fig2, Fig3 的 CSV 源数据表，方便 Origin/Prism 重绘。")
        self.log("2. 包含具体的斜率值、F-Ratio 值、拟合截距等详细参数。")

    def log(self, message):
        def append_log():
            if not self.master.winfo_exists(): return
            self.log_text.config(state='normal')
            self.log_text.insert(END, str(message) + "\n")
            self.log_text.see(END)
            self.log_text.config(state='disabled')

        self.master.after(0, append_log)

    def start_analysis_thread(self):
        threading.Thread(target=self.run_analysis_wrapper, daemon=True).start()

    def start_baseline_analysis_thread(self):
        threading.Thread(target=self.run_baseline_analysis_wrapper, daemon=True).start()

    def run_analysis_wrapper(self):
        try:
            self.run_main_analysis()
        except Exception as e:
            self._handle_error(e)

    def run_baseline_analysis_wrapper(self):
        try:
            self.run_baseline_analysis()
        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, e):
        self.log(f"Error: {e}\n{traceback.format_exc()}")
        messagebox.showerror("Error", f"{e}")

    # =================================================================================
    #                               基线分析主逻辑
    # =================================================================================
    def run_baseline_analysis(self):
        train_path, test_path, save_dir = self._ask_files_and_dir()
        if not all([train_path, test_path, save_dir]): return

        self.log("加载数据...")
        train_df = self._load_file(train_path)
        test_df = self._load_file(test_path)

        meta_cols = ['AA', '浓度/uM', 'Group_ID', 'group_id']
        feature_cols = [c for c in train_df.columns if
                        c not in meta_cols and pd.api.types.is_numeric_dtype(train_df[c])]

        baseline_dir = os.path.join(save_dir, "Publication_Source_Data")
        os.makedirs(baseline_dir, exist_ok=True)

        # 1. 数据处理
        self.log("处理数据 (平滑校正)...")
        raw_combined = pd.concat([train_df, test_df], ignore_index=True)
        corrected_df = self.correct_interleaved_data(train_df, test_df, feature_cols)

        # 2. 计算统计量
        self.log("计算统计指标...")
        stats_results = self.calculate_publication_stats(corrected_df, feature_cols)
        raw_stats_results = self.calculate_publication_stats(raw_combined, feature_cols)

        # --- 3. 关键步骤：导出源数据表格 (方便重画) ---
        self.log("正在导出 Fig 1-3 的源数据 CSV...")
        self.export_source_data_for_plots(corrected_df, stats_results, feature_cols, baseline_dir)

        # 4. 生成图表
        self.log("生成可视化图表...")
        self.plot_spaghetti_fit(corrected_df, feature_cols, stats_results, baseline_dir)
        self.plot_slope_boxplot(corrected_df, feature_cols, baseline_dir)
        self.plot_f_ratio_log(stats_results, baseline_dir)
        self.plot_raw_vs_smooth_comparison(raw_combined, corrected_df, feature_cols, baseline_dir)

        # 5. 生成报告
        self.export_publication_report(stats_results, raw_stats_results, baseline_dir)

        self.log(f"完成！数据和图表已保存至: {baseline_dir}")
        messagebox.showinfo("成功",
                            f"报告与源数据表已生成！\n\n文件夹: {baseline_dir}\n包含: Fig1_SourceData.csv 等文件")

    def export_source_data_for_plots(self, df, stats_df, features, save_dir):
        """专门生成用于画图的 CSV 表格"""

        # --- Fig 1 Source Data: 浓度-响应曲线 ---
        # 包含每个 AA 在每个浓度点的均值
        grouped = df.groupby(['AA', '浓度/uM'])[features].mean().reset_index()
        # 为了方便画图，我们把 HCl 的拟合参数也合进去不太容易，所以单独存曲线数据
        grouped.to_csv(os.path.join(save_dir, "Fig1_SourceData_Curves.csv"), index=False)
        self.log("-> 已导出 Fig1 数据: Fig1_SourceData_Curves.csv")

        # --- Fig 2 Source Data: 斜率分布 ---
        # 需要重新计算每个个体的斜率并列表
        slope_list = []
        for feat in features:
            # HCl Slope
            hcl_sub = grouped[grouped['AA'] == 'HCl']
            hcl_slope = 0
            if len(hcl_sub) > 1:
                hcl_slope = linregress(hcl_sub['浓度/uM'], hcl_sub[feat]).slope
            slope_list.append({'Feature': feat, 'AA': 'HCl', 'Group': 'HCl', 'Slope': hcl_slope})

            # AA Slopes
            for aa in grouped['AA'].unique():
                if aa == 'HCl': continue
                sub = grouped[grouped['AA'] == aa]
                if len(sub) > 1:
                    s = linregress(sub['浓度/uM'], sub[feat]).slope
                    slope_list.append({'Feature': feat, 'AA': aa, 'Group': 'Amino Acids', 'Slope': s})

        pd.DataFrame(slope_list).to_csv(os.path.join(save_dir, "Fig2_SourceData_IndividualSlopes.csv"), index=False)
        self.log("-> 已导出 Fig2 数据: Fig2_SourceData_IndividualSlopes.csv")

        # --- Fig 3 Source Data: F-Ratio 与统计汇总 ---
        # 直接使用 stats_df，包含了 F-Ratio, P值, HCl斜率, 截距, CI 等
        stats_df.to_csv(os.path.join(save_dir, "Fig3_SourceData_Statistics.csv"), index=False)
        self.log("-> 已导出 Fig3 数据: Fig3_SourceData_Statistics.csv")

    def calculate_publication_stats(self, df, features):
        """计算包含 CI, p_adj, Intercept 的完整统计表"""
        grouped = df.groupby(['AA', '浓度/uM'])[features].mean().reset_index()
        stats = []
        p_vals_f, p_vals_slope = [], []

        for feat in features:
            hcl_sub = grouped[grouped['AA'] == 'HCl']
            other_sub = grouped[grouped['AA'] != 'HCl']

            var_hcl = np.var(hcl_sub[feat], ddof=1)
            var_other = np.var(other_sub[feat], ddof=1)
            f_ratio = var_other / var_hcl if var_hcl > 0 else np.inf
            p_f = 1 - f.cdf(f_ratio, len(other_sub) - 1, len(hcl_sub) - 1)
            p_vals_f.append(p_f)

            try:
                _, p_lev = levene(hcl_sub[feat], other_sub[feat], center='median')
            except:
                p_lev = 1.0

            if len(hcl_sub) > 1:
                res = linregress(hcl_sub['浓度/uM'], hcl_sub[feat])
                slope, intercept, stderr, p_slope = res.slope, res.intercept, res.stderr, res.pvalue
                t_crit = t.ppf(0.975, df=len(hcl_sub) - 2)
                ci_low, ci_high = slope - t_crit * stderr, slope + t_crit * stderr
            else:
                slope, intercept, stderr, p_slope, ci_low, ci_high = 0, 0, 0, 1.0, 0, 0
            p_vals_slope.append(p_slope)

            stats.append({
                'Feature': feat, 'F_Ratio': f_ratio, 'F_P_Raw': p_f, 'Levene_P': p_lev,
                'HCl_Slope': slope, 'HCl_Intercept': intercept,  # 增加截距，方便画拟合线
                'HCl_Slope_SE': stderr, 'HCl_Slope_P_Raw': p_slope,
                'HCl_Slope_CI_Lower': ci_low, 'HCl_Slope_CI_Upper': ci_high
            })

        _, p_f_adj, _, _ = multipletests(p_vals_f, method='holm')
        _, p_slope_adj, _, _ = multipletests(p_vals_slope, method='holm')
        for i, row in enumerate(stats):
            row['F_P_Adj'] = p_f_adj[i]
            row['HCl_Slope_P_Adj'] = p_slope_adj[i]

        return pd.DataFrame(stats)

    # --- 绘图函数保持不变，但为了完整性包含在此 ---
    def plot_spaghetti_fit(self, df, features, stats_df, save_dir):
        grouped = df.groupby(['AA', '浓度/uM'])[features].mean().reset_index()
        n = len(features);
        cols = 2;
        rows = (n + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
        axes = axes.flatten()
        for i, feat in enumerate(features):
            ax = axes[i]
            stat_row = stats_df[stats_df['Feature'] == feat].iloc[0]
            for aa in grouped['AA'].unique():
                if aa == 'HCl': continue
                sub = grouped[grouped['AA'] == aa].sort_values('浓度/uM')
                y_show = savgol_filter(sub[feat], min(5, len(sub)), 2) if len(sub) > 3 else sub[feat]
                ax.plot(sub['浓度/uM'], y_show, color='grey', alpha=0.2, lw=0.8)
            hcl = grouped[grouped['AA'] == 'HCl'].sort_values('浓度/uM')
            ax.scatter(hcl['浓度/uM'], hcl[feat], color='#D32F2F', s=40, zorder=5, label='HCl Raw')
            x_fit = np.linspace(hcl['浓度/uM'].min(), hcl['浓度/uM'].max(), 100)
            slope = stat_row['HCl_Slope']
            intercept = stat_row['HCl_Intercept']  # 使用准确截距
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color='#D32F2F', lw=2.5, label=f'HCl Fit', zorder=6)
            ci_low = stat_row['HCl_Slope_CI_Lower']
            ci_high = stat_row['HCl_Slope_CI_Upper']
            # CI计算稍微复杂，这里用 simplified visual representation
            # 实际上重画时可以用 stats_df 里的 slope CI 直接标注
            y_low = ci_low * x_fit + (hcl[feat].mean() - ci_low * hcl['浓度/uM'].mean())
            y_high = ci_high * x_fit + (hcl[feat].mean() - ci_high * hcl['浓度/uM'].mean())
            ax.fill_between(x_fit, y_low, y_high, color='#D32F2F', alpha=0.15)
            ax.set_title(f"{feat}", fontweight='bold');
            ax.set_xlabel("Concentration (uM)")
            if i == 0: ax.legend(loc='lower right')
        plt.tight_layout();
        plt.savefig(os.path.join(save_dir, "Fig1_Spaghetti_Fit_Plot.png"), dpi=300);
        plt.close()

    def plot_slope_boxplot(self, df, features, save_dir):
        grouped = df.groupby(['AA', '浓度/uM'])[features].mean().reset_index()
        slope_data = []
        for feat in features:
            hcl_sub = grouped[grouped['AA'] == 'HCl']
            slope_hcl = linregress(hcl_sub['浓度/uM'], hcl_sub[feat]).slope if len(hcl_sub) > 1 else 0
            for aa in grouped['AA'].unique():
                if aa == 'HCl': continue
                sub = grouped[grouped['AA'] == aa]
                if len(sub) > 1: slope_data.append(
                    {'Feature': feat, 'Type': 'Amino Acids', 'Slope': linregress(sub['浓度/uM'], sub[feat]).slope})
            slope_data.append({'Feature': feat, 'Type': 'HCl', 'Slope': slope_hcl})
        plot_df = pd.DataFrame(slope_data)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Feature', y='Slope', data=plot_df[plot_df['Type'] == 'Amino Acids'], color='lightgrey',
                    width=0.5, showfliers=False)
        sns.stripplot(x='Feature', y='Slope', data=plot_df[plot_df['Type'] == 'Amino Acids'], color='grey', alpha=0.4,
                      size=3)
        sns.stripplot(x='Feature', y='Slope', data=plot_df[plot_df['Type'] == 'HCl'], color='red', size=10, marker='D',
                      jitter=False)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.xticks(rotation=45, ha='right');
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig2_Slope_Distribution.png"), dpi=300);
        plt.close()

    def plot_f_ratio_log(self, stats_df, save_dir):
        plt.figure(figsize=(10, 6))
        df_sorted = stats_df.sort_values('F_Ratio', ascending=False)
        sns.barplot(x='Feature', y='F_Ratio', data=df_sorted, palette='viridis')
        plt.yscale('log')
        plt.axhline(2.43, color='red', linestyle='--', lw=2)
        plt.axhline(100, color='orange', linestyle=':', lw=2)
        plt.xticks(rotation=45, ha='right');
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig3_F_Ratio_Log.png"), dpi=300);
        plt.close()

    def plot_raw_vs_smooth_comparison(self, raw_df, smooth_df, features, save_dir):
        raw_grouped = raw_df.groupby(['AA', '浓度/uM'])[features].mean().reset_index()
        smooth_grouped = smooth_df.groupby(['AA', '浓度/uM'])[features].mean().reset_index()
        feat = features[0]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, df, title in zip(axes, [raw_grouped, smooth_grouped], ["Raw", "Smoothed"]):
            for aa in df['AA'].unique():
                if aa == 'HCl': continue
                sub = df[df['AA'] == aa].sort_values('浓度/uM')
                ax.plot(sub['浓度/uM'], sub[feat], color='grey', alpha=0.3)
            hcl = df[df['AA'] == 'HCl'].sort_values('浓度/uM')
            ax.plot(hcl['浓度/uM'], hcl[feat], color='red', marker='o', lw=2, label='HCl')
            ax.set_title(f"{feat} - {title}")
        plt.tight_layout();
        plt.savefig(os.path.join(save_dir, "FigS1_Raw_vs_Smoothed.png"), dpi=300);
        plt.close()

    def export_publication_report(self, stats_df, raw_stats_df, save_dir):
        path = os.path.join(save_dir, "Publication_Results_Text.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=== Results Data ===\n\n")
            f.write(stats_df.round(5).to_string())

    def correct_interleaved_data(self, train_df, test_df, feature_cols):
        train_corrected = train_df.copy()
        combined_raw = pd.concat([train_df, test_df], ignore_index=True)
        aa_means = combined_raw.groupby(['AA', '浓度/uM'])[feature_cols].mean()
        for aa in combined_raw['AA'].unique():
            if aa not in train_df['AA'].values: continue
            train_concs = sorted(train_df[train_df['AA'] == aa]['浓度/uM'].unique())
            for col in feature_cols:
                diffs = []
                for c_train in train_concs:
                    c_prev, c_next = c_train - 10, c_train + 10
                    if (aa, c_prev) in aa_means.index and (aa, c_next) in aa_means.index:
                        est = (aa_means.loc[(aa, c_prev), col] + aa_means.loc[(aa, c_next), col]) / 2
                        diffs.append(aa_means.loc[(aa, c_train), col] - est)
                if diffs:
                    train_corrected.loc[train_corrected['AA'] == aa, col] -= np.mean(diffs)
        return pd.concat([train_corrected, test_df], ignore_index=True)

    def _ask_files_and_dir(self):
        t = filedialog.askopenfilename(title="选择Train数据");
        if not t: return None, None, None
        p = filedialog.askopenfilename(title="选择Test数据");
        if not p: return None, None, None
        d = filedialog.askdirectory(title="选择保存目录");
        if not d: return None, None, None
        return t, p, d

    def _load_file(self, path):
        return pd.read_excel(path) if path.endswith(('.xls', '.xlsx')) else pd.read_csv(path)

    def run_main_analysis(self):
        pass


if __name__ == '__main__':
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()