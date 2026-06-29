import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Checkbutton, Button, BooleanVar, Frame, Radiobutton, StringVar, \
    Text, Scrollbar, END, Label, Entry
import os
import matplotlib as mpl
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import traceback
import sys
import threading
import joblib

# --- Matplotlib全局设置 ---
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


class AnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("光谱数据分析工具 v16.0 (可靠性验证版)")
        master.geometry("700x600")

        # --- GUI 组件 ---
        self.control_frame = Frame(master, padx=10, pady=10)
        self.control_frame.pack(fill=tk.X)

        Label(self.control_frame, text="重复验证次数:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(10, 0))
        self.repeats_var = StringVar(value="5")
        self.repeats_entry = Entry(self.control_frame, textvariable=self.repeats_var, width=5)
        self.repeats_entry.pack(side=tk.LEFT, padx=(5, 20))

        self.start_button = Button(self.control_frame, text="开始分析", font=('Arial', 12, 'bold'),
                                   command=self.start_analysis_thread)
        self.start_button.pack(side=tk.RIGHT, pady=10, padx=20)

        self.log_frame = Frame(master, padx=10, pady=10)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_frame.grid_rowconfigure(0, weight=1);
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = Text(self.log_frame, wrap=tk.WORD, state='disabled', font=('Courier New', 10))
        self.scrollbar = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew");
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.log("欢迎使用模型可靠性验证工具！")
        self.log("此版本将多次重复训练-测试流程以评估模型性能的稳定性。")
        self.log("请在上方输入框指定重复次数，然后点击 '开始分析'。")

    def log(self, message):
        def append_log():
            if not self.master.winfo_exists(): return
            self.log_text.config(state='normal')
            self.log_text.insert(END, str(message) + "\n")
            self.log_text.see(END)
            self.log_text.config(state='disabled')

        self.master.after(0, append_log)

    def start_analysis_thread(self):
        try:
            n_repeats = int(self.repeats_var.get())
            if n_repeats < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "重复验证次数必须是一个大于0的整数。")
            return

        self.start_button.config(state="disabled", text="分析中...")
        self.log("\n" + "=" * 70)
        self.log(f"新的可靠性验证任务已开始，将重复 {n_repeats} 次。")
        self.log("注意：这可能会非常耗时！")

        analysis_thread = threading.Thread(target=self.run_analysis_wrapper, args=(n_repeats,))
        analysis_thread.daemon = True
        analysis_thread.start()

    def run_analysis_wrapper(self, n_repeats):
        try:
            self.run_repeated_analysis(n_repeats)
        except Exception as e:
            error_details = traceback.format_exc()
            self.log(f"!!! 发生严重错误 !!!\n错误类型: {type(e).__name__}\n错误信息: {e}")
            self.log(f"详细追溯信息:\n{error_details}")
            messagebox.showerror("程序遇到意外错误", f"发生未处理的错误: {e}\n\n请查看主窗口日志获取详细信息。")
        finally:
            self.master.after(0, lambda: self.start_button.config(state="normal", text="开始分析"))
            self.log("所有分析任务结束。")
            self.log("=" * 70 + "\n")

    def run_repeated_analysis(self, n_repeats):
        # --- 步骤 1: 数据加载与选择 (只执行一次) ---
        self.log("--- 阶段一：数据加载与选择 ---")
        file_path = filedialog.askopenfilename(parent=self.master, title="选择数据文件",
                                               filetypes=[("CSV files", "*.csv")])
        if not file_path: self.log("操作取消。"); return

        save_base_dir = filedialog.askdirectory(parent=self.master, title="选择结果保存目录")

        df = pd.read_csv(file_path)
        label_column = self.select_label_column(df.columns.tolist())
        if not label_column: self.log("操作取消。"); return

        all_features = [col for col in df.columns if col != label_column]
        selected_features = self.select_features(all_features)
        if not selected_features: self.log("操作取消。"); return

        columns_to_keep = selected_features + [label_column]
        df_processed = df[columns_to_keep]

        all_amino_acids = df_processed[label_column].dropna().unique().tolist()
        all_amino_acids.sort()
        selected_acids = self.select_amino_acids(all_amino_acids)
        if not selected_acids: self.log("操作取消。"); return
        df_filtered = df_processed[df_processed[label_column].isin(selected_acids)].copy()

        # --- 步骤 2: 重复验证循环 ---
        self.log("\n--- 阶段二：开始重复验证循环 ---")

        all_runs_scores = {'dt': [], 'rf': [], 'svm': [], 'xgb': [], 'lr': [], 'mlp': []}

        for i in range(n_repeats):
            run_seed = i
            self.log(f"\n--- 第 {i + 1}/{n_repeats} 轮验证 (随机种子: {run_seed}) ---")

            # 数据预处理
            X = df_filtered.drop(label_column, axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
            y_raw = df_filtered[label_column]
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            # 【确认修改】按照2/3训练，1/3测试划分数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=run_seed,
                                                                stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)

            # 模型定义
            models_to_tune = {'dt': DecisionTreeClassifier(random_state=run_seed),
                              'rf': RandomForestClassifier(random_state=run_seed),
                              'svm': SVC(probability=True, random_state=run_seed, kernel='linear'),
                              'xgb': XGBClassifier(random_state=run_seed, use_label_encoder=False,
                                                   eval_metric='mlogloss'),
                              'lr': LogisticRegression(random_state=run_seed, max_iter=1000),
                              'mlp': MLPClassifier(random_state=run_seed, max_iter=1000, early_stopping=True)}
            hp_space = {'dt': {'criterion': hp.choice('criterion', ['gini', 'entropy']),
                               'max_depth': hp.quniform('max_depth', 3, 20, 1),
                               'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1), },
                        'rf': {'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                               'max_depth': hp.quniform('max_depth', 5, 50, 1),
                               'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1), },
                        'svm': {'C': hp.loguniform('C', np.log(0.1), np.log(100))},
                        'xgb': {'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                                'max_depth': hp.quniform('max_depth', 3, 15, 1),
                                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                                'subsample': hp.uniform('subsample', 0.7, 1.0), },
                        'lr': {'C': hp.loguniform('C', np.log(0.01), np.log(100)),
                               'solver': hp.choice('solver', ['liblinear', 'saga']), }, 'mlp': {
                    'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
                    'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(0.1)),
                    'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.001), np.log(0.1)), }}

            # 超参数优化与评估
            for name, model in models_to_tune.items():
                self.log(f"  优化并评估 {name.upper()}...")

                def objective(params):
                    params_for_model = params.copy()
                    for key in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
                        if key in params_for_model: params_for_model[key] = int(params_for_model[key])
                    model.set_params(**params_for_model)
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
                    score = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='f1_macro', n_jobs=-1,
                                            error_score='raise').mean()
                    return {'loss': -score, 'status': STATUS_OK}

                trials = Trials()
                best_params_raw = fmin(fn=objective, space=hp_space[name], algo=tpe.suggest, max_evals=50,
                                       trials=trials, rstate=np.random.default_rng(run_seed), show_progressbar=False,
                                       verbose=False)
                best_params = space_eval(hp_space[name], best_params_raw)
                for key in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
                    if key in best_params: best_params[key] = int(best_params[key])
                final_model = models_to_tune[name].set_params(**best_params)
                final_model.fit(X_train_scaled, y_train)

                y_pred = final_model.predict(X_test_scaled)
                score = f1_score(y_test, y_pred, average='macro')
                all_runs_scores[name].append(score)
                self.log(f"    -> 本轮 F1 分数 (macro): {score:.4f}")

        # --- 步骤 3: 汇总结果与可视化 ---
        self.log("\n--- 阶段三：汇总所有验证结果 ---")
        results_df = pd.DataFrame(all_runs_scores)

        self.log("各模型在多次验证中的F1分数(macro)统计:")
        self.log(results_df.describe().to_string())

        self.log("\n正在生成性能稳定性箱线图...")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=results_df, ax=ax)
        ax.set_title(f'模型性能稳定性 ({n_repeats} 次重复验证)', fontsize=20)
        ax.set_ylabel('F1 分数 (macro)', fontsize=16)
        ax.set_xlabel('模型', fontsize=16)
        plt.tight_layout()
        plt.show()
        self.log("箱线图已生成。箱体越窄，代表模型性能越稳定可靠。")

    def write(self, text):
        if text.strip(): self.log(text.strip())

    def flush(self):
        pass

    def visualize_data_distribution(self, df, label_column):
        X = df.drop(label_column, axis=1).apply(pd.to_numeric, errors='coerce').fillna(0);
        y = df[label_column];
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2);
        X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(10, 8));
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, ax=ax, palette='viridis', legend='full')
        ax.set_title(f'PCA 可视化 (解释方差: {pca.explained_variance_ratio_.sum():.2%})', fontsize=16)
        ax.set_xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]:.1%})');
        ax.set_ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.legend(title=label_column, bbox_to_anchor=(1.05, 1), loc='upper left');
        plt.tight_layout();
        plt.show()

    def select_label_column(self, columns):
        dialog = Toplevel(self.master);
        dialog.title("请指定标签列 (Y)");
        dialog.geometry("400x450")
        selected_col = StringVar(value=columns[0] if columns else "");
        top_frame = Frame(dialog);
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(top_frame, text="你的哪一列是分类目标?", font=('Arial', 12, 'bold')).pack()
        mid_frame = Frame(dialog);
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(mid_frame);
        scrollbar = tk.Scrollbar(mid_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas);
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for col in columns: Radiobutton(scrollable_frame, text=col, variable=selected_col, value=col,
                                        font=('Arial', 12)).pack(anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        bottom_frame = Frame(dialog);
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        Button(bottom_frame, text="确认", command=dialog.destroy, font=('Arial', 12, 'bold')).pack()
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_col.get()

    def select_features(self, all_features):
        dialog = Toplevel(self.master);
        dialog.title("请选择特征列 (X)");
        dialog.geometry("400x500")
        selected_features_list = [];
        vars = {feature: BooleanVar(value=True) for feature in all_features}

        def confirm():
            nonlocal selected_features_list;
            selected_features_list = [feature for feature, var in vars.items() if var.get()]
            if not selected_features_list: messagebox.showerror("错误", "请至少选择一个特征列！", parent=dialog); return
            dialog.destroy()

        top_frame = Frame(dialog);
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        Button(top_frame, text="全选", command=lambda: [v.set(True) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                  expand=True, padx=5)
        Button(top_frame, text="全不选", command=lambda: [v.set(False) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                     expand=True,
                                                                                                     padx=5)
        mid_frame = Frame(dialog);
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(mid_frame);
        scrollbar = tk.Scrollbar(mid_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas);
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for feature in all_features: Checkbutton(scrollable_frame, text=feature, var=vars[feature],
                                                 font=('Arial', 12)).pack(anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        bottom_frame = Frame(dialog);
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        Button(bottom_frame, text="确认选择", command=confirm, font=('Arial', 12, 'bold')).pack()
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_features_list

    def select_amino_acids(self, all_labels):
        dialog = Toplevel(self.master);
        dialog.title("选择要训练的氨基酸");
        dialog.geometry("400x500")
        selected_labels_list = [];
        vars = {label: BooleanVar(value=True) for label in all_labels}

        def confirm():
            nonlocal selected_labels_list;
            selected_labels_list = [label for label, var in vars.items() if var.get()]
            if len(selected_labels_list) < 2: messagebox.showerror("错误", "请至少选择两种氨基酸！",
                                                                   parent=dialog); return
            dialog.destroy()

        top_frame = Frame(dialog);
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        Button(top_frame, text="全选", command=lambda: [v.set(True) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                  expand=True, padx=5)
        Button(top_frame, text="全不选", command=lambda: [v.set(False) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                     expand=True,
                                                                                                     padx=5)
        mid_frame = Frame(dialog);
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(mid_frame);
        scrollbar = tk.Scrollbar(mid_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas);
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for label in all_labels: Checkbutton(scrollable_frame, text=label, var=vars[label], font=('Arial', 12)).pack(
            anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        bottom_frame = Frame(dialog);
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        Button(bottom_frame, text="确认选择", command=confirm, font=('Arial', 12, 'bold')).pack()
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_labels_list


if __name__ == '__main__':
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

