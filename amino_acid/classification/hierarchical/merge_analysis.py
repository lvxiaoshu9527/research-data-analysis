import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Checkbutton, Button, BooleanVar, Frame, Radiobutton, StringVar, \
    Text, Scrollbar, END, Entry
import os
import matplotlib as mpl
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import traceback
import sys
import threading
import joblib

# --- Matplotlib全局设置 ---
try:
    mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
except:
    pass
mpl.rcParams['axes.unicode_minus'] = False


class AnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("合并混合打散学习工具 (基于原版修改)")
        master.geometry("750x600")

        self.control_frame = Frame(master, padx=10, pady=10)
        self.control_frame.pack(fill=tk.X)

        # --- 文件选择区域 (修改了标签文本以符合合并逻辑) ---
        self.file_frame = tk.LabelFrame(self.control_frame, text="数据文件选择 (将合并以下两个文件)", padx=5, pady=5)
        self.file_frame.pack(fill=tk.X, pady=5)

        tk.Label(self.file_frame, text="数据文件 A:").grid(row=0, column=0, sticky='w')
        self.train_path = StringVar()
        Entry(self.file_frame, textvariable=self.train_path, width=50).grid(row=0, column=1, padx=5)
        Button(self.file_frame, text="浏览", command=lambda: self.browse_file(self.train_path)).grid(row=0, column=2)

        tk.Label(self.file_frame, text="数据文件 B:").grid(row=1, column=0, sticky='w')
        self.test_path = StringVar()
        Entry(self.file_frame, textvariable=self.test_path, width=50).grid(row=1, column=1, padx=5)
        Button(self.file_frame, text="浏览", command=lambda: self.browse_file(self.test_path)).grid(row=1, column=2)

        tk.Label(self.file_frame, text="结果保存目录:").grid(row=2, column=0, sticky='w')
        self.save_dir = StringVar()
        Entry(self.file_frame, textvariable=self.save_dir, width=50).grid(row=2, column=1, padx=5)
        Button(self.file_frame, text="浏览", command=self.browse_dir).grid(row=2, column=2)

        # --- 模型选择区域 (保持不变) ---
        self.model_frame = tk.LabelFrame(self.control_frame, text="模型选择 (支持多选)", padx=5, pady=5)
        self.model_frame.pack(fill=tk.X, pady=5)

        self.models_vars = {
            "RandomForest": BooleanVar(value=True),
            "SVM": BooleanVar(value=True),
            "XGBoost": BooleanVar(value=True),
            "DecisionTree": BooleanVar(value=False),
            "LogisticRegression": BooleanVar(value=False),
            "MLP (Neural Net)": BooleanVar(value=False)
        }

        col = 0
        for name, var in self.models_vars.items():
            Checkbutton(self.model_frame, text=name, variable=var).grid(row=0, column=col, sticky='w', padx=10)
            col += 1

        # --- 功能选项 (保持不变) ---
        self.option_frame = tk.LabelFrame(self.control_frame, text="高级选项", padx=5, pady=5)
        self.option_frame.pack(fill=tk.X, pady=5)

        self.use_hyperopt = BooleanVar(value=False)
        Checkbutton(self.option_frame, text="启用贝叶斯超参数优化 (Hyperopt) - 耗时较长",
                    variable=self.use_hyperopt).pack(side=tk.LEFT)

        # --- 运行按钮 ---
        self.run_btn = Button(self.control_frame, text="开始合并、打散并分析", command=self.start_analysis, bg="green",
                              fg="white", font=("Arial", 12, "bold"))
        self.run_btn.pack(pady=10)

        # --- 日志区域 ---
        self.log_text = Text(master, height=15, width=90)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = Scrollbar(master, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def browse_file(self, path_var):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if filename:
            path_var.set(filename)

    def browse_dir(self):
        dirname = filedialog.askdirectory()
        if dirname:
            self.save_dir.set(dirname)

    def log(self, message):
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)

    def start_analysis(self):
        if not self.train_path.get() or not self.test_path.get():
            messagebox.showerror("错误", "请选择两个数据文件！")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.log("-" * 50)
        self.log("任务开始...")

        threading.Thread(target=self.run_analysis_thread, daemon=True).start()

    def run_analysis_thread(self):
        try:
            self.run_analysis()
        except Exception as e:
            self.log(f"发生错误:\n{traceback.format_exc()}")
            messagebox.showerror("运行错误", str(e))
        finally:
            self.run_btn.config(state=tk.NORMAL)
            self.log("任务结束。")

    def run_analysis(self):
        p1 = self.train_path.get()
        p2 = self.test_path.get()
        save_dir = self.save_dir.get()
        use_opt = self.use_hyperopt.get()

        # =========================================================================
        # 1. 修改核心：数据读取、合并与打散划分
        # =========================================================================
        self.log(f"读取文件 A: {os.path.basename(p1)}")
        df_a = pd.read_csv(p1)
        self.log(f"读取文件 B: {os.path.basename(p2)}")
        df_b = pd.read_csv(p2)

        self.log("正在合并数据...")
        df_full = pd.concat([df_a, df_b], axis=0, ignore_index=True)
        self.log(f"合并后总样本数: {len(df_full)}")

        # 假设第0列是 Label, 第2列往后是特征 (保留原代码的数据切片逻辑)
        # 原逻辑：train_df.iloc[:, 0] 是 y, train_df.iloc[:, 2:] 是 X
        try:
            y_all_raw = df_full.iloc[:, 0].values
            X_all_raw = df_full.iloc[:, 2:].values
            feature_names = df_full.columns[2:]
        except IndexError:
            self.log("数据格式错误：无法切片第0列和第2列。请检查CSV格式。")
            return

        # 标签编码
        le = LabelEncoder()
        y_all_enc = le.fit_transform(y_all_raw)
        self.log(f"类别检测: {le.classes_}")

        # === 核心修改：混合打散并切分 ===
        self.log("正在执行混合打散与切分 (Stratified Shuffle Split)...")
        # test_size=0.2 表示 20% 做测试，80% 做训练，stratify 保证类别比例一致
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_all_raw, y_all_enc,
            test_size=0.33,
            stratify=y_all_enc,
            random_state=42
        )
        self.log(f"训练集大小: {len(X_train_raw)}, 测试集大小: {len(X_test_raw)}")

        # 标准化 (必须在 split 之后)
        self.log("执行 StandardScaler 标准化...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # =========================================================================
        # 以下逻辑保持原版不变，直接使用准备好的变量：
        # X_train_scaled, X_test_scaled, y_train, y_test, feature_names, le
        # =========================================================================

        results_summary = []

        # 定义模型空间 (仅在启用 Hyperopt 时使用)
        search_spaces = {
            "RandomForest": {
                'n_estimators': hp.choice('n_estimators', range(50, 500)),
                'max_depth': hp.choice('max_depth', range(5, 30)),
                'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
                'criterion': hp.choice('criterion', ["gini", "entropy"])
            },
            "SVM": {
                'C': hp.loguniform('C', np.log(0.01), np.log(100)),
                'gamma': hp.loguniform('gamma', np.log(0.001), np.log(1.0)),
                'kernel': hp.choice('kernel', ['rbf', 'linear'])
            },
            "XGBoost": {
                'n_estimators': hp.choice('n_estimators', range(50, 300)),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                'max_depth': hp.choice('max_depth', range(3, 10))
            }
        }

        # 遍历选择的模型
        for name, var in self.models_vars.items():
            if not var.get():
                continue

            self.log(f"\n>>> 处理模型: {name} ...")

            model = None
            best_params = "Default"

            # 1. 获取模型实例
            if use_opt and name in search_spaces:
                self.log(f"   正在进行贝叶斯优化 ({name})...")

                def objective(params):
                    # 内部转换参数类型
                    if name == "RandomForest":
                        params['n_estimators'] = int(params['n_estimators'])
                        params['max_depth'] = int(params['max_depth'])
                    if name == "XGBoost":
                        params['n_estimators'] = int(params['n_estimators'])
                        params['max_depth'] = int(params['max_depth'])

                    clf = self.get_model_instance(name, params)
                    # 使用 3-Fold CV
                    score = cross_val_score(clf, X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
                    return {'loss': -score, 'status': STATUS_OK}

                trials = Trials()
                best = fmin(fn=objective, space=search_spaces[name], algo=tpe.suggest, max_evals=15, trials=trials)
                best_params = space_eval(search_spaces[name], best)

                # 转换整数参数
                if name == "RandomForest" or name == "XGBoost":
                    if 'n_estimators' in best_params: best_params['n_estimators'] = int(best_params['n_estimators'])
                    if 'max_depth' in best_params: best_params['max_depth'] = int(best_params['max_depth'])

                self.log(f"   最佳参数: {best_params}")
                model = self.get_model_instance(name, best_params)
            else:
                model = self.get_model_instance(name)

            if model is None: continue

            # 2. 训练
            model.fit(X_train_scaled, y_train)

            # 3. 预测
            y_pred = model.predict(X_test_scaled)

            # 4. 评估
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            self.log(f"   Accuracy: {acc:.4f}")
            self.log(f"   F1 Score: {f1:.4f}")

            results_summary.append({
                "Model": name,
                "Accuracy": acc,
                "F1_Score": f1,
                "Params": str(best_params)
            })

            # 5. 详细报告
            report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
            self.log("--- Classification Report ---")
            self.log(report)

            # 6. 绘图与保存 (如果选择了目录)
            if save_dir and os.path.exists(save_dir):
                # 混淆矩阵
                self.plot_confusion_matrix(y_test, y_pred, le.classes_, name, save_dir)
                # 特征重要性
                if name in ["RandomForest", "XGBoost", "DecisionTree"]:
                    self.display_feature_importance(model, feature_names, name, save_dir)

        # 汇总
        self.log("\n" + "=" * 30)
        self.log("所有模型汇总结果")
        self.log("=" * 30)
        df_res = pd.DataFrame(results_summary).sort_values(by="Accuracy", ascending=False)
        self.log(df_res.to_string(index=False))

        if save_dir:
            df_res.to_csv(os.path.join(save_dir, "comparison_summary.csv"), index=False)
            self.log("汇总表已保存。")

    def get_model_instance(self, name, params=None):
        # 如果没有传参数，使用默认参数 (与原代码保持一致)
        if params is None:
            if name == "RandomForest":
                return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            elif name == "SVM":
                return SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            elif name == "XGBoost":
                return XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False,
                                     eval_metric='mlogloss', random_state=42, n_jobs=-1)
            elif name == "DecisionTree":
                return DecisionTreeClassifier(random_state=42)
            elif name == "LogisticRegression":
                return LogisticRegression(max_iter=1000)
            elif name == "MLP (Neural Net)":
                return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else:
            # 使用传入的参数
            if name == "RandomForest":
                return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif name == "SVM":
                return SVC(**params, probability=True, random_state=42)
            elif name == "XGBoost":
                return XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss', random_state=42,
                                     n_jobs=-1)
            # 其他模型如果加了超参数搜索逻辑可在此扩展
        return None

    def plot_confusion_matrix(self, y_true, y_pred, classes, model_name, save_dir):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        # 计算百分比混淆矩阵
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f"{model_name} Confusion Matrix")
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_cm.png'))
        plt.close()

        # 也可以保存百分比版，如果需要
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=classes, yticklabels=classes)
        plt.title(f"{model_name} CM (Normalized)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_cm_norm.png'))
        plt.close()

    def display_feature_importance(self, model, columns, model_name, save_dir):
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

        if importances is not None:
            df_imp = pd.DataFrame({'feature': columns, 'importance': importances}).sort_values('importance',
                                                                                               ascending=False).head(20)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=df_imp)
            plt.title(f'{model_name} Top 20 Features')
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{model_name}_importance.png'))
            plt.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()