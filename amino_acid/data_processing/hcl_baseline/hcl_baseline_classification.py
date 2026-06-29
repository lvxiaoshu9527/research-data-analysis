# 请把本文件保存为你的主脚本（覆盖原文件前请先备份原始版本）
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
from scipy.stats import ttest_ind

# --- Matplotlib全局设置 ---
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


class AnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("合并学习与预测工具 (带CV汇总报告)")
        master.geometry("750x550")

        self.control_frame = Frame(master, padx=10, pady=10)
        self.control_frame.pack(fill=tk.X)
        self.start_button = Button(self.control_frame, text="开始分析", font=('Arial', 12, 'bold'),
                                   command=self.start_analysis_thread)
        self.start_button.pack(pady=10)

        self.log_frame = Frame(master, padx=10, pady=10)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = Text(self.log_frame, wrap=tk.WORD, state='disabled', font=('Courier New', 10))
        self.scrollbar = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.log("欢迎使用合并学习与预测工具！")
        self.log("此版本已实现分层分组抽样，并会导出包含交叉验证结果的最终汇总报告。")

    def log(self, message):
        def append_log():
            if not self.master.winfo_exists(): return
            self.log_text.config(state='normal')
            self.log_text.insert(END, str(message) + "\n")
            self.log_text.see(END)
            self.log_text.config(state='disabled')

        self.master.after(0, append_log)

    def start_analysis_thread(self):
        self.start_button.config(state="disabled", text="分析中...")
        self.log("\n" + "=" * 60)
        self.log("新的分析任务已开始...")
        analysis_thread = threading.Thread(target=self.run_analysis_wrapper)
        analysis_thread.daemon = True
        analysis_thread.start()

    def run_analysis_wrapper(self):
        try:
            self.run_main_analysis()
        except Exception as e:
            error_details = traceback.format_exc()
            self.log(f"!!! 发生严重错误 !!!\n错误类型: {type(e).__name__}\n错误信息: {e}")
            self.log(f"详细追溯信息:\n{error_details}")
            messagebox.showerror("程序遇到意外错误", f"发生未处理的错误: {e}\n\n请查看主窗口日志获取详细信息。")
        finally:
            self.master.after(0, lambda: self.start_button.config(state="normal", text="开始分析"))
            self.log("分析任务结束。")
            self.log("=" * 60 + "\n")

    def run_main_analysis(self):
        """核心分析流程"""
        # --- 步骤 1: 文件选择 ---
        self.log("步骤 1/11: 请选择 **基础训练** 数据文件...")
        train_path = filedialog.askopenfilename(parent=self.master, title="选择 **基础训练** 数据CSV文件",
                                                filetypes=[
                                                    ("CSV/Excel files", "*.csv *.xlsx *.xls"),
                                                    ("CSV files", "*.csv"),
                                                    ("Excel files", "*.xlsx *.xls")
                                                ]
                                                )
        if not train_path: self.log("操作取消。"); return

        self.log("步骤 2/11: 请选择 **新数据/预测** 数据文件...")
        predict_path = filedialog.askopenfilename(parent=self.master, title="选择 **新数据/预测** 数据CSV文件",
                                                  filetypes=[
                                                      ("CSV/Excel files", "*.csv *.xlsx *.xls"),
                                                      ("CSV files", "*.csv"),
                                                      ("Excel files", "*.xlsx *.xls")
                                                  ]
                                                  )
        if not predict_path: self.log("操作取消。"); return

        self.log("步骤 3/11: 请选择结果保存目录...")
        save_base_dir = filedialog.askdirectory(parent=self.master, title="选择结果保存目录")
        if not save_base_dir:
            self.log("操作取消。"); return

        self.log("步骤 4/11: 正在加载数据...")

        # 自动根据后缀判断读取方式
        def load_file(path):
            if path.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(path)
            else:
                return pd.read_csv(path)

        train_df = load_file(train_path)
        predict_df = load_file(predict_path)

        # --- 步骤 5: 关键策略选择与配置 ---
        self.log("步骤 5/11: (关键) 请选择处理数据的捆绑策略...")
        strategy = self.select_strategy()
        if not strategy: self.log("操作取消。"); return

        self.log("步骤 6/11: (关键) 请配置新数据用于训练的比例...")
        percentage_to_train = self.ask_for_incremental_percentage()
        if percentage_to_train is None: self.log("操作取消。"); return
        self.log(f"配置：将尝试使用 {percentage_to_train}% 的新数据组加入训练集。")

        # --- 步骤 7: 定义列角色 ---
        self.log("步骤 7/11: 请指定主要类别列 (预测目标Y)...")
        label_column = self.select_column_role(train_df.columns.tolist(), "请选择主要类别列 (如 品种/类别)")
        if not label_column: self.log("操作取消。"); return

        self.log("步骤 8/11: 请指定 **唯一的** 组内条件列 (定义平行样)...")
        condition_column = self.select_column_role(train_df.columns.tolist(), "请选择组内条件列 (如 浓度/批次)")
        if not condition_column: self.log("操作取消。"); return

        if not all(col in df.columns for col in [label_column, condition_column] for df in [train_df, predict_df]):
            messagebox.showerror("列缺失", f"您选择的 '{label_column}' 或 '{condition_column}' 并非在两个文件中都存在！")
            self.log("错误：关键列在两个文件中不匹配。");
            return

        self.log("步骤 9/11: 请选择特征列 (X)...")
        common_cols = list(set(train_df.columns) & set(predict_df.columns))
        cols_to_exclude = [label_column, condition_column]
        all_features = [col for col in common_cols if col not in cols_to_exclude]
        selected_features = self.select_features(all_features)
        if not selected_features: self.log("操作取消。"); return

        # --- 步骤 10: 根据策略构建最终训练/测试集 ---
        self.log(f"步骤 10/11: 正在重构数据流并应用 '{strategy}' 策略...")

        predict_df_copy = predict_df.copy()
        group_id_col = "__AUTO_GROUP_ID__"
        predict_df_copy[group_id_col] = predict_df_copy[label_column].astype(str) + "_" + predict_df_copy[
            condition_column].astype(str)

        if percentage_to_train == 0:
            new_data_for_training = pd.DataFrame()
            new_data_for_testing = predict_df_copy
        else:
            self.log("正在对新数据进行分层分组抽样...")
            unique_groups_df = predict_df_copy.drop_duplicates(subset=[group_id_col])
            group_ids = unique_groups_df[group_id_col]
            group_labels = unique_groups_df[label_column]

            stratify_param = group_labels if group_labels.value_counts().min() >= 2 else None
            if stratify_param is None:
                self.log("警告: 检测到至少一个类别中只有一个独立样本组。无法进行分层抽样，将回退到普通分组抽样。")

            test_size_prop = 1.0 - (percentage_to_train / 100.0)
            try:
                train_group_ids, test_group_ids = train_test_split(
                    group_ids, test_size=test_size_prop, random_state=42, stratify=stratify_param)
            except Exception as e:
                self.log(f"错误: 在数据划分时发生错误: {e}")
                messagebox.showerror("划分错误",
                                     f"数据划分失败: {e}\n\n请检查新数据文件，确保每个类别下有足够的独立组用于划分。")
                return

            new_data_for_training = predict_df_copy[predict_df_copy[group_id_col].isin(train_group_ids)]
            new_data_for_testing = predict_df_copy[predict_df_copy[group_id_col].isin(test_group_ids)]
            self.log(f"已按组进行划分: {len(train_group_ids)} 组进入训练, {len(test_group_ids)} 组进入测试。")

        raw_train_df = pd.concat([train_df, new_data_for_training], ignore_index=True)
        raw_test_df = new_data_for_testing

        if strategy == "均值聚合":
            self.log("聚合训练数据...")
            X_train, y_train_raw = self.aggregate_dataframe(raw_train_df, selected_features, label_column,
                                                            condition_column)
            self.log("聚合测试数据...")
            X_test, y_test_raw = self.aggregate_dataframe(raw_test_df, selected_features, label_column,
                                                          condition_column)
        else:  # 分组捆绑
            X_train = raw_train_df[selected_features]
            y_train_raw = raw_train_df[label_column]
            X_test = raw_test_df[selected_features]
            y_test_raw = raw_test_df[label_column]

        self.log(f"最终生效的训练集大小: {X_train.shape[0]} 行, {X_train.shape[1]} 列")
        self.log(f"最终生效的测试集大小: {X_test.shape[0]} 行, {X_test.shape[1]} 列")
        if X_test.empty:
            messagebox.showerror("错误", "最终测试集为空，无法继续分析。");
            return

        # --- 步骤 11: 数据预处理与模型训练 ---
        self.log("步骤 11/11: 数据预处理、模型训练与评估...")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_test = le.transform(y_test_raw)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- [修改点] 将模型评估逻辑移至新函数，并传入原始DataFrame与raw标签以便绘图/统计 ---
        # 注意：此处将 X_train_scaled, X_test_scaled 作为模型输入（保持训练一致性）
        # 同时传入 X_train (DataFrame), X_test (DataFrame), y_test_raw (原始标签字符串) 供后续可视化使用
        self.run_evaluation(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            y_test_raw,
            X_train,
            X_test,
            list(X_train.columns),
            le,
            save_base_dir
        )

    def run_evaluation(self, X_train, y_train, X_test, y_test, y_test_raw, X_train_df, X_test_df, feature_columns, le, save_base_dir):
        """执行所有模型的超参数优化、交叉验证、最终评估和报告生成。
        已修改以接收 y_test_raw 与 原始 DataFrame（用于统计图与 LDA 可视化）。
        """
        models_to_tune = {'dt': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                          'rf': RandomForestClassifier(random_state=42, class_weight='balanced'),
                          'svm': SVC(probability=True, random_state=42, kernel='linear', class_weight='balanced'),
                          'xgb': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
                          'lr': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                          'mlp': MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)}

        hp_space = {'dt': {'criterion': hp.choice('criterion', ['gini', 'entropy']),
                           'max_depth': hp.quniform('max_depth', 3, 20, 1)},
                    'rf': {'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                           'max_depth': hp.quniform('max_depth', 5, 50, 1)},
                    'svm': {'C': hp.loguniform('C', np.log(0.1), np.log(100))},
                    'xgb': {'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                            'max_depth': hp.quniform('max_depth', 3, 15, 1)},
                    'lr': {'C': hp.loguniform('C', np.log(0.01), np.log(100))},
                    'mlp': {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
                            'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(0.1))}}

        final_models = {}
        all_reports_data = {}
        summary_stats = []  # 用于存储最终汇总信息
        original_stdout = sys.stdout;
        sys.stdout = self

        for name, model in models_to_tune.items():
            self.log(f"--- 正在优化 {name.upper()} ---")

            def objective(params):
                p = {k: (int(v) if k in ['max_depth', 'n_estimators'] else v) for k, v in params.items()}
                model.set_params(**p)
                score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(4, shuffle=True, random_state=42),
                                        scoring='f1_macro').mean()
                return {'loss': -score, 'status': STATUS_OK}

            trials = Trials()
            best_params_raw = fmin(fn=objective, space=hp_space[name], algo=tpe.suggest, max_evals=50, trials=trials,
                                   rstate=np.random.default_rng(42), show_progressbar=False)
            best_params = space_eval(hp_space[name], best_params_raw)
            best_params = {k: (int(v) if k in ['max_depth', 'n_estimators'] else v) for k, v in best_params.items()}

            # --- 动态交叉验证以获取均值和标准差 ---
            min_class_count = np.min(np.bincount(y_train))
            n_splits_cv = min(5, min_class_count)
            cv_line = ""
            if n_splits_cv < 2:
                cv_mean, cv_std = 0, 0
                cv_line = "交叉验证跳过: 某类别样本数少于2。\n"
            else:
                cv_model = model.set_params(**best_params)
                cv_scores = cross_val_score(cv_model, X_train, y_train,
                                            cv=StratifiedKFold(n_splits_cv, shuffle=True, random_state=88),
                                            scoring='f1_macro')
                cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)
                cv_line = f"交叉验证 ({n_splits_cv}-折) F1 (macro): {cv_mean:.4f} (+/- {cv_std:.4f})\n"

            # 最终评估
            final_model = model.set_params(**best_params)
            final_model.fit(X_train, y_train)
            final_models[name] = final_model
            y_pred = final_model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            # 存储报告和统计数据
            report_str = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
            all_reports_data[name] = {
                "report": f"--- {name.upper()} 在测试集上的最终评估 ---\n最佳参数: {best_params}\n{cv_line}{report_str}\n",
                "confusion_matrix": cm
            }
            summary_stats.append(
                {'Model': name.upper(), 'Test_F1_Score': test_f1, 'CV_F1_Mean': cv_mean, 'CV_F1_Std': cv_std})

        sys.stdout = original_stdout
        self.log("所有模型评估完成。")

        for name, data in all_reports_data.items(): self.log(data["report"])

        # ========== 新增：在模型评估完成后生成矩阵 B、统计柱状图与 LDA 图 ==========
        try:
            # ensure save dir exists
            if save_base_dir and not os.path.exists(save_base_dir):
                os.makedirs(save_base_dir, exist_ok=True)

            self.log("\n[新增] 生成矩阵 B（仅 AA 子集）...")
            self.generate_matrix_B(final_models, X_test, y_test, y_test_raw, le, save_base_dir)

            self.log("[新增] 绘制统计柱状图（Train 与 Test 分开）...")
            # train 图使用 X_train_df if available, test 图使用 X_test_df
            self.plot_statistical_bars(X_train_df, X_test_df, y_train_raw=None, y_test_raw=y_test_raw,
                                      feature_columns=feature_columns, save_dir=save_base_dir)

            # pick best model (already computed above)
            best_model_name = pd.DataFrame(summary_stats).sort_values('CV_F1_Mean', ascending=False).iloc[0]['Model'].lower()
            best_model = final_models[best_model_name]
            self.log(f"\n根据交叉验证表现，最佳模型是: {best_model_name.upper()}")

            self.log("[新增] 绘制 LDA 感知图（仅最佳模型）...")
            self.plot_lda_map(X_test_df, y_test_raw, best_model_name, best_model, list(le.classes_), save_base_dir)

        except Exception as e:
            self.log(f"新增可视化生成时出错: {e}")
            self.log(traceback.format_exc())

        # ========== 继续原有流程：显示混淆矩阵，特征重要性，导出报告等 ==========
        for model_name, model_instance in final_models.items():
            self.display_confusion_matrix(model_instance, X_test, y_test, np.array(list(le.classes_)), model_name, save_base_dir)
            self.display_feature_importance(model_instance, feature_columns, model_name)

        if save_base_dir:
            self.export_all_reports_to_excel(all_reports_data, np.array(list(le.classes_)), save_base_dir)
            self.export_summary_to_txt(all_reports_data, summary_stats, save_base_dir)

        # 使用原始（未缩放）的特征列名导出模型
        scaler_to_export = StandardScaler().fit(X_train)  # 重新fit一个scaler以包含所有训练数据
        self.export_best_model(best_model_name, best_model, le, scaler_to_export, feature_columns, save_base_dir)

    # ---------------- 新增函数：生成矩阵 B（AA-only 子集评估） ----------------
    def generate_matrix_B(self, final_models, X_test_scaled, y_test_encoded, y_test_raw, label_encoder, save_dir):
        """
        在同一 19 类模型上，只选取真实标签属于 18 个氨基酸（即 y_test_raw != 'HCl'）的样本，
        然后用模型预测这些样本并绘制 confusion matrix（行仅为 18 个真实 AA，列仍为全部 19 个预测类别）。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # ensure save_dir exists
        os.makedirs(save_dir, exist_ok=True)

        # boolean mask for AA-only true labels
        mask = np.array([str(x) != "HCl" for x in y_test_raw])
        if mask.sum() == 0:
            self.log("警告: 测试集中没有非 HCl 的样本，跳过矩阵 B 生成。")
            return

        # indices of true AA labels in encoded space
        all_classes = list(label_encoder.classes_)
        aa_true_names = [c for c in all_classes if c != "HCl"]
        aa_true_indices = [int(label_encoder.transform([c])[0]) for c in aa_true_names]

        for model_name, model in final_models.items():
            try:
                X_sub = X_test_scaled[mask]
                # true labels encoded for the sub
                y_true_sub = np.array([label_encoder.transform([s])[0] for s in np.array(y_test_raw)[mask]])
                y_pred_sub = model.predict(X_sub)

                # full confusion matrix across all classes (19 x 19)
                cm_full = confusion_matrix(y_true_sub, y_pred_sub, labels=np.arange(len(all_classes)))

                # select only rows corresponding to AA true labels (order follows aa_true_names)
                cm_B = cm_full[aa_true_indices, :]

                plt.figure(figsize=(12, 6))
                sns.heatmap(cm_B, annot=True, fmt='g', cmap="Blues",
                            xticklabels=all_classes, yticklabels=aa_true_names)
                plt.title(f"Confusion Matrix B (AA-only true rows) - {model_name}")
                plt.ylabel("True (AA only)")
                plt.xlabel("Predicted (All classes)")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                out_path = os.path.join(save_dir, f"{model_name}_confusion_matrix_B.png")
                plt.savefig(out_path, dpi=300)
                plt.close()
                self.log(f"矩阵 B 已保存: {out_path}")
            except Exception as e:
                self.log(f"生成矩阵 B ({model_name}) 时出错: {e}")

    # ---------------- 新增函数：统计柱状图（Train / Test 分开） ----------------
    def plot_statistical_bars(self, X_train_df, X_test_df, y_train_raw=None, y_test_raw=None, feature_columns=None, save_dir="."):
        """
        为每个特征列分别绘制两张图（train 与 test），X 轴为 19 类（HCl + 18 AA），Y 轴为均值 ± SEM。
        与 HCl 做 Welch t-test（unequal variance），并在图上标注显著性(* / ns)。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import ttest_ind

        os.makedirs(save_dir, exist_ok=True)

        classes = None
        if y_test_raw is not None:
            classes = list(pd.Series(y_test_raw).unique())
        elif y_train_raw is not None:
            classes = list(pd.Series(y_train_raw).unique())
        else:
            self.log("无法绘制统计图：缺少 y_test_raw 和 y_train_raw。")
            return

        # ensure HCl placed first for consistent visuals
        if "HCl" in classes:
            classes = ["HCl"] + [c for c in classes if c != "HCl"]
        else:
            classes = list(classes)

        # helper to compute mean/sem for given df
        def compute_stats(df, y_raw, feature):
            means = []
            sems = []
            for cls in classes:
                vals = df[np.array(y_raw) == cls][feature] if (y_raw is not None) else pd.Series(dtype=float)
                vals = pd.to_numeric(vals, errors='coerce').dropna()
                means.append(vals.mean() if len(vals) > 0 else 0.0)
                sems.append(vals.sem() if len(vals) > 1 else 0.0)
            return np.array(means), np.array(sems)

        for feat in feature_columns:
            try:
                # -- Test 集 --
                means_t, sems_t = compute_stats(X_test_df, y_test_raw, feat)
                # Welch t-test: HCl vs each other class
                pvals = []
                control_vals = pd.to_numeric(X_test_df[np.array(y_test_raw) == "HCl"][feat], errors='coerce').dropna()
                for cls in classes:
                    if cls == "HCl":
                        pvals.append(np.nan)
                        continue
                    vals = pd.to_numeric(X_test_df[np.array(y_test_raw) == cls][feat], errors='coerce').dropna()
                    if len(control_vals) < 2 or len(vals) < 2:
                        pvals.append(np.nan)
                    else:
                        pvals.append(ttest_ind(control_vals, vals, equal_var=False, nan_policy='omit').pvalue)

                # plotting test
                x = np.arange(len(classes))
                plt.figure(figsize=(12, 5))
                plt.bar(x, means_t, yerr=sems_t, capsize=4, color='tab:blue')
                plt.xticks(x, classes, rotation=90)
                plt.ylabel(feat)
                plt.title(f"Test: Mean ± SEM - {feat}")
                # annotate significance
                ymax = np.nanmax(means_t + sems_t) if len(means_t)>0 else 1
                for i, p in enumerate(pvals):
                    if np.isnan(p):
                        text = '' if classes[i] == "HCl" else 'ns'
                    else:
                        if p < 1e-4:
                            text = '****'
                        elif p < 1e-3:
                            text = '***'
                        elif p < 1e-2:
                            text = '**'
                        elif p < 0.05:
                            text = '*'
                        else:
                            text = 'ns'
                    if text:
                        plt.text(i, means_t[i] + sems_t[i] + 0.02 * (ymax if ymax!=0 else 1), text, ha='center')
                plt.tight_layout()
                out_test = os.path.join(save_dir, f"stat_test_{feat}.png")
                plt.savefig(out_test, dpi=300)
                plt.close()
                self.log(f"已保存 Test 统计图: {out_test}")

                # -- Train 集（若提供） --
                if X_train_df is not None and y_train_raw is not None:
                    means_tr, sems_tr = compute_stats(X_train_df, y_train_raw, feat)
                    # no t-test for train in current config (could be added similarly)
                    x = np.arange(len(classes))
                    plt.figure(figsize=(12, 5))
                    plt.bar(x, means_tr, yerr=sems_tr, capsize=4, color='tab:orange')
                    plt.xticks(x, classes, rotation=90)
                    plt.ylabel(feat)
                    plt.title(f"Train: Mean ± SEM - {feat}")
                    plt.tight_layout()
                    out_train = os.path.join(save_dir, f"stat_train_{feat}.png")
                    plt.savefig(out_train, dpi=300)
                    plt.close()
                    self.log(f"已保存 Train 统计图: {out_train}")
            except Exception as e:
                self.log(f"绘制统计图 {feat} 时出错: {e}")

    # ---------------- 新增函数：LDA 感知图（仅最佳模型） ----------------
    def plot_lda_map(self, X_df, y_raw, best_model_name, best_model, class_names, save_dir):
        """
        使用 LDA (2D) 将高维特征投影并绘制散点图。
        要求：X_df 为原始 DataFrame（未缩放也可），y_raw 为原始标签字符串数组。
        仅对最佳模型绘制。
        """
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.preprocessing import StandardScaler
            import matplotlib.pyplot as plt
            import numpy as np
            os.makedirs(save_dir, exist_ok=True)

            # prepare numeric array
            X = X_df.values
            y = np.array(y_raw)

            # standardize (local only)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # LDA
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_lda = lda.fit_transform(Xs, y)

            plt.figure(figsize=(8, 6))
            unique_classes = list(class_names)
            palette = sns.color_palette("tab20", n_colors=len(unique_classes))
            for idx, cls in enumerate(unique_classes):
                mask = (y == cls)
                if cls == "HCl":
                    plt.scatter(X_lda[mask, 0], X_lda[mask, 1], marker='X', s=80, color=palette[idx], edgecolor='k', label=cls)
                else:
                    plt.scatter(X_lda[mask, 0], X_lda[mask, 1], marker='o', s=30, color=palette[idx], alpha=0.6, label=cls)

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.xlabel('LDA1')
            plt.ylabel('LDA2')
            plt.title(f"LDA Perception Map - Best model: {best_model_name}")
            plt.tight_layout()
            out_path = os.path.join(save_dir, f"LDA_map_{best_model_name}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            self.log(f"LDA 图已保存: {out_path}")
        except Exception as e:
            self.log(f"LDA 可视化时出错: {e}")
            self.log(traceback.format_exc())

    # ---------------- 其余函数（未改动或省略） ----------------
    def select_strategy(self):
        # ... (代码无变化, 已省略)
        dialog = Toplevel(self.master);
        dialog.title("选择捆绑策略");
        dialog.geometry("450x220")
        strategy_var = StringVar(value="分组捆绑")
        tk.Label(dialog, text="请选择处理数据的捆绑策略:", font=('Arial', 14, 'bold')).pack(pady=10)
        Radiobutton(dialog, text="分组捆绑 (保持原貌，公平考试)", variable=strategy_var, value="分组捆绑",
                    font=('Arial', 12)).pack(anchor='w', padx=20)
        Radiobutton(dialog, text="均值聚合 (提炼特征，简化问题)", variable=strategy_var, value="均值聚合",
                    font=('Arial', 12)).pack(anchor='w', padx=20)
        tk.Label(dialog, text="注意: 两种策略都会先按组拆分新数据。", font=('Arial', 10, 'italic')).pack(pady=5)
        result = {"strategy": ""}

        def confirm(): result["strategy"] = strategy_var.get(); dialog.destroy()

        Button(dialog, text="确认", command=confirm, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return result["strategy"]

    def ask_for_incremental_percentage(self):
        # ... (代码无变化, 已省略)
        dialog = Toplevel(self.master)
        dialog.title("配置新数据训练比例")
        dialog.geometry("400x200")
        tk.Label(dialog, text="请输入从'新数据文件'中，\n抽取多少百分比的组用于训练？\n(0-99，剩余的组将作为测试集)",
                 font=('Arial', 12)).pack(pady=10)
        percentage_var = StringVar(value="30")
        entry = Entry(dialog, textvariable=percentage_var, font=('Arial', 12), width=10)
        entry.pack(pady=5)
        result = {'percentage': None}

        def confirm():
            try:
                val = float(percentage_var.get())
                if 0 <= val < 100:
                    result['percentage'] = val
                    dialog.destroy()
                else:
                    messagebox.showerror("输入无效", "请输入一个0到100之间的数字 (不包括100)。", parent=dialog)
            except ValueError:
                messagebox.showerror("输入无效", "请输入一个有效的数字。", parent=dialog)

        Button(dialog, text="确认", command=confirm, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return result['percentage']

    def aggregate_dataframe(self, df, features, label_col, condition_col):
        # ... (代码无变化, 已省略)
        if df.empty:
            return pd.DataFrame(), pd.Series()

        valid_features = [f for f in features if f in df.columns]
        if len(valid_features) != len(features):
            self.log(f"警告: 聚合时发现特征列表与数据不完全匹配。")
            self.log(f"将仅使用 {len(valid_features)} 个有效特征进行聚合。")

        if not valid_features:
            self.log("错误: 找不到任何有效的公共特征进行聚合。")
            messagebox.showerror("聚合错误", "找不到任何有效的公共特征进行聚合操作！")
            return pd.DataFrame(), pd.Series()

        group_id_col = "__AUTO_GROUP_ID__"
        df_copy = df.copy()
        df_copy[group_id_col] = df_copy[label_col].astype(str) + "_" + df_copy[condition_col].astype(str)

        grouped = df_copy.groupby(group_id_col)

        mean_df = grouped[valid_features].mean()
        std_df = grouped[valid_features].std().fillna(0)
        label_df = grouped[label_col].first()

        mean_df.columns = [f"{col}_mean" for col in valid_features]
        std_df.columns = [f"{col}_std" for col in valid_features]

        X_agg = pd.concat([mean_df, std_df], axis=1)
        y_raw_agg = label_df

        self.log(f"数据聚合完成: {len(df_copy)}行 -> {len(X_agg)}行")
        return X_agg.reset_index(drop=True), y_raw_agg.reset_index(drop=True)

    def export_all_reports_to_excel(self, reports_data, class_names, save_dir):
        # ... (代码无变化, 已省略)
        excel_path = os.path.join(save_dir, "model_evaluation_summary.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for model_name, data in reports_data.items():
                    sheet_name = f"{model_name}_Report"
                    report_lines = data['report'].strip().split('\n')
                    report_df = pd.DataFrame(report_lines)
                    report_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

                    cm_df = pd.DataFrame(data['confusion_matrix'], index=class_names, columns=class_names)
                    worksheet = writer.sheets[sheet_name]
                    start_row = len(report_lines) + 2
                    worksheet.cell(row=start_row, column=1, value="Confusion Matrix")
                    cm_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row + 1)
            self.log(f"所有模型评估报告和混淆矩阵已导出至Excel: {excel_path}")
        except Exception as e:
            self.log(f"导出到Excel失败: {e}")

    def export_summary_to_txt(self, reports_data, summary_stats, save_dir):
        """将详细报告和最终汇总表写入TXT文件。"""
        txt_path = os.path.join(save_dir, "model_summary_report.txt")
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 20 + " 模型评估详细报告 " + "=" * 20 + "\n\n")
                for name, data in reports_data.items():
                    f.write(data['report'])
                    f.write("\n" + "-" * 50 + "\n\n")

                f.write("\n\n" + "=" * 20 + " 所有模型性能最终汇总 " + "=" * 20 + "\n\n")
                summary_df = pd.DataFrame(summary_stats)
                summary_df['Test_F1_Score'] = summary_df['Test_F1_Score'].map('{:.4f}'.format)
                summary_df['CV_F1_Mean'] = summary_df['CV_F1_Mean'].map('{:.4f}'.format)
                summary_df['CV_F1_Std'] = summary_df['CV_F1_Std'].map('{:.4f}'.format)

                summary_df = summary_df.sort_values(by='CV_F1_Mean', ascending=False).reset_index(drop=True)

                f.write("说明:\n")
                f.write("- Test_F1_Score: 模型在独立测试集上的宏平均F1分数，反映泛化能力。\n")
                f.write("- CV_F1_Mean: 在训练集上进行交叉验证的平均F1分数，反映模型在当前数据上的综合性能。\n")
                f.write("- CV_F1_Std: 交叉验证F1分数的标准差，值越小说明模型性能越稳定。\n\n")
                f.write(summary_df.to_string())
                f.write("\n\n" + "=" * 62)

            self.log(f"所有模型评估摘要及汇总已导出至TXT文件: {txt_path}")
        except Exception as e:
            self.log(f"导出到TXT文件失败: {e}")

    def select_column_role(self, columns, title):
        # ... (代码无变化, 已省略)
        dialog = Toplevel(self.master);
        dialog.title(title);
        dialog.geometry("400x450")
        selected_col = StringVar(value=columns[0] if columns else "")
        top_frame = Frame(dialog, pady=10);
        top_frame.pack(fill=tk.X)
        tk.Label(top_frame, text=title, font=('Arial', 12, 'bold')).pack()
        scroll_frame = Frame(dialog);
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(scroll_frame);
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        inner_frame = Frame(canvas);
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for col in columns: Radiobutton(inner_frame, text=col, variable=selected_col, value=col,
                                        font=('Arial', 12)).pack(anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        bottom_frame = Frame(dialog, pady=10);
        bottom_frame.pack(fill=tk.X)
        Button(bottom_frame, text="确认", command=dialog.destroy, font=('Arial', 12, 'bold')).pack()
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_col.get()

    def write(self, text):
        if text.strip(): self.log(text.strip())

    def flush(self):
        pass

    def visualize_best_model(self, best_model_name, best_model, feature_names, class_names):
        # ... (代码无变化, 已省略)
        if not best_model: return
        if isinstance(best_model, DecisionTreeClassifier):
            fig, ax = plt.subplots(figsize=(20, 15));
            plot_tree(best_model, filled=True, feature_names=feature_names.tolist(), class_names=class_names.tolist(),
                      rounded=True, fontsize=10, max_depth=4)
            ax.set_title(f"最佳模型 ({best_model_name}) 决策树可视化", fontsize=20);
            plt.show(block=False)
        elif isinstance(best_model, RandomForestClassifier):
            fig, ax = plt.subplots(figsize=(20, 15));
            plot_tree(best_model.estimators_[0], filled=True, feature_names=feature_names.tolist(),
                      class_names=class_names.tolist(), rounded=True, fontsize=10, max_depth=4)
            ax.set_title(f"最佳模型 ({best_model_name}) 的第一棵树", fontsize=20);
            plt.show(block=False)

    def export_best_model(self, best_model_name, best_model, label_encoder, scaler, feature_columns, save_dir):
        # ... (代码无变化, 已省略)
        if not all([best_model_name, best_model, save_dir, os.path.exists(save_dir)]): return
        model_package = {'model_name': best_model_name, 'model': best_model, 'label_encoder': label_encoder,
                         'scaler': scaler, 'feature_columns': feature_columns.tolist()}
        file_path = os.path.join(save_dir, 'best_model_package.joblib')
        try:
            joblib.dump(model_package, file_path)
            self.log(f"成功！最佳模型包已导出至: {file_path}")
            messagebox.showinfo("模型已导出", f"最佳模型包已成功保存至:\n{file_path}", parent=self.master)
        except Exception as e:
            self.log(f"错误：导出模型失败 - {e}")

    def select_features(self, all_features):
        # ... (代码无变化, 已省略)
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

    def display_confusion_matrix(self, model, X_test, y_test, target_names, model_name, save_dir):
        # ... (代码无变化, 已省略)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(target_names)))

        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = np.where(cm_sum > 0, (cm.astype('float') / cm_sum) * 100, 0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_percent, annot=True, fmt='.0f', cmap='Blues', xticklabels=target_names, yticklabels=target_names,
                    ax=ax)

        ax.set_title(f'{model_name} Confusion Matrix (%)', fontsize=20)
        ax.set_ylabel('True Label', fontsize=16)
        ax.set_xlabel('Predicted Label', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_dir and os.path.exists(save_dir):
            plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix_percent.png'))
        plt.show(block=False)

    def display_feature_importance(self, model, columns, model_name):
        # ... (代码无变化, 已省略)
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
        else:
            self.log(f"注意: 模型 {model_name} 不提供直接的特征重要性度量。");
            return
        df_imp = pd.DataFrame({'feature': columns, 'importance': importances}).sort_values('importance',
                                                                                           ascending=False).head(20)
        self.log(f"\n--- {model_name} Top Feature Importances ---");
        self.log(df_imp.to_string())
        fig, ax = plt.subplots(figsize=(12, 8));
        sns.barplot(x='importance', y='feature', data=df_imp, ax=ax)
        ax.set_title(f'{model_name} Feature Importance', fontsize=20);
        ax.set_xlabel('Importance', fontsize=16);
        ax.set_ylabel('Feature', fontsize=16)
        plt.tight_layout();
        plt.show(block=False)


if __name__ == '__main__':
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()
