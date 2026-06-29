import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
    Text, Scrollbar, END, simpledialog
import os
import matplotlib as mpl
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import traceback
import sys
import threading
import joblib
import openpyxl
import re

# --- Matplotlib全局设置 ---
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


class AnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("分类工具 (CV修复版)")
        master.geometry("750x550")

        self.control_frame = Frame(master, padx=10, pady=10)
        self.control_frame.pack(fill=tk.X)
        self.start_button = Button(self.control_frame, text="开始分析", font=('Arial', 14, 'bold'),
                                   command=self.prepare_and_start_analysis)
        self.start_button.pack(pady=10)

        self.log_frame = Frame(master, padx=10, pady=10)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = Text(self.log_frame, wrap=tk.WORD, state='disabled', font=('Courier New', 11))
        self.scrollbar = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.log("欢迎使用分类工具！")
        self.log("此版本已修复均值聚合策略下的交叉验证问题。")

    # --- UI 和线程管理 ---
    def log(self, message):
        def append_log():
            if not self.master.winfo_exists(): return
            self.log_text.config(state='normal')
            self.log_text.insert(END, str(message) + "\n")
            self.log_text.see(END)
            self.log_text.config(state='disabled')

        self.master.after(0, append_log)

    def prepare_and_start_analysis(self):
        self.log("准备分析... 请根据弹窗提示操作。")

        file_path = filedialog.askopenfilename(parent=self.master, title="选择数据CSV文件",
                                               filetypes=[("CSV files", "*.csv")])
        if not file_path: self.log("操作取消。"); return

        save_base_dir = filedialog.askdirectory(parent=self.master, title="选择结果保存目录")
        if not save_base_dir: self.log("操作取消。"); return

        try:
            df_for_ui = pd.read_csv(file_path)
        except Exception as e:
            messagebox.showerror("文件读取错误", f"无法读取文件: {e}")
            self.log(f"文件读取错误: {e}")
            return

        strategy = self.select_strategy()
        if not strategy: self.log("操作取消。"); return

        label_column = self.select_column_role(df_for_ui.columns.tolist(), "请选择主要类别列 (如 AA)")
        if not label_column: self.log("操作取消。"); return

        condition_column = self.select_column_role(df_for_ui.columns.tolist(), "请选择组内条件列 (如 浓度/uM)")
        if not condition_column: self.log("操作取消。"); return

        all_features = [col for col in df_for_ui.columns if col not in [label_column, condition_column]]
        selected_features = self.select_features(all_features)
        if not selected_features: self.log("操作取消。"); return

        all_amino_acids = df_for_ui[label_column].dropna().unique().tolist()
        all_amino_acids.sort()
        selected_acids = self.select_amino_acids(all_amino_acids)
        if not selected_acids: self.log("操作取消。"); return

        test_size = simpledialog.askfloat("设置测试集比例", "请输入测试集的比例 (0.1 到 0.5 之间):",
                                          parent=self.master, minvalue=0.1, maxvalue=0.5, initialvalue=0.33)
        if not test_size: self.log("操作取消。"); return

        params = {
            'file_path': file_path, 'save_base_dir': save_base_dir, 'strategy': strategy,
            'label_column': label_column, 'condition_column': condition_column,
            'selected_features': selected_features, 'selected_acids': selected_acids,
            'test_size': test_size
        }

        self.start_button.config(state="disabled", text="分析中...")
        self.log("\n" + "=" * 50)
        self.log("用户输入完成，开始后台分析任务...")
        analysis_thread = threading.Thread(target=self.run_analysis_wrapper, args=(params,))
        analysis_thread.daemon = True
        analysis_thread.start()

    def run_analysis_wrapper(self, params):
        try:
            self.run_optimization_analysis(params)
        except Exception as e:
            error_details = traceback.format_exc()
            self.log(f"!!! 发生严重错误 !!!\n错误类型: {type(e).__name__}\n错误信息: {e}")
            self.log(f"详细追溯信息:\n{error_details}")
            self.master.after(0, lambda: messagebox.showerror("程序遇到意外错误",
                                                              f"发生未处理的错误: {e}\n\n请查看主窗口日志获取详细信息。"))
        finally:
            self.master.after(0, lambda: self.start_button.config(state="normal", text="开始分析"))
            self.log("分析任务结束。")

    # --- 主分析流程 ---
    def run_optimization_analysis(self, params):
        file_path = params['file_path'];
        save_base_dir = params['save_base_dir'];
        strategy = params['strategy']
        label_column = params['label_column'];
        condition_column = params['condition_column']
        selected_features = params['selected_features'];
        selected_acids = params['selected_acids']
        test_size = params['test_size']

        self.log("步骤 3: 正在加载数据...");
        df = pd.read_csv(file_path)
        self.log(f"已选择策略: {strategy}")
        self.log("步骤 7: 正在智能创建分组ID...")
        group_id_col = "__AUTO_GROUP_ID__"
        df[group_id_col] = df[label_column].astype(str) + "_" + df[condition_column].astype(str)
        self.log(f"已创建分组ID，例如: {df[group_id_col].iloc[0]}")
        df_filtered = df[df[label_column].isin(selected_acids)].copy()

        self.log("步骤 10: 数据降维可视化 (基于完整数据)...")
        self.visualize_data_distribution(df_filtered.copy(), label_column, selected_features)

        self.log("步骤 11: 数据预处理与稳健分割...")
        le = LabelEncoder();
        le.fit(df_filtered[label_column])
        X_train, y_train, X_test, y_test = None, None, None, None

        if strategy == "分组捆绑":
            X = df_filtered[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            y = le.transform(df_filtered[label_column]);

            self.log("采用分层分组策略进行数据划分...")
            group_labels = df_filtered.drop_duplicates(subset=[group_id_col])[[group_id_col, label_column]]

            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            try:
                train_group_idx, test_group_idx = next(
                    sss.split(group_labels[group_id_col], group_labels[label_column]))
            except ValueError as e:
                self.log(f"!!! 严重错误: 分层划分失败! {e}")
                self.log("!!! 这通常意味着某个类别中的组（平行实验）数量太少，无法进行按比例划分。")
                self.log("!!! 建议: 1. 增加该类别的数据组数。 2. 切换到'均值聚合'策略。")
                messagebox.showerror("数据划分失败", f"分层划分失败，请检查日志获取详细信息。\n错误: {e}");
                return

            train_groups = group_labels.iloc[train_group_idx][group_id_col]
            test_groups = group_labels.iloc[test_group_idx][group_id_col]

            train_idx = df_filtered[df_filtered[group_id_col].isin(train_groups)].index
            test_idx = df_filtered[df_filtered[group_id_col].isin(test_groups)].index

            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.log(f"分层分组划分成功！训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
            test_support = pd.Series(le.inverse_transform(y_test)).value_counts().to_dict()
            self.log(f"测试集样本分布 (Support): {test_support}")

        else:  # 均值聚合策略
            X_agg, y_agg_raw, le_agg = self.preprocess_with_aggregation(df_filtered, selected_features, label_column,
                                                                        group_id_col)
            # 使用聚合后的le
            le = le_agg
            y_agg = le.transform(y_agg_raw)
            X_train, X_test, y_train, y_test = train_test_split(X_agg, y_agg, test_size=test_size, random_state=42,
                                                                stratify=y_agg)

        scaler = StandardScaler();
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        best_model_name, best_params = self.run_hyperopt_and_evaluation(
            X_train_scaled, y_train, X_test_scaled, y_test, le, scaler, save_base_dir)

        if best_model_name is None:
            self.log("未找到最佳模型，无法继续进行重训练和导出。");
            return

        self.log("\n" + "=" * 20 + " 最终模型重训练 " + "=" * 20)
        self.log("步骤 14: 使用全部数据重新训练最佳模型以供导出...")

        if strategy == "分组捆绑":
            X_all = df_filtered[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            y_all_raw = df_filtered[label_column]
        else:
            X_all, y_all_raw_agg, _ = self.preprocess_with_aggregation(df_filtered, selected_features, label_column,
                                                                       group_id_col)
            y_all_raw = y_all_raw_agg

        le_final = LabelEncoder();
        y_all = le_final.fit_transform(y_all_raw)
        scaler_final = StandardScaler();
        X_all_scaled = scaler_final.fit_transform(X_all)

        models_dict = {
            'dt': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'rf': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'svm': SVC(probability=True, random_state=42, kernel='linear', class_weight='balanced'),
            'xgb': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'lr': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'mlp': MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
        }

        final_production_model = models_dict[best_model_name].set_params(**best_params)
        final_production_model.fit(X_all_scaled, y_all)
        self.log(f"最佳模型 ({best_model_name.upper()}) 已在全部 {len(X_all)} 个样本上重新训练完成。")

        self.export_best_model(best_model_name, final_production_model, le_final, scaler_final, X_all.columns,
                               save_base_dir)

    def run_hyperopt_and_evaluation(self, X_train_scaled, y_train, X_test_scaled, y_test, le, scaler, save_base_dir):
        self.log("步骤 12: 超参数优化与评估...")
        models_to_tune = {
            'dt': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'rf': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'svm': SVC(probability=True, random_state=42, kernel='linear', class_weight='balanced'),
            'xgb': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'lr': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'mlp': MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
        }
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

        final_models = {};
        all_models_report = [];
        all_best_params = {}
        summary_stats = []
        original_stdout = sys.stdout;
        sys.stdout = self

        for name, model in models_to_tune.items():
            self.log(f"--- 正在优化 {name.upper()} ---")

            def objective(params):
                params_for_model = params.copy()
                for key in ['max_depth', 'n_estimators']:
                    if key in params_for_model: params_for_model[key] = int(params_for_model[key])
                model.set_params(**params_for_model)
                skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                score = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='f1_macro', n_jobs=-1).mean()
                return {'loss': -score, 'status': STATUS_OK}

            trials = Trials()
            best_params_raw = fmin(fn=objective, space=hp_space[name], algo=tpe.suggest, max_evals=50, trials=trials,
                                   rstate=np.random.default_rng(42), show_progressbar=False)
            best_params = space_eval(hp_space[name], best_params_raw)
            for key in ['max_depth', 'n_estimators']:
                if key in best_params: best_params[key] = int(best_params[key])

            all_best_params[name] = best_params

            # --- [修改点] 智能动态交叉验证 ---
            # 1. 计算训练集中最小类别的样本数
            min_class_count = np.min(np.bincount(y_train))
            # 2. 动态确定CV折数，必须>=2且<=最小类别样本数
            n_splits_cv = min(5, min_class_count)

            if n_splits_cv < 2:
                cv_mean, cv_std = 0, 0
                cv_line = "Cross-Validation skipped: a class has less than 2 samples in the training set.\n"
                self.log(f"模型 {name.upper()}: 训练集中有类别样本数少于2，跳过交叉验证。")
            else:
                final_model_for_cv = models_to_tune[name].set_params(**best_params)
                skf_for_stats = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=88)
                try:
                    cv_scores = cross_val_score(final_model_for_cv, X_train_scaled, y_train, cv=skf_for_stats,
                                                scoring='f1_macro', n_jobs=-1)
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    cv_line = f"Cross-Validation ({n_splits_cv}-fold) F1 (macro): {cv_mean:.4f} (+/- {cv_std:.4f})\n"
                except Exception as cv_e:
                    cv_mean, cv_std = 0, 0
                    cv_line = f"Cross-Validation failed: {cv_e}\n"
            # --- [修改点结束] ---

            final_model = models_to_tune[name].set_params(**best_params)
            final_model.fit(X_train_scaled, y_train)
            final_models[name] = final_model
            y_pred = final_model.predict(X_test_scaled)

            test_f1_score = f1_score(y_test, y_pred, average='macro', zero_division=0)

            report = classification_report(y_test, y_pred, labels=np.arange(len(le.classes_)), target_names=le.classes_,
                                           zero_division=0)

            report_str = f"--- {name.upper()} 评估模型报告 ---\nBest Parameters: {best_params}\n{cv_line}{report}\n"
            all_models_report.append(report_str)

            summary_stats.append({
                'Model': name.upper(),
                'Test_F1_Score': test_f1_score,
                'CV_F1_Mean': cv_mean,
                'CV_F1_Std': cv_std
            })

        sys.stdout = original_stdout;
        self.log("评估模型优化完成。")

        self.log("步骤 13: 生成评估结果...")
        for report in all_models_report: self.log(report)
        best_model_name, best_model_instance = self.find_best_model(final_models, X_test_scaled, y_test)

        if best_model_name is None:
            self.log("所有模型评估分数都为0，无法确定最佳模型。");
            return None, None

        self.visualize_best_model(best_model_name, best_model_instance, X_train_scaled.columns, le.classes_)

        for model_name, model in final_models.items():
            self.display_confusion_matrix(model, X_test_scaled, y_test, le.classes_, model_name, save_base_dir)
            self.display_feature_importance(model, X_train_scaled.columns, model_name)

        self.export_all_reports_to_excel(all_models_report, save_base_dir)
        self.export_summary_report_to_txt(all_models_report, summary_stats, save_base_dir)
        return best_model_name, all_best_params.get(best_model_name, {})

    # --- 辅助函数 ---
    def display_confusion_matrix(self, model, X_test, y_test, target_names, model_name, save_dir):
        y_pred = model.predict(X_test)
        all_possible_labels = np.arange(len(target_names))
        cm = confusion_matrix(y_test, y_pred, labels=all_possible_labels)

        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = cm.astype('float') / cm_sum
            cm_percent[np.isnan(cm_percent)] = 0
        annot_labels = (np.asarray([f'{p * 100:.0f}' for p in cm_percent.flatten()])
                        ).reshape(cm.shape)

        num_classes = len(target_names)
        annot_fontsize = 14 if num_classes < 15 else 12
        tick_fontsize = 14
        label_fontsize = 20
        title_fontsize = 24
        figsize_w = 13 if num_classes <= 15 else 16
        figsize_h = 11 if num_classes <= 15 else 13

        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))

        sns.heatmap(cm_percent, annot=annot_labels, fmt='s', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names, ax=ax, vmin=0, vmax=1,
                    annot_kws={"size": annot_fontsize})

        ax.set_title(f'{model_name} Confusion Matrix', fontsize=title_fontsize)
        ax.set_ylabel('True Label', fontsize=label_fontsize)
        ax.set_xlabel('Predicted Label', fontsize=label_fontsize)
        plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)
        plt.tight_layout(pad=3.0)
        if save_dir and os.path.exists(save_dir):
            plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
        plt.show(block=False)

    def export_summary_report_to_txt(self, reports, summary_stats, save_dir):
        if not save_dir or not os.path.exists(save_dir): return
        txt_path = os.path.join(save_dir, "model_summary_report.txt")
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 20 + " 模型评估详细报告 " + "=" * 20 + "\n\n")
                for report_str in reports:
                    f.write(report_str);
                    f.write("\n" + "-" * 50 + "\n\n")

                f.write("\n\n" + "=" * 20 + " 所有模型性能最终汇总 " + "=" * 20 + "\n\n")
                summary_df = pd.DataFrame(summary_stats)
                summary_df['Test_F1_Score'] = summary_df['Test_F1_Score'].map('{:.4f}'.format)
                summary_df['CV_F1_Mean'] = summary_df['CV_F1_Mean'].map('{:.4f}'.format)
                summary_df['CV_F1_Std'] = summary_df['CV_F1_Std'].map('{:.4f}'.format)

                summary_df = summary_df.sort_values(by='CV_F1_Mean', ascending=False)
                summary_df.reset_index(drop=True, inplace=True)

                f.write("说明:\n")
                f.write("- Test_F1_Score: 模型在独立测试集上的宏平均F1分数，反映泛化能力。\n")
                f.write("- CV_F1_Mean: 在训练集上进行交叉验证的平均F1分数，反映模型在当前数据上的综合性能。\n")
                f.write("- CV_F1_Std: 交叉验证F1分数的标准差，值越小说明模型性能越稳定。\n\n")

                f.write(summary_df.to_string())
                f.write("\n\n" + "=" * 62)

            self.log(f"所有模型评估报告及最终汇总已保存至TXT文件: {txt_path}")
        except Exception as e:
            self.log(f"导出到TXT失败: {e}")

    def preprocess_with_aggregation(self, df, features, label_col, group_col):
        self.log("策略: 均值聚合。")
        df_features = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        df_full = pd.concat([df[[label_col, group_col]], df_features], axis=1)
        grouped = df_full.groupby(group_col)
        mean_df = grouped[features].mean();
        std_df = grouped[features].std().fillna(0)
        label_df = grouped[label_col].first().reset_index()
        mean_df.columns = [f"{col}_mean" for col in features]
        std_df.columns = [f"{col}_std" for col in features]
        X_agg = pd.concat([mean_df.reset_index(drop=True), std_df.reset_index(drop=True)], axis=1)
        y_raw_agg = label_df[label_col]
        self.log(f"数据聚合完成。样本数从 {len(df)} 减少到 {len(X_agg)}。")
        le_agg = LabelEncoder();
        le_agg.fit(y_raw_agg)
        return X_agg, y_raw_agg, le_agg

    def select_strategy(self):
        dialog = Toplevel(self.master);
        dialog.title("选择策略");
        dialog.geometry("500x220")
        strategy_var = StringVar(value="分组捆绑")
        tk.Label(dialog, text="请选择处理平行实验的策略:", font=('Arial', 16, 'bold')).pack(pady=10)
        Radiobutton(dialog, text="分组捆绑 (保持原貌，公平考试 - 推荐)", variable=strategy_var, value="分组捆绑",
                    font=('Arial', 14)).pack(anchor='w', padx=20)
        Radiobutton(dialog, text="均值聚合 (提炼特征，简化问题)", variable=strategy_var, value="均值聚合",
                    font=('Arial', 14)).pack(anchor='w', padx=20)
        result = {"strategy": ""}

        def confirm(): result["strategy"] = strategy_var.get(); dialog.destroy()

        Button(dialog, text="确认", command=confirm, font=('Arial', 14, 'bold')).pack(pady=15)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return result["strategy"]

    def write(self, text):
        if text.strip(): self.log(text.strip())

    def flush(self):
        pass

    def find_best_model(self, final_models, X_test, y_test):
        best_model_name = None;
        best_score = -1
        for name, model in final_models.items():
            y_pred = model.predict(X_test);
            score = f1_score(y_test, y_pred, average='macro', zero_division=0)
            self.log(f"模型 {name} 的 F1 分数 (macro): {score:.4f}")
            if score > best_score: best_score = score; best_model_name = name
        if best_score <= 0: return None, None
        self.log(f"\n性能最佳的模型是: {best_model_name} (F1 score: {best_score:.4f})")
        return best_model_name, final_models.get(best_model_name)

    def visualize_best_model(self, best_model_name, best_model, feature_names, class_names):
        if best_model is None: return
        if isinstance(best_model, DecisionTreeClassifier):
            fig, ax = plt.subplots(figsize=(20, 15));
            plot_tree(best_model, filled=True, feature_names=feature_names.tolist(), class_names=class_names,
                      rounded=True, fontsize=12, max_depth=4)
            ax.set_title(f"最佳模型 ({best_model_name}) 决策树可视化", fontsize=22);
            plt.show(block=False)
        elif isinstance(best_model, RandomForestClassifier):
            fig, ax = plt.subplots(figsize=(20, 15));
            plot_tree(best_model.estimators_[0], filled=True, feature_names=feature_names.tolist(),
                      class_names=class_names, rounded=True, fontsize=12, max_depth=4)
            ax.set_title(f"最佳模型 ({best_model_name}) 的第一棵树", fontsize=22);
            plt.show(block=False)

    def export_best_model(self, best_model_name, best_model, label_encoder, scaler, feature_columns, save_dir):
        if not save_dir or not os.path.exists(save_dir): self.log("未选择有效保存目录，跳过模型导出。"); return
        model_package = {'model_name': best_model_name, 'model': best_model, 'label_encoder': label_encoder,
                         'scaler': scaler, 'feature_columns': feature_columns.tolist()}
        file_path = os.path.join(save_dir, 'best_model_package.joblib')
        try:
            joblib.dump(model_package, file_path);
            self.log(f"成功！最终生产模型包已导出至: {file_path}")
            if self.master.winfo_exists():
                messagebox.showinfo("模型已导出", f"最终生产模型包已成功保存至:\n{file_path}", parent=self.master)
        except Exception as e:
            self.log(f"错误：导出模型失败 - {e}")

    def visualize_data_distribution(self, df, label_column, selected_features):
        self.log("正在准备数据用于可视化...")
        X = df[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df[label_column];
        X_scaled = StandardScaler().fit_transform(X)

        self.log("正在进行 PCA 计算...");
        pca = PCA(n_components=2);
        X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(10, 8));
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, ax=ax, palette='viridis', legend='full')
        ax.set_title(f'PCA 可视化 (解释方差: {pca.explained_variance_ratio_.sum():.2%})', fontsize=18)
        ax.legend(title=label_column, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12);
        plt.tight_layout();
        plt.show(block=False)

        self.log("正在进行 t-SNE 计算...");
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) - 1));
        X_tsne = tsne.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(10, 8));
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, ax=ax, palette='viridis', legend='full')
        ax.set_title('t-SNE 可视化', fontsize=18);
        ax.legend(title=label_column, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12);
        plt.tight_layout();
        plt.show(block=False)

    def select_column_role(self, columns, title):
        dialog = Toplevel(self.master);
        dialog.title(title);
        dialog.geometry("400x450")
        selected_col = StringVar(value=columns[0] if columns else "")
        top_frame = Frame(dialog, pady=10);
        top_frame.pack(fill=tk.X)
        tk.Label(top_frame, text=title, font=('Arial', 14, 'bold')).pack()
        scroll_frame = Frame(dialog);
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(scroll_frame);
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        inner_frame = Frame(canvas);
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for col in columns: Radiobutton(inner_frame, text=col, variable=selected_col, value=col,
                                        font=('Arial', 14)).pack(anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        bottom_frame = Frame(dialog, pady=10);
        bottom_frame.pack(fill=tk.X)
        Button(bottom_frame, text="确认", command=dialog.destroy, font=('Arial', 14, 'bold')).pack()
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
            nonlocal selected_features_list
            selected_features_list = [feature for feature, var in vars.items() if var.get()]
            if not selected_features_list: messagebox.showerror("错误", "请至少选择一个特征列！", parent=dialog); return
            dialog.destroy()

        top_frame = Frame(dialog, pady=5);
        top_frame.pack(fill=tk.X, padx=10)
        Button(top_frame, text="全选", command=lambda: [v.set(True) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                  expand=True, padx=5)
        Button(top_frame, text="全不选", command=lambda: [v.set(False) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                     expand=True,
                                                                                                     padx=5)
        mid_frame = Frame(dialog);
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(mid_frame);
        scrollbar = tk.Scrollbar(mid_frame, orient="vertical", command=canvas.yview)
        inner_frame = Frame(canvas);
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for feature in all_features: Checkbutton(inner_frame, text=feature, var=vars[feature], font=('Arial', 14)).pack(
            anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        bottom_frame = Frame(dialog, pady=10);
        bottom_frame.pack(fill=tk.X, padx=10)
        Button(bottom_frame, text="确认选择", command=confirm, font=('Arial', 14, 'bold')).pack()
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_features_list

    def select_amino_acids(self, all_labels):
        dialog = Toplevel(self.master);
        dialog.title("选择要训练的分类目标");
        dialog.geometry("400x500")
        selected_labels_list = [];
        vars = {label: BooleanVar(value=True) for label in all_labels}

        def confirm():
            nonlocal selected_labels_list
            selected_labels_list = [label for label, var in vars.items() if var.get()]
            if len(selected_labels_list) < 2: messagebox.showerror("错误", "请至少选择两个分类目标！",
                                                                   parent=dialog); return
            dialog.destroy()

        top_frame = Frame(dialog, pady=5);
        top_frame.pack(fill=tk.X, padx=10)
        Button(top_frame, text="全选", command=lambda: [v.set(True) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                  expand=True, padx=5)
        Button(top_frame, text="全不选", command=lambda: [v.set(False) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                     expand=True,
                                                                                                     padx=5)
        mid_frame = Frame(dialog);
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(mid_frame);
        scrollbar = tk.Scrollbar(mid_frame, orient="vertical", command=canvas.yview)
        inner_frame = Frame(canvas);
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for label in all_labels: Checkbutton(inner_frame, text=label, var=vars[label], font=('Arial', 14)).pack(
            anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        bottom_frame = Frame(dialog, pady=10);
        bottom_frame.pack(fill=tk.X, padx=10)
        Button(bottom_frame, text="确认选择", command=confirm, font=('Arial', 14, 'bold')).pack()
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_labels_list

    def display_feature_importance(self, model, columns, model_name):
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
        else:
            self.log(f"注意: 模型 {model_name} 不提供直接的特征重要性度量。");
            return
        df_imp = pd.DataFrame({'feature': columns, 'importance': importances}).sort_values('importance',
                                                                                           ascending=False).head(20)
        self.log(f"\n--- {model_name} Top Feature Importances ---");
        self.log(df_imp.to_string())

        fig, ax = plt.subplots(figsize=(12, 8));
        sns.barplot(x='importance', y='feature', data=df_imp, ax=ax)
        ax.set_title(f'{model_name} Feature Importance', fontsize=22);
        ax.set_xlabel('Importance', fontsize=18);
        ax.set_ylabel('Feature', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout();
        plt.show(block=False)

    def export_all_reports_to_excel(self, reports, save_dir):
        if not save_dir or not os.path.exists(save_dir): return
        excel_path = os.path.join(save_dir, "model_evaluation_summary.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                summary_data = []
                for report_str in reports:
                    model_name_match = re.search(r"--- (.*?) 评估模型报告 ---", report_str)
                    if not model_name_match: continue
                    model_name = model_name_match.group(1).strip()
                    params_match = re.search(r"Best Parameters: (\{.*?\})", report_str, re.DOTALL)
                    params = params_match.group(1) if params_match else "{}"
                    report_lines = report_str.split('\n')
                    header_line = [h for h in report_lines if 'precision' in h]
                    if not header_line: continue
                    col_names = ['class'] + [h for h in header_line[0].split() if h]
                    report_data = []
                    for line in report_lines:
                        if line.strip() and len(line.split()) > 2 and any(char.isdigit() for char in line):
                            parts = line.split()
                            split_point = -4
                            if 'avg' in parts or 'accuracy' in parts:
                                split_point = -3 if 'accuracy' in parts else -4

                            class_name = " ".join(parts[:split_point]).strip()
                            metrics = parts[split_point:]
                            if class_name and len(metrics) in [3, 4]:
                                report_data.append([class_name] + metrics)

                    if report_data:
                        df_report = pd.DataFrame(report_data)
                        df_report.columns = col_names[:len(df_report.columns)]
                        df_report.to_excel(writer, sheet_name=f"{model_name}_report", index=False)

                    macro_avg_line = [line for line in report_lines if 'macro avg' in line]
                    if macro_avg_line:
                        f1_score_val = macro_avg_line[0].split()[-2]
                        summary_data.append(
                            {'Model': model_name, 'F1 Score (macro avg)': f1_score_val, 'Best Parameters': params})

                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name="Summary", index=False)
            self.log(f"所有模型评估报告已导出至Excel文件: {excel_path}")
        except Exception as e:
            self.log(f"导出到Excel失败: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()
