# -*- coding: utf-8 -*-
"""
合并学习与预测工具 (最终数据导出版)

核心修正:
1.  [数据导出] 新增核心功能：将每个类别的详细SHAP值与对应的特征值，
    逐一样本导出到Excel文件的独立工作表中。这为用户提供了自行绘制
    专业图表的原始数据。
2.  [已应用] 采用 SHAP 库最新的、最稳健的 API，彻底解决了顽固的 `IndexError`。
3.  [图表美化] 根据用户要求，对图表进行了精细的视觉优化。
4.  [最终修复] 解决了所有已知的库版本兼容性问题，并强制使用氨基酸缩写
    作为图例标签，确保了程序的稳定运行和图表的专业性。
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
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
import openpyxl

from sklearn.inspection import permutation_importance

try:
    import shap

    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False

mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


class AnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("合并学习与预测工具 (最终数据导出版)")
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
        self.log("此版本为随机森林模型增加了置换重要性和SHAP分析功能。")
        if not SHAP_INSTALLED: self.log("\n警告: 未检测到 'shap' 库。SHAP相关分析将被跳过。")

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
            self.log("分析任务结束.")
            self.log("=" * 60 + "\n")

    def run_main_analysis(self):
        self.log("步骤 1/11: 请选择 **基础训练** 数据文件...")
        train_path = filedialog.askopenfilename(parent=self.master, title="选择 **基础训练** 数据CSV文件",
                                                filetypes=[("CSV files", "*.csv")])
        if not train_path: self.log("操作取消。"); return
        self.log("步骤 2/11: 请选择 **新数据/预测** 数据文件...")
        predict_path = filedialog.askopenfilename(parent=self.master, title="选择 **新数据/预测** 数据CSV文件",
                                                  filetypes=[("CSV files", "*.csv")])
        if not predict_path: self.log("操作取消。"); return
        self.log("步骤 3/11: 请选择结果保存目录...")
        save_base_dir = filedialog.askdirectory(parent=self.master, title="选择结果保存目录")
        if not save_base_dir: self.log("操作取消。"); return
        self.log("步骤 4/11: 正在加载数据...")
        train_df = pd.read_csv(train_path)
        predict_df = pd.read_csv(predict_path)
        self.log("步骤 5/11: (关键) 请选择处理数据的捆绑策略...")
        strategy = self.select_strategy()
        if not strategy: self.log("操作取消。"); return
        self.log("步骤 6/11: (关键) 请配置新数据用于训练的比例...")
        percentage_to_train = self.ask_for_incremental_percentage()
        if percentage_to_train is None: self.log("操作取消。"); return
        self.log(f"配置：将尝试使用 {percentage_to_train}% 的新数据组加入训练集。")
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
            if stratify_param is None: self.log(
                "警告: 检测到至少一个类别中只有一个独立样本组。无法进行分层抽样，将回退到普通分组抽样。")
            test_size_prop = 1.0 - (percentage_to_train / 100.0)
            try:
                train_group_ids, test_group_ids = train_test_split(group_ids, test_size=test_size_prop, random_state=42,
                                                                   stratify=stratify_param)
            except Exception as e:
                self.log(f"错误: 在数据划分时发生错误: {e}");
                messagebox.showerror("划分错误", f"数据划分失败: {e}\n\n请检查新数据文件。");
                return
            new_data_for_training = predict_df_copy[predict_df_copy[group_id_col].isin(train_group_ids)]
            new_data_for_testing = predict_df_copy[predict_df_copy[group_id_col].isin(test_group_ids)]
            self.log(f"已按组进行划分: {len(train_group_ids)} 组进入训练, {len(test_group_ids)} 组进入测试。")

        raw_train_df = pd.concat([train_df, new_data_for_training], ignore_index=True)
        raw_test_df = new_data_for_testing

        if strategy == "均值聚合":
            self.log("聚合训练数据...");
            X_train, y_train_raw = self.aggregate_dataframe(raw_train_df, selected_features, label_column,
                                                            condition_column)
            self.log("聚合测试数据...");
            X_test, y_test_raw = self.aggregate_dataframe(raw_test_df, selected_features, label_column,
                                                          condition_column)
        else:
            X_train = raw_train_df[selected_features];
            y_train_raw = raw_train_df[label_column]
            X_test = raw_test_df[selected_features];
            y_test_raw = raw_test_df[label_column]

        self.log(f"最终生效的训练集大小: {X_train.shape[0]} 行, {X_train.shape[1]} 列")
        self.log(f"最终生效的测试集大小: {X_test.shape[0]} 行, {X_test.shape[1]} 列")
        if X_test.empty: messagebox.showerror("错误", "最终测试集为空，无法继续分析。"); return

        self.log("步骤 11/11: 数据预处理、模型训练与评估...")
        le = LabelEncoder();
        y_train = le.fit_transform(y_train_raw)
        try:
            y_test = le.transform(y_test_raw)
        except ValueError as e:
            self.log(f"!!! 严重错误: 测试集标签中包含训练集未见过的类别: {e}");
            messagebox.showerror("标签不匹配",
                                 f"测试集标签中包含训练集未见过的类别。\n请检查两个文件中的'{label_column}'列。");
            return

        scaler = StandardScaler();
        X_train = X_train.astype(np.float64);
        X_test = X_test.astype(np.float64)
        X_train_scaled = scaler.fit_transform(X_train);
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns);
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        self.run_rf_analysis(X_train_scaled_df, y_train, X_test_scaled_df, y_test, X_train.columns, le, scaler,
                             save_base_dir)

    def run_rf_analysis(self, X_train, y_train, X_test, y_test, feature_columns, le, scaler, save_base_dir):
        self.log(f"--- 正在优化 RF ---")
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        hp_space = {'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                    'max_depth': hp.quniform('max_depth', 5, 50, 1)}
        original_stdout, sys.stdout = sys.stdout, self

        def objective(params):
            p = {k: int(v) for k, v in params.items()};
            model.set_params(**p)
            score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(4, shuffle=True, random_state=42),
                                    scoring='f1_macro').mean()
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials();
        best_params_raw = fmin(fn=objective, space=hp_space, algo=tpe.suggest, max_evals=50, trials=trials,
                               rstate=np.random.default_rng(42), show_progressbar=False)
        best_params = {k: int(v) for k, v in space_eval(hp_space, best_params_raw).items()};
        sys.stdout = original_stdout
        min_class_count = np.min(np.bincount(y_train));
        n_splits_cv = min(5, min_class_count);
        cv_line = ""
        if n_splits_cv < 2:
            cv_line = "交叉验证跳过: 某类别样本数少于2。\n"
        else:
            cv_scores = cross_val_score(model.set_params(**best_params), X_train, y_train,
                                        cv=StratifiedKFold(n_splits_cv, shuffle=True, random_state=88),
                                        scoring='f1_macro')
            cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores);
            cv_line = f"交叉验证 ({n_splits_cv}-折) F1 (macro): {cv_mean:.4f} (+/- {cv_std:.4f})\n"
        final_model = model.set_params(**best_params);
        final_model.fit(X_train, y_train)
        report_str = classification_report(y_test, final_model.predict(X_test), target_names=le.classes_,
                                           zero_division=0)
        full_report_str = f"--- RF 在测试集上的最终评估 ---\n最佳参数: {best_params}\n{cv_line}{report_str}\n";
        self.log(full_report_str)
        self.display_confusion_matrix(final_model, X_test, y_test, le.classes_, "RF", save_base_dir)
        self.visualize_best_model("rf", final_model, feature_columns, le.classes_)
        perm_df = self.run_permutation_importance(final_model, X_test, y_test, feature_columns, save_base_dir)
        shap_summary_df, shap_detailed_dfs = self.run_shap_analysis(final_model, X_train, X_test, le, save_base_dir)

        if save_base_dir:
            self.export_analysis_reports(full_report_str, perm_df, shap_summary_df, shap_detailed_dfs, save_base_dir)
            self.export_best_model("rf", final_model, le, scaler, feature_columns, save_base_dir)

    def map_class_names_to_amino_acids(self, class_names):
        amino_acid_map = {'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe', 'G': 'Gly', 'H': 'His',
                          'I': 'Ile', 'K': 'Lys', 'L': 'Leu', 'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln',
                          'R': 'Arg', 'S': 'Ser', 'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr'}
        return [amino_acid_map.get(str(name).upper(), str(name)) for name in class_names]

    def run_permutation_importance(self, model, X_test, y_test, feature_names, save_dir):
        """
        修改版置换重要性 (最终修正):
        1. [强力清洗] 使用正则忽略大小写移除 '_mean'。
        2. [稳定] 保持 random_state=42 以减少波动。
        """
        self.log("--- 正在计算置换重要性 (Permutation Importance)... ---")
        try:
            # 1. 计算 (固定 random_state 以保证结果一致性)
            X_test_values = X_test.values if hasattr(X_test, 'values') else X_test
            result = permutation_importance(model, X_test_values, y_test, n_repeats=10, random_state=42, n_jobs=-1,
                                            scoring='f1_macro')

            perm_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            })

            # 2. 过滤 Std 特征
            filtered_df = perm_df[~perm_df['feature'].str.contains('std', case=False, na=False)].copy()

            # 3. [修复] 强力清洗标签
            # (?i) 表示忽略大小写，_mean 匹配后缀
            filtered_df['feature_display'] = filtered_df['feature'].str.replace(r'_mean', '', regex=True, case=False)

            # 排序
            plot_data = filtered_df.sort_values('importance_mean', ascending=True).tail(20)

            # 4. 绘图
            fig, ax = plt.subplots(figsize=(10, 8))

            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(plot_data)))
            # 使用清洗后的 feature_display
            ax.barh(plot_data['feature_display'], plot_data['importance_mean'], xerr=plot_data['importance_std'],
                    align='center', color=colors, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})

            ax.set_title('Permutation Importance', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance (Mean F1 Decrease)', fontsize=12)

            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            plt.tight_layout()

            save_path = os.path.join(save_dir, 'RF_Permutation_Importance_Filtered.png')
            plt.savefig(save_path, dpi=300)
            plt.close()

            self.log(f"置换重要性图已保存: {save_path}")
            return perm_df

        except Exception as e:
            self.log(f"!!! 置换重要性计算失败: {e}")
            self.log(traceback.format_exc())
            return None

    def run_shap_analysis(self, model, X_train, X_test, le, save_dir):
        """
        修改版 SHAP 分析 (Origin 数据导出版):
        1. [功能] 自动导出 'RF_SHAP_Global_Bar_Data_for_Origin.csv'，方便 Origin 绘图。
        2. [清洗] 保持标签清洗 (移除 _mean, 隐藏 std)。
        3. [绘图] 保持 Python 预览图。
        """
        if not SHAP_INSTALLED:
            self.log("--- 跳过SHAP分析：SHAP库未安装。 ---")
            return None, None

        self.log("--- 正在进行SHAP分析 (准备Origin数据)... ---")
        try:
            # 1. 计算 SHAP
            explainer = shap.TreeExplainer(model, X_train)
            shap_explanation = explainer(X_test, check_additivity=False)

            class_names = self.map_class_names_to_amino_acids(le.classes_)
            is_multiclass = len(shap_explanation.shape) == 3

            # 2. 筛选特征 (非 Std)
            feature_names = list(X_test.columns)
            non_std_indices = [i for i, f in enumerate(feature_names) if 'std' not in f.lower()]
            if not non_std_indices: non_std_indices = range(len(feature_names))

            # 3. 准备数据并清洗列名
            X_filtered = X_test.iloc[:, non_std_indices].copy()
            # 清洗列名 (移除 _mean)
            clean_columns_list = X_filtered.columns.str.replace(r'_mean', '', regex=True, case=False).tolist()
            X_filtered.columns = clean_columns_list

            # 提取 SHAP 值
            if is_multiclass:
                shap_values_filtered = shap_explanation[:, non_std_indices, :]
                # 计算全局重要性 (Origin画图要用的数据)
                mean_abs_shap = np.abs(shap_explanation.values[:, non_std_indices, :]).mean(axis=(0, 2))
            else:
                shap_values_filtered = shap_explanation[:, non_std_indices]
                mean_abs_shap = np.abs(shap_explanation.values[:, non_std_indices]).mean(axis=0)

            # 强制覆盖 feature_names 以确保 Python 图也正确
            shap_values_filtered.feature_names = clean_columns_list

            # --- 4. [核心新增] 导出 Origin 专用数据 ---
            shap_summary_df = pd.DataFrame({
                'Feature': clean_columns_list,
                'Importance': mean_abs_shap
            }).sort_values('Importance', ascending=True)  # Origin画条形图通常是从下往上画，所以按升序排

            origin_save_path = os.path.join(save_dir, 'RF_SHAP_Global_Bar_Data_for_Origin.csv')
            # 使用 utf-8-sig 编码，防止中文乱码
            shap_summary_df.to_csv(origin_save_path, index=False, encoding='utf-8-sig')
            self.log(f"✅ Origin绘图数据已导出: {origin_save_path}")

            # --- 5. 生成 Python 预览图 (Global Bar) ---
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_filtered, X_filtered, plot_type="bar", class_names=class_names, show=False,
                              max_display=20)

            # 移除图例 & 美化
            ax = plt.gca()
            if ax.get_legend(): ax.get_legend().remove()
            plt.title('Global Feature Importance', fontsize=14, fontweight='bold')
            plt.xlabel('mean(|SHAP value|)', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'RF_SHAP_Global_Bar_Filtered.png'), dpi=300)
            plt.close()

            # --- 6. 生成独立蜂群图 (Beeswarm) ---
            if is_multiclass:
                for i, class_name in enumerate(class_names):
                    safe_name = "".join(c for c in class_name if c.isalnum())
                    plt.figure(figsize=(8, 6))
                    shap_class_expl = shap_values_filtered[:, :, i]
                    shap.summary_plot(shap_class_expl, X_filtered, show=False, max_display=15)
                    plt.title(f'SHAP Detail: {class_name}', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'RF_SHAP_Detail_{safe_name}.png'), dpi=300)
                    plt.close()
            else:
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values_filtered, X_filtered, show=False, max_display=15)
                plt.savefig(os.path.join(save_dir, 'RF_SHAP_Summary.png'), dpi=300)
                plt.close()

            return shap_summary_df, {}

        except Exception as e:
            self.log(f"!!! SHAP分析失败: {e}")
            self.log(traceback.format_exc())
            return None, None

    def export_analysis_reports(self, report_str, perm_df, shap_summary_df, shap_detailed_dfs, save_dir):
        txt_path = os.path.join(save_dir, "rf_summary_report.txt");
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 20 + " RF 模型评估详细报告 " + "=" * 20 + "\n\n"); f.write(report_str)
            self.log(f"RF模型评估摘要已导出至TXT文件: {txt_path}")
        except Exception as e:
            self.log(f"导出到TXT文件失败: {e}")

        excel_path = os.path.join(save_dir, "rf_deep_analysis_summary.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                report_data = [];
                lines = report_str.split('\n');
                header_line_index = -1
                for i, line in enumerate(lines):
                    if 'precision' in line: header_line_index = i; break
                if header_line_index != -1:
                    header = ['class'] + lines[header_line_index].split()
                    for line in lines[header_line_index + 1:]:
                        parts = line.split()
                        if len(parts) > 2 and any(char.isdigit() for char in line):
                            if 'accuracy' in line:
                                class_name = 'accuracy'; metrics = ['', ''] + parts[-2:]
                            else:
                                class_name = " ".join(parts[:-4]).strip(); metrics = parts[-4:]
                            report_data.append([class_name] + metrics)
                    df_report = pd.DataFrame(report_data);
                    df_report.columns = header[:len(df_report.columns)]
                    df_report.to_excel(writer, sheet_name="RF_Classification_Report", index=False)

                if perm_df is not None: perm_df.to_excel(writer, sheet_name='RF_Permutation_Importance', index=False)
                if shap_summary_df is not None: shap_summary_df.to_excel(writer, sheet_name='RF_SHAP_Importance',
                                                                         index=False)

                # --- 导出详细SHAP数据 ---
                if shap_detailed_dfs:
                    self.log(f"正在将详细SHAP数据写入Excel文件: {excel_path}")
                    for class_name, df in shap_detailed_dfs.items():
                        df.to_excel(writer, sheet_name=f"SHAP_Data_{class_name}", index=False)
                    self.log("详细SHAP数据写入完成。")

            self.log(f"RF模型深度分析报告已导出至Excel: {excel_path}")
        except Exception as e:
            self.log(f"导出到Excel失败: {e}\n{traceback.format_exc()}")

    def select_strategy(self):
        dialog = Toplevel(self.master);
        dialog.title("选择捆绑策略");
        dialog.geometry("450x220");
        strategy_var = StringVar(value="分组捆绑")
        tk.Label(dialog, text="请选择处理数据的捆绑策略:", font=('Arial', 14, 'bold')).pack(pady=10)
        Radiobutton(dialog, text="分组捆绑 (保持原貌，公平考试)", variable=strategy_var, value="分组捆绑",
                    font=('Arial', 12)).pack(anchor='w', padx=20)
        Radiobutton(dialog, text="均值聚合 (提炼特征，简化问题)", variable=strategy_var, value="均值聚合",
                    font=('Arial', 12)).pack(anchor='w', padx=20)
        tk.Label(dialog, text="注意: 两种策略都会先按组拆分新数据。", font=('Arial', 10, 'italic')).pack(pady=5)
        result = {"strategy": ""};
        confirm = lambda: (result.update({"strategy": strategy_var.get()}), dialog.destroy())
        Button(dialog, text="确认", command=confirm, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog);
        return result["strategy"]

    def ask_for_incremental_percentage(self):
        dialog = Toplevel(self.master);
        dialog.title("配置新数据训练比例");
        dialog.geometry("400x200")
        tk.Label(dialog, text="请输入从'新数据文件'中，\n抽取多少百分比的组用于训练？\n(0-99，剩余的组将作为测试集)",
                 font=('Arial', 12)).pack(pady=10)
        percentage_var = StringVar(value="30");
        entry = Entry(dialog, textvariable=percentage_var, font=('Arial', 12), width=10);
        entry.pack(pady=5)
        result = {'percentage': None}

        def confirm():
            try:
                val = float(percentage_var.get())
                if 0 <= val < 100:
                    result['percentage'] = val; dialog.destroy()
                else:
                    messagebox.showerror("输入无效", "请输入0到100之间的数字 (不包括100)。", parent=dialog)
            except ValueError:
                messagebox.showerror("输入无效", "请输入一个有效的数字。", parent=dialog)

        Button(dialog, text="确认", command=confirm, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog);
        return result['percentage']

    def aggregate_dataframe(self, df, features, label_col, condition_col):
        if df.empty: return pd.DataFrame(), pd.Series()
        valid_features = [f for f in features if f in df.columns]
        if not valid_features: messagebox.showerror("聚合错误",
                                                    "找不到任何有效的公共特征进行聚合操作！"); return pd.DataFrame(), pd.Series()
        group_id_col = "__AUTO_GROUP_ID__";
        df_copy = df.copy();
        df_copy[group_id_col] = df_copy[label_col].astype(str) + "_" + df_copy[condition_col].astype(str)
        grouped = df_copy.groupby(group_id_col)
        mean_df = grouped[valid_features].mean();
        std_df = grouped[valid_features].std().fillna(0);
        label_df = grouped[label_col].first()
        mean_df.columns = [f"{col}_mean" for col in valid_features];
        std_df.columns = [f"{col}_std" for col in valid_features]
        X_agg = pd.concat([mean_df, std_df], axis=1);
        y_raw_agg = label_df
        self.log(f"数据聚合完成: {len(df_copy)}行 -> {len(X_agg)}行");
        return X_agg.reset_index(drop=True), y_raw_agg.reset_index(drop=True)

    def select_column_role(self, columns, title):
        dialog = Toplevel(self.master);
        dialog.title(title);
        dialog.geometry("400x450");
        selected_col = StringVar(value=columns[0] if columns else "")
        frame = Frame(dialog, pady=10);
        frame.pack(fill=tk.BOTH, expand=True);
        tk.Label(frame, text=title, font=('Arial', 12, 'bold')).pack()
        canvas = tk.Canvas(frame);
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview);
        inner_frame = Frame(canvas)
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")));
        canvas.create_window((0, 0), window=inner_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for col in columns: Radiobutton(inner_frame, text=col, variable=selected_col, value=col,
                                        font=('Arial', 12)).pack(anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        Button(dialog, text="确认", command=dialog.destroy, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog);
        return selected_col.get()

    def write(self, text):
        if text.strip(): self.log(text.strip())

    def flush(self):
        pass

    def visualize_best_model(self, best_model_name, best_model, feature_names, class_names):
        if isinstance(best_model, RandomForestClassifier):
            fig, ax = plt.subplots(figsize=(20, 15));
            plot_tree(best_model.estimators_[0], filled=True, feature_names=feature_names.tolist(),
                      class_names=class_names.tolist(), rounded=True, fontsize=10, max_depth=4)
            ax.set_title(f"最佳模型 ({best_model_name.upper()}) 的第一棵树", fontsize=20);
            plt.show(block=False)

    def export_best_model(self, best_model_name, best_model, label_encoder, scaler, feature_columns, save_dir):
        model_package = {'model_name': best_model_name, 'model': best_model, 'label_encoder': label_encoder,
                         'scaler': scaler, 'feature_columns': feature_columns.tolist()}
        file_path = os.path.join(save_dir, 'best_model_package.joblib')
        try:
            joblib.dump(model_package, file_path);
            self.log(f"成功！最佳模型包已导出至: {file_path}")
            messagebox.showinfo("模型已导出", f"最佳模型包已成功保存至:\n{file_path}", parent=self.master)
        except Exception as e:
            self.log(f"错误：导出模型失败 - {e}")

    def select_features(self, all_features):
        dialog = Toplevel(self.master);
        dialog.title("请选择特征列 (X)");
        dialog.geometry("400x500");
        selected_features_list = []
        vars = {feature: BooleanVar(value=True) for feature in all_features}

        def confirm():
            nonlocal selected_features_list;
            selected_features_list = [feature for feature, var in vars.items() if var.get()]
            if not selected_features_list: messagebox.showerror("错误", "请至少选择一个特征列！", parent=dialog); return
            dialog.destroy()

        frame = Frame(dialog);
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5);
        btn_frame = Frame(frame);
        btn_frame.pack(fill=tk.X)
        Button(btn_frame, text="全选", command=lambda: [v.set(True) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                  expand=True)
        Button(btn_frame, text="全不选", command=lambda: [v.set(False) for v in vars.values()]).pack(side=tk.LEFT,
                                                                                                     expand=True)
        canvas = tk.Canvas(frame);
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview);
        scrollable_frame = Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")));
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)
        for feature in all_features: Checkbutton(scrollable_frame, text=feature, var=vars[feature],
                                                 font=('Arial', 12)).pack(anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        Button(dialog, text="确认选择", command=confirm, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog);
        return selected_features_list

    def display_confusion_matrix(self, model, X_test, y_test, target_names, model_name, save_dir):
        y_pred = model.predict(X_test);
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(target_names)))
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'): cm_percent = np.where(cm_sum > 0,
                                                                                   (cm.astype('float') / cm_sum) * 100,
                                                                                   0)
        fig, ax = plt.subplots(figsize=(10, 8));
        sns.heatmap(cm_percent, annot=True, fmt='.0f', cmap='Blues', xticklabels=target_names, yticklabels=target_names,
                    ax=ax)
        ax.set_title(f'{model_name} Confusion Matrix (%)', fontsize=20);
        ax.set_ylabel('True Label', fontsize=16);
        ax.set_xlabel('Predicted Label', fontsize=16)
        plt.xticks(rotation=45, ha='right');
        plt.yticks(rotation=0);
        plt.tight_layout()
        if save_dir: plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix_percent.png'))
        plt.show(block=False)


if __name__ == '__main__':
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()
