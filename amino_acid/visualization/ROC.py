import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
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
from itertools import cycle

# --- Matplotlib全局设置 ---
# Matplotlib Global Settings
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


class AnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("RF模型分析与ROC曲线工具 (V4-数据驱动版)")
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

        self.log("欢迎使用RF模型分析与ROC曲线工具！")
        self.log("V4版: 已根据您的数据格式，修正ROC分类绘图逻辑。")

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
        # ... (此部分代码与之前版本相同) ...
        self.log("步骤 1/9: 请选择 **基础训练** 数据文件...")
        train_path = filedialog.askopenfilename(parent=self.master, title="选择 **基础训练** 数据CSV文件",
                                                filetypes=[("CSV files", "*.csv")])
        if not train_path: self.log("操作取消。"); return

        self.log("步骤 2/9: 请选择 **新数据/预测** 数据文件...")
        predict_path = filedialog.askopenfilename(parent=self.master, title="选择 **新数据/预测** 数据CSV文件",
                                                  filetypes=[("CSV files", "*.csv")])
        if not predict_path: self.log("操作取消。"); return

        self.log("步骤 3/9: 请选择结果保存目录...")
        save_base_dir = filedialog.askdirectory(parent=self.master, title="选择结果保存目录")
        if not save_base_dir: self.log("操作取消。"); return

        self.log("步骤 4/9: 正在加载数据...")
        train_df = pd.read_csv(train_path)
        predict_df = pd.read_csv(predict_path)

        self.log("步骤 5/9: (关键) 请选择处理数据的捆绑策略...")
        strategy = self.select_strategy()
        if not strategy: self.log("操作取消。"); return

        self.log("步骤 6/9: (关键) 请配置新数据用于训练的比例...")
        percentage_to_train = self.ask_for_incremental_percentage()
        if percentage_to_train is None: self.log("操作取消。"); return

        self.log("步骤 7/9: 请指定主要类别列、组内条件列和特征列...")
        label_column = self.select_column_role(train_df.columns.tolist(), "请选择主要类别列 (预测目标Y)")
        if not label_column: self.log("操作取消。"); return
        condition_column = self.select_column_role(train_df.columns.tolist(),
                                                   "请选择 **唯一的** 组内条件列 (定义平行样)")
        if not condition_column: self.log("操作取消。"); return

        common_cols = list(set(train_df.columns) & set(predict_df.columns))
        cols_to_exclude = [label_column, condition_column]
        all_features = [col for col in common_cols if col not in cols_to_exclude]
        selected_features = self.select_features(all_features)
        if not selected_features: self.log("操作取消。"); return

        self.log("步骤 8/9: 正在重构数据流并应用策略...")
        predict_df_copy = predict_df.copy()
        group_id_col = "__AUTO_GROUP_ID__"
        predict_df_copy[group_id_col] = predict_df_copy[label_column].astype(str) + "_" + predict_df_copy[
            condition_column].astype(str)

        if percentage_to_train == 0:
            new_data_for_training, new_data_for_testing = pd.DataFrame(), predict_df_copy
        else:
            unique_groups_df = predict_df_copy.drop_duplicates(subset=[group_id_col])
            group_ids, group_labels = unique_groups_df[group_id_col], unique_groups_df[label_column]
            stratify_param = group_labels if group_labels.value_counts().min() >= 2 else None
            test_size_prop = 1.0 - (percentage_to_train / 100.0)
            train_group_ids, test_group_ids = train_test_split(group_ids, test_size=test_size_prop, random_state=42,
                                                               stratify=stratify_param)
            new_data_for_training = predict_df_copy[predict_df_copy[group_id_col].isin(train_group_ids)]
            new_data_for_testing = predict_df_copy[predict_df_copy[group_id_col].isin(test_group_ids)]

        raw_train_df = pd.concat([train_df, new_data_for_training], ignore_index=True)
        raw_test_df = new_data_for_testing.copy()  # 使用 .copy() 避免后续操作影响原始数据

        if strategy == "均值聚合":
            X_train, y_train_raw = self.aggregate_dataframe(raw_train_df, selected_features, label_column,
                                                            condition_column)
            X_test, y_test_raw = self.aggregate_dataframe(raw_test_df, selected_features, label_column,
                                                          condition_column)
        else:
            X_train, y_train_raw = raw_train_df[selected_features], raw_train_df[label_column]
            X_test, y_test_raw = raw_test_df[selected_features], raw_test_df[label_column]

        if X_test.empty: messagebox.showerror("错误", "最终测试集为空。"); return

        self.log("步骤 9/9: 数据预处理、RF模型训练与评估...")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_test = le.transform(y_test_raw)

        # 将原始测试标签（字符串）保存为DataFrame，以便后续按索引筛选
        y_test_raw_df = pd.DataFrame({'label': y_test_raw, 'encoded': y_test}, index=y_test_raw.index)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.run_rf_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, y_test_raw_df, X_train.columns, le,
                               scaler, save_base_dir)

    def run_rf_evaluation(self, X_train, y_train, X_test, y_test, y_test_raw_df, feature_columns, le, scaler,
                          save_base_dir):
        # ... (此部分代码与之前版本相同) ...
        self.log("--- 正在为随机森林(RF)模型优化超参数 ---")
        original_stdout = sys.stdout;
        sys.stdout = self

        hp_space_rf = {'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                       'max_depth': hp.quniform('max_depth', 5, 50, 1),
                       'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)}

        def objective(params):
            p = {k: int(v) for k, v in params.items()}
            model = RandomForestClassifier(random_state=42, class_weight='balanced', **p)
            score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(4, shuffle=True, random_state=42),
                                    scoring='f1_macro').mean()
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best_params_raw = fmin(fn=objective, space=hp_space_rf, algo=tpe.suggest, max_evals=50, trials=trials,
                               rstate=np.random.default_rng(42), show_progressbar=False)
        best_params = {k: int(v) for k, v in space_eval(hp_space_rf, best_params_raw).items()}
        self.log(f"找到的最佳参数: {best_params}")

        final_model = RandomForestClassifier(random_state=42, class_weight='balanced', **best_params)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        report_str = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
        self.log("\n--- RF 在测试集上的最终评估 ---\n" + report_str)

        self.display_confusion_matrix(final_model, X_test, y_test, le.classes_, "RandomForest", save_base_dir)
        self.display_feature_importance(final_model, feature_columns, "RandomForest")

        self.log("\n--- 正在生成ROC曲线 ---")
        self.plot_roc_curves(final_model, X_test, y_test_raw_df, le, best_params, save_base_dir)

        self.export_report_to_txt(report_str, best_params, save_base_dir)
        self.export_best_model("RandomForest", final_model, le, scaler, feature_columns, save_base_dir)
        sys.stdout = original_stdout
        self.log("RF模型评估完成。")

    def _plot_multiclass_roc(self, model, X_test_subset, y_test_subset, class_names_subset, title, filepath):
        """一个绘制多分类ROC曲线并保存的辅助函数（V3样式）"""
        try:
            # 预测概率
            y_score = model.predict_proba(X_test_subset)
            n_classes = len(class_names_subset)

            # 创建一个本地的LabelEncoder来处理子集内的标签
            le_subset = LabelEncoder().fit(y_test_subset)
            y_test_encoded = le_subset.transform(y_test_subset)

            # 确保y_score的列数与子集类别数匹配
            if y_score.shape[1] < n_classes:
                # 这种情况理论上不应发生，但作为安全检查
                self.log(f"警告: 预测概率的列数 ({y_score.shape[1]}) 少于类别数 ({n_classes}) in {title}")
                return

            fpr, tpr, roc_auc = dict(), dict(), dict()
            for i in range(n_classes):
                # 使用one-hot编码的真实标签和对应的预测概率列
                fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test_encoded))[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            all_y_test_dummies = pd.get_dummies(y_test_encoded).values.ravel()
            all_y_score = y_score.ravel()
            fpr["micro"], tpr["micro"], _ = roc_curve(all_y_test_dummies, all_y_score)
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Plotting
            plt.figure(figsize=(10, 8));
            ax = plt.gca()
            for spine in ax.spines.values(): spine.set_linewidth(2)

            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                     color='deeppink', linestyle='-', linewidth=4)
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'olive', 'brown', 'pink'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_names_subset[i]} (AUC = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=2)
            plt.xlabel('1 - Specificity', fontsize=22, fontweight='bold')
            plt.ylabel('Sensitivity', fontsize=22, fontweight='bold')
            plt.title(title, fontsize=20, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.xlim([0.0, 1.0]);
            plt.ylim([0.0, 1.05])
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(False);
            plt.tight_layout();
            plt.savefig(filepath);
            plt.close()
            self.log(f"成功保存ROC曲线图: {filepath}")

        except Exception as e:
            self.log(f"绘制ROC曲线时出错 ({title}): {e}");
            traceback.print_exc()

    def plot_roc_curves(self, model, X_test, y_test_raw_df, le, best_params, save_dir):
        """绘制总ROC曲线和按氨基酸类别划分的ROC曲线（V4: 数据驱动逻辑）"""
        # 1. 绘制使用所有特征的总ROC图
        self._plot_multiclass_roc(model, X_test, y_test_raw_df['encoded'].values, le.classes_,
                                  'ROC Curve - All Features', os.path.join(save_dir, 'ROC_Curve_All_Features.png'))

        # 2. 根据氨基酸分类绘制ROC图
        amino_acid_categories = {
            "Nonpolar_Aliphatic": ["Gly", "Ala", "Val", "Leu", "Ile", "Met", "Pro"],
            "Aromatic": ["Phe", "Tyr", "Trp"],
            "Polar_Uncharged": ["Ser", "Thr", "Cys"],
            "Positively_Charged": ["Lys", "Arg", "His"],
            "Negatively_Charged": ["Asp", "Glu"]
        }

        for category, aa_list in amino_acid_categories.items():
            self.log(f"  - 为 '{category}' 类别特征生成ROC图...")

            # V4核心逻辑: 筛选数据行，而不是特征列
            # 智能匹配 'L-Val', 'D-Val' 等标签
            mask = y_test_raw_df['label'].apply(lambda x: any(aa in x for aa in aa_list))

            # 获取子集数据
            y_subset_df = y_test_raw_df[mask]

            # 检查子集是否有效（至少有两个类别）
            if y_subset_df['label'].nunique() < 2:
                self.log(f"    警告: '{category}' 类别的数据不足或只包含一个类别，无法生成ROC图，跳过。")
                continue

            # 从主测试集中按索引筛选出对应的X
            X_test_subset = X_test[y_subset_df.index]
            y_test_subset_encoded = y_subset_df['encoded'].values

            # 获取子集中的类别名称
            class_names_subset = np.unique(y_subset_df['label'])

            # 使用全局模型为这个子集创建独立的ROC图
            self._plot_multiclass_roc(model, X_test_subset, y_test_subset_encoded, class_names_subset,
                                      f'ROC Curve - {category}',
                                      os.path.join(save_dir, f'ROC_Curve_{category}.png'))

    # --- 以下是窗口交互和辅助函数，与V3版本相同 ---
    def export_report_to_txt(self, report_str, best_params, save_dir):
        txt_path = os.path.join(save_dir, "RandomForest_summary_report.txt")
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 20 + " 随机森林(RF)模型评估报告 " + "=" * 20 + "\n\n")
                f.write(f"最佳参数: {best_params}\n\n")
                f.write("--- 测试集分类报告 ---\n");
                f.write(report_str)
            self.log(f"模型评估摘要已导出至TXT文件: {txt_path}")
        except Exception as e:
            self.log(f"导出到TXT文件失败: {e}")

    def select_strategy(self):
        dialog = Toplevel(self.master);
        dialog.title("选择捆绑策略");
        dialog.geometry("450x220")
        strategy_var = StringVar(value="分组捆绑")
        tk.Label(dialog, text="请选择处理数据的捆绑策略:", font=('Arial', 14, 'bold')).pack(pady=10)
        Radiobutton(dialog, text="分组捆绑 (保持原貌)", variable=strategy_var, value="分组捆绑",
                    font=('Arial', 12)).pack(anchor='w', padx=20)
        Radiobutton(dialog, text="均值聚合 (提炼特征)", variable=strategy_var, value="均值聚合",
                    font=('Arial', 12)).pack(anchor='w', padx=20)
        result = {"strategy": ""};
        confirm = lambda: (result.update(strategy=strategy_var.get()), dialog.destroy())
        Button(dialog, text="确认", command=confirm, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return result["strategy"]

    def ask_for_incremental_percentage(self):
        dialog = Toplevel(self.master);
        dialog.title("配置新数据训练比例");
        dialog.geometry("400x200")
        tk.Label(dialog, text="请输入新数据用于训练的组的百分比(0-99)\n(剩余的组将作为测试集)",
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
                    messagebox.showerror("输入无效", "请输入一个0到99之间的数字。", parent=dialog)
            except ValueError:
                messagebox.showerror("输入无效", "请输入一个有效的数字。", parent=dialog)

        Button(dialog, text="确认", command=confirm, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return result['percentage']

    def aggregate_dataframe(self, df, features, label_col, condition_col):
        if df.empty: return pd.DataFrame(), pd.Series()
        group_id_col = "__AUTO_GROUP_ID__"
        df_copy = df.copy();
        df_copy[group_id_col] = df_copy[label_col].astype(str) + "_" + df_copy[condition_col].astype(str)
        grouped = df_copy.groupby(group_id_col)
        mean_df = grouped[features].mean();
        std_df = grouped[features].std().fillna(0);
        label_df = grouped[label_col].first()
        mean_df.columns = [f"{col}_mean" for col in features];
        std_df.columns = [f"{col}_std" for col in features]
        X_agg = pd.concat([mean_df, std_df], axis=1)
        return X_agg.reset_index(drop=True), label_df.reset_index(drop=True)

    def select_column_role(self, columns, title):
        dialog = Toplevel(self.master);
        dialog.title(title);
        dialog.geometry("400x450")
        selected_col = StringVar(value=columns[0] if columns else "")
        frame = Frame(dialog);
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=title, font=('Arial', 12, 'bold')).pack(pady=10)
        for col in columns: Radiobutton(frame, text=col, variable=selected_col, value=col, font=('Arial', 12)).pack(
            anchor='w', padx=20)
        Button(dialog, text="确认", command=dialog.destroy, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_col.get()

    def write(self, text):
        if text.strip(): self.log(text.strip())

    def flush(self):
        pass

    def export_best_model(self, model_name, model, label_encoder, scaler, feature_columns, save_dir):
        model_package = {'model_name': model_name, 'model': model, 'label_encoder': label_encoder, 'scaler': scaler,
                         'feature_columns': feature_columns.tolist()}
        file_path = os.path.join(save_dir, 'best_model_package.joblib')
        try:
            joblib.dump(model_package, file_path)
            self.log(f"成功！最佳模型包已导出至: {file_path}")
            messagebox.showinfo("模型已导出", f"最佳模型包已成功保存至:\n{file_path}", parent=self.master)
        except Exception as e:
            self.log(f"错误：导出模型失败 - {e}")

    def select_features(self, all_features):
        dialog = Toplevel(self.master);
        dialog.title("请选择特征列 (X)");
        dialog.geometry("400x500")
        selected_features_list = [];
        vars = {feature: BooleanVar(value=True) for feature in all_features}

        def confirm():
            nonlocal selected_features_list
            selected_features_list = [f for f, v in vars.items() if v.get()]
            if not selected_features_list: messagebox.showerror("错误", "请至少选择一个特征列！", parent=dialog); return
            dialog.destroy()

        Button(dialog, text="全选", command=lambda: [v.set(True) for v in vars.values()]).pack(side=tk.LEFT)
        Button(dialog, text="全不选", command=lambda: [v.set(False) for v in vars.values()]).pack(side=tk.LEFT)
        frame = Frame(dialog);
        frame.pack(fill="both", expand=True)
        for feature in all_features: Checkbutton(frame, text=feature, var=vars[feature], font=('Arial', 12)).pack(
            anchor='w', padx=20)
        Button(dialog, text="确认选择", command=confirm, font=('Arial', 12, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_features_list

    def display_confusion_matrix(self, model, X_test, y_test, target_names, model_name, save_dir):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(target_names)))
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'): cm_percent = np.where(cm_sum > 0,
                                                                                   (cm.astype('float') / cm_sum) * 100,
                                                                                   0)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_percent, annot=True, fmt='.0f', cmap='Blues', xticklabels=target_names, yticklabels=target_names,
                    ax=ax)
        ax.set_title(f'{model_name} Confusion Matrix (%)', fontsize=20)
        ax.set_ylabel('True Label', fontsize=16);
        ax.set_xlabel('Predicted Label', fontsize=16)
        plt.tight_layout();
        plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'));
        plt.close(fig)
        self.log(f"混淆矩阵图已保存至: {os.path.join(save_dir, f'{model_name}_confusion_matrix.png')}")

    def display_feature_importance(self, model, columns, model_name):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            df_imp = pd.DataFrame({'feature': columns, 'importance': importances}).sort_values('importance',
                                                                                               ascending=False).head(20)
            self.log(f"\n--- {model_name} Top 20 Feature Importances ---");
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
