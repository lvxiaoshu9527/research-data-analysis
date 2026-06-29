import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Checkbutton, Button, BooleanVar, Frame, Radiobutton, StringVar, \
    Text, Scrollbar, END
import os
import matplotlib as mpl
import traceback
import sys
import threading

# --- Matplotlib全局设置 ---
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


class AnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("分类工具 (4v2 自动分组版)")
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

        self.log("欢迎使用 4v2 自动分组验证工具！")
        self.log("流程: 自动按类别4:2分组 -> 交叉验证 -> 生成混淆矩阵和分数汇总。")

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
            messagebox.showerror("文件读取错误", f"无法读取文件: {e}");
            self.log(f"文件读取错误: {e}");
            return

        label_column = self.select_column_role(df_for_ui.columns.tolist(), "请选择类别列 (如 氨基酸)")
        if not label_column: self.log("操作取消。"); return

        condition_column = self.select_column_role(df_for_ui.columns.tolist(), "请选择分组条件列 (如 浓度)")
        if not condition_column: self.log("操作取消。"); return

        all_features = [col for col in df_for_ui.columns if col not in [label_column, condition_column]]
        selected_features = self.select_features(all_features)
        if not selected_features: self.log("操作取消。"); return

        params = {'file_path': file_path, 'save_base_dir': save_base_dir, 'label_column': label_column,
                  'condition_column': condition_column, 'selected_features': selected_features}

        self.start_button.config(state="disabled", text="分析中...")
        self.log("\n" + "=" * 50);
        self.log("用户输入完成，开始后台分析任务...")
        analysis_thread = threading.Thread(target=self.run_analysis_wrapper, args=(params,));
        analysis_thread.daemon = True;
        analysis_thread.start()

    def run_analysis_wrapper(self, params):
        try:
            self.run_main_analysis(params)
        except Exception as e:
            error_details = traceback.format_exc()
            self.log(f"!!! 发生严重错误 !!!\n错误类型: {type(e).__name__}\n错误信息: {e}");
            self.log(f"详细追溯信息:\n{error_details}")
            self.master.after(0, lambda: messagebox.showerror("程序遇到意外错误",
                                                              f"发生未处理的错误: {e}\n\n请查看主窗口日志获取详细信息。"))
        finally:
            self.master.after(0, lambda: self.start_button.config(state="normal", text="开始分析"));
            self.log("分析任务结束。")

    # --- 主分析流程 ---
    def run_main_analysis(self, params):
        file_path = params['file_path'];
        save_base_dir = params['save_base_dir']
        label_column = params['label_column'];
        condition_column = params['condition_column']
        selected_features = params['selected_features']

        self.log("步骤 1: 正在加载数据...");
        df = pd.read_csv(file_path)

        # 步骤 2: 自动进行 4v2 分组
        self.log("步骤 2: 自动按类别进行 4v2 随机分组...")
        group_counts = df.groupby(label_column)[condition_column].nunique()
        valid_labels = group_counts[group_counts == 6].index
        if len(valid_labels) == 0:
            self.log(
                f"错误: 在 '{label_column}' 列中，没有任何一个类别包含恰好6个不同的 '{condition_column}' 组。无法执行4v2拆分。")
            messagebox.showerror("分组错误", f"没有任何类别包含恰好6个不同的组，无法执行4v2拆分。")
            return

        df_filtered = df[df[label_column].isin(valid_labels)].copy()
        self.log(f"已筛选出 {len(valid_labels)} 个符合条件的类别进行分析: {list(valid_labels)}")

        train_indices = []
        validation_indices = []
        np.random.seed(42)  # for reproducible splits

        for label, group_df in df_filtered.groupby(label_column):
            conditions = group_df[condition_column].unique()
            validation_conds = np.random.choice(conditions, 2, replace=False)
            train_conds = np.setdiff1d(conditions, validation_conds)

            self.log(f"  - 类别 '{label}': 训练组={list(train_conds)}, 验证组={list(validation_conds)}")

            train_indices.extend(group_df[group_df[condition_column].isin(train_conds)].index)
            validation_indices.extend(group_df[group_df[condition_column].isin(validation_conds)].index)

        train_df = df.loc[train_indices];
        validation_df = df.loc[validation_indices]
        self.log(f"\n分组完成！训练集样本数: {len(train_df)}, 验证集样本数: {len(validation_df)}")

        # 步骤 3: 数据预处理
        self.log("步骤 3: 数据预处理...")
        le = LabelEncoder();
        le.fit(df_filtered[label_column])
        X_train = train_df[selected_features];
        y_train = le.transform(train_df[label_column])
        X_validation = validation_df[selected_features];
        y_validation = le.transform(validation_df[label_column])

        scaler = StandardScaler();
        X_train_scaled = scaler.fit_transform(X_train)
        X_validation_scaled = scaler.transform(X_validation)

        # 步骤 4: 模型评估
        self.log("步骤 4: 开始模型评估...")
        models_to_evaluate = {
            'DT': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'SVM': SVC(random_state=42, kernel='linear', class_weight='balanced'),
            'XGB': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'LR': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'MLP': MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, hidden_layer_sizes=(50, 20))
        }

        summary_stats = []
        all_cms = {}  # 用于存储所有混淆矩阵数据
        original_stdout = sys.stdout;
        sys.stdout = self

        for name, model in models_to_evaluate.items():
            self.log(f"\n--- 正在评估 {name} ---")

            # 1. 在训练集上进行交叉验证
            min_class_count = np.min(np.bincount(y_train))
            n_splits = max(2, min(5, min_class_count))
            if n_splits < 2:
                cv_mean, cv_std = 0, 0
            else:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
                cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)
            self.log(f"  CV F1 (macro) on Train Set: {cv_mean:.4f} (+/- {cv_std:.4f})")

            # 2. 在整个训练集上训练，并在验证集上评估
            model.fit(X_train_scaled, y_train)
            y_pred_validation = model.predict(X_validation_scaled)

            # 保存混淆矩阵数据用于Excel导出
            cm = confusion_matrix(y_validation, y_pred_validation, labels=np.arange(len(le.classes_)))
            all_cms[name] = (cm, le.classes_)

            self.display_confusion_matrix(y_validation, y_pred_validation, le.classes_, f"{name}_Validation",
                                          save_base_dir)

            summary_stats.append({'Model': name, 'CV_F1_Mean': cv_mean, 'CV_F1_Std': cv_std})

        sys.stdout = original_stdout
        self.log("\n" + "=" * 20 + " 全部分析完成 " + "=" * 20)
        self.export_summary_report_to_txt(summary_stats, save_base_dir)
        self.export_cms_to_excel(all_cms, save_base_dir)  # 导出混淆矩阵到Excel
        messagebox.showinfo("分析完成", "所有模型评估完成，请查看日志和结果文件夹。")

    # --- 辅助函数 ---
    def display_confusion_matrix(self, y_true, y_pred, target_names, model_name, save_dir):
        """
        显示混淆矩阵，单元格内只显示百分比数值。
        """
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(target_names)))

        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_percent = np.nan_to_num(cm_percent)

        annot_labels = (np.asarray([f'{p * 100:.0f}' for p in cm_percent.flatten()])).reshape(cm.shape)

        num_classes = len(target_names)
        annot_fontsize = 14 if num_classes < 10 else 12
        tick_fontsize = 12
        label_fontsize = 16
        title_fontsize = 20
        figsize_w = 10 if num_classes <= 15 else 13
        figsize_h = 8 if num_classes <= 15 else 10

        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))

        sns.heatmap(cm_percent, annot=annot_labels, fmt='s', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names, ax=ax,
                    vmin=0, vmax=1, annot_kws={"size": annot_fontsize})

        ax.set_title(f'混淆矩阵 (%): {model_name}', fontsize=title_fontsize)
        ax.set_ylabel('真实标签', fontsize=label_fontsize)
        ax.set_xlabel('预测标签', fontsize=label_fontsize)
        plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)
        plt.tight_layout(pad=3.0)
        if save_dir and os.path.exists(save_dir):
            plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
        plt.show(block=False)

    def export_summary_report_to_txt(self, summary_stats, save_dir):
        if not save_dir or not os.path.exists(save_dir): return
        txt_path = os.path.join(save_dir, "model_summary_report.txt")
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 20 + " 模型性能汇总 " + "=" * 20 + "\n\n")
                summary_df = pd.DataFrame(summary_stats)
                summary_df['CV_F1_Mean'] = summary_df['CV_F1_Mean'].map('{:.4f}'.format)
                summary_df['CV_F1_Std'] = summary_df['CV_F1_Std'].map('{:.4f}'.format)
                summary_df = summary_df.sort_values(by='CV_F1_Mean', ascending=False).reset_index(drop=True)
                f.write(
                    "说明:\n- CV_F1_Mean: 模型在训练集上进行交叉验证的平均F1分数。\n- CV_F1_Std: 交叉验证F1分数的标准差，值越小说明模型性能越稳定。\n\n")
                f.write(summary_df.to_string());
                f.write("\n\n" + "=" * 54)
            self.log(f"\n最终汇总报告已保存至TXT文件: {txt_path}")
        except Exception as e:
            self.log(f"导出到TXT失败: {e}")

    def export_cms_to_excel(self, cms_dict, save_dir):
        """
        将所有模型的混淆矩阵数值导出到单个Excel文件的不同工作表中。
        """
        if not save_dir or not os.path.exists(save_dir): return
        excel_path = os.path.join(save_dir, "confusion_matrices_data.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for model_name, (cm, target_names) in cms_dict.items():
                    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
                    df_cm.index.name = 'True Label'
                    df_cm.columns.name = 'Predicted Label'
                    df_cm.to_excel(writer, sheet_name=f"{model_name}_CM")
            self.log(f"所有混淆矩阵的数值已导出至Excel文件: {excel_path}")
        except Exception as e:
            self.log(f"导出混淆矩阵到Excel失败: {e}")

    def write(self, text):
        if text.strip(): self.log(text.strip())

    def flush(self):
        pass

    def select_column_role(self, columns, title):
        dialog = Toplevel(self.master);
        dialog.title(title);
        dialog.geometry("400x450");
        selected_col = StringVar(value=columns[0] if columns else "")
        tk.Label(dialog, text=title, font=('Arial', 14, 'bold')).pack(pady=10)
        inner_frame = Frame(dialog);
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        for col in columns: Radiobutton(inner_frame, text=col, variable=selected_col, value=col,
                                        font=('Arial', 12)).pack(anchor='w')
        Button(dialog, text="确认", command=dialog.destroy, font=('Arial', 14, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_col.get()

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
        for feature in all_features: Checkbutton(inner_frame, text=feature, var=vars[feature], font=('Arial', 12)).pack(
            anchor='w', padx=20)
        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")

        Button(dialog, text="确认选择", command=confirm, font=('Arial', 14, 'bold')).pack(pady=10)
        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_features_list


if __name__ == '__main__':
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

