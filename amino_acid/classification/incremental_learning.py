import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Frame, Button, Text, Scrollbar, END, StringVar, Entry, Label, \
    Checkbutton, BooleanVar, Radiobutton
import os
import matplotlib as mpl
import traceback
import threading
import numpy as np

# --- Matplotlib全局设置 ---
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


class IncrementalApp:
    def __init__(self, master):
        self.master = master
        master.title("增量学习验证工具 v1.2 (布局修复)")
        master.geometry("700x600")

        self.original_data_path = ""
        self.new_data_path = ""

        # --- UI 控件框架 ---
        self.control_frame = Frame(master, padx=10, pady=10)
        self.control_frame.pack(fill=tk.X)

        self.btn_load_original = Button(self.control_frame, text="1. 加载原始训练数据 (20, 40uM...)",
                                        font=('Arial', 12), command=self.load_original_data)
        self.btn_load_original.pack(fill=tk.X, pady=4)

        self.btn_load_new = Button(self.control_frame, text="2. 加载新数据 (10, 30uM...)", font=('Arial', 12),
                                   command=self.load_new_data)
        self.btn_load_new.pack(fill=tk.X, pady=4)

        # --- 参数设置框架 ---
        param_frame = Frame(self.control_frame, pady=5)
        param_frame.pack(fill=tk.X)
        Label(param_frame, text="提取新数据的", font=('Arial', 12)).pack(side=tk.LEFT, padx=(0, 5))
        self.percent_var = StringVar(value="50")
        Entry(param_frame, textvariable=self.percent_var, width=5, font=('Arial', 12)).pack(side=tk.LEFT)
        Label(param_frame, text="% 用于增强训练，其余用于最终验证", font=('Arial', 12)).pack(side=tk.LEFT, padx=(5, 0))

        self.btn_run = Button(self.control_frame, text="开始增量训练与验证", font=('Arial', 12, 'bold'),
                              command=self.start_analysis_thread, state="disabled")
        self.btn_run.pack(fill=tk.X, pady=10)

        # --- 日志显示框架 ---
        self.log_frame = Frame(master, padx=10, pady=10)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_frame.grid_rowconfigure(0, weight=1);
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = Text(self.log_frame, wrap=tk.WORD, state='disabled', font=('Courier New', 10))
        self.scrollbar = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew");
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.log("欢迎使用增量学习验证工具！")
        self.log("请按顺序加载原始数据和新数据。")

    def log(self, message):
        def append_log():
            if not self.master.winfo_exists(): return
            self.log_text.config(state='normal')
            self.log_text.insert(END, str(message) + "\n");
            self.log_text.see(END)
            self.log_text.config(state='disabled')

        self.master.after(0, append_log)

    def load_original_data(self):
        path = filedialog.askopenfilename(parent=self.master, title="选择原始训练数据CSV",
                                          filetypes=[("CSV files", "*.csv")])
        if path:
            self.original_data_path = path
            self.log(f"原始数据已加载: {os.path.basename(path)}")
            self.check_runnable()

    def load_new_data(self):
        path = filedialog.askopenfilename(parent=self.master, title="选择新数据CSV", filetypes=[("CSV files", "*.csv")])
        if path:
            self.new_data_path = path
            self.log(f"新数据已加载: {os.path.basename(path)}")
            self.check_runnable()

    def check_runnable(self):
        if self.original_data_path and self.new_data_path:
            self.btn_run.config(state="normal")
            self.log("\n准备就绪，可以开始验证。")

    def start_analysis_thread(self):
        self.btn_run.config(state="disabled", text="分析中...")
        self.log("\n" + "=" * 50);
        self.log("新的分析任务已开始...")
        analysis_thread = threading.Thread(target=self.run_analysis_wrapper)
        analysis_thread.daemon = True
        analysis_thread.start()

    def run_analysis_wrapper(self):
        try:
            self.run_analysis()
        except Exception as e:
            error_details = traceback.format_exc()
            self.log(f"!!! 发生严重错误 !!!\n错误类型: {type(e).__name__}\n错误信息: {e}")
            self.log(f"详细追溯信息:\n{error_details}")
            messagebox.showerror("程序遇到意外错误", f"发生未处理的错误: {e}\n\n请查看主窗口日志。")
        finally:
            self.master.after(0, lambda: self.btn_run.config(state="normal", text="开始增量训练与验证"))
            self.log("分析任务结束。");
            self.log("=" * 50 + "\n")

    def run_analysis(self):
        # ... (The logic remains the same)
        self.log("步骤 1/8: 正在加载数据...")
        try:
            df_original = pd.read_csv(self.original_data_path)
            df_new = pd.read_csv(self.new_data_path)
            percent_to_train = float(self.percent_var.get())
            if not (0 < percent_to_train < 100):
                self.log("错误：提取百分比必须在0和100之间。");
                return
        except Exception as e:
            self.log(f"加载数据或解析参数时出错: {e}");
            return

        self.log("步骤 2/8: 请指定标签列和特征列...")
        columns = df_original.columns.tolist()
        if set(columns) != set(df_new.columns.tolist()):
            self.log("错误：两个数据文件的列名不完全匹配，请检查文件。")
            messagebox.showerror("文件错误", "原始数据和新数据的列名不一致，请处理后再试。")
            return

        selections = self.select_columns_dialog(columns)
        if not selections: self.log("操作取消。"); return
        label_col, feature_cols = selections['label'], selections['features']

        self.log(f"步骤 3/8: 划分新数据... ( {percent_to_train}% 用于训练, {100 - percent_to_train}% 用于验证)")
        X_new = df_new[feature_cols]
        y_new = df_new[label_col]
        X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
            X_new, y_new, test_size=(100 - percent_to_train) / 100.0, random_state=42, stratify=y_new
        )

        self.log("步骤 4/8: 组合成扩展训练集...")
        X_original = df_original[feature_cols]
        y_original = df_original[label_col]
        X_train_combined = pd.concat([X_original, X_new_train], ignore_index=True)
        y_train_combined = pd.concat([y_original, y_new_train], ignore_index=True)
        self.log(f"原始训练集样本数: {len(X_original)}")
        self.log(f"增补训练集样本数: {len(X_new_train)}")
        self.log(f"扩展后总训练样本数: {len(X_train_combined)}")
        self.log(f"最终验证集样本数: {len(X_new_test)}")

        self.log("步骤 5/8: 数据预处理...")
        le = LabelEncoder().fit(y_train_combined)
        y_train_encoded = le.transform(y_train_combined)
        y_test_encoded = le.transform(y_new_test)

        scaler = StandardScaler().fit(X_train_combined)
        X_train_scaled = scaler.transform(X_train_combined)
        X_test_scaled = scaler.transform(X_new_test)

        self.log("步骤 6/8: 训练增强后的模型...")
        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train_encoded)
        self.log("模型训练完成。")

        self.log("步骤 7/8: 在最终验证集上评估...")
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test_encoded, y_pred, target_names=le.classes_)
        self.log("\n--- 在最终验证集上的性能评估报告 ---")
        self.log(report)

        self.log("步骤 8/8: 可视化混淆矩阵...")
        cm = confusion_matrix(y_test_encoded, y_pred, labels=np.arange(len(le.classes_)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('在最终验证集上的混淆矩阵', fontsize=20)
        plt.ylabel('真实标签', fontsize=16);
        plt.xlabel('预测标签', fontsize=16)
        plt.xticks(rotation=45, ha='right');
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def select_columns_dialog(self, columns):
        dialog = Toplevel(self.master)
        dialog.title("选择标签列和特征列")
        dialog.geometry("500x600")

        selections = {}

        # --- 确认按钮框架 (置于底部) ---
        bottom_frame = Frame(dialog)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # --- 主内容框架 (占据剩余空间) ---
        main_frame = Frame(dialog)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- 标签列选择框架 ---
        label_frame = Frame(main_frame, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
        label_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Label(label_frame, text="1. 选择唯一的标签列 (Y)", font=('Arial', 12, 'bold')).pack()

        label_var = StringVar(value=columns[0])

        label_canvas_frame = Frame(label_frame, height=100)
        label_canvas_frame.pack(fill=tk.X, pady=5)
        label_canvas = tk.Canvas(label_canvas_frame);
        label_scrollbar = tk.Scrollbar(label_canvas_frame, orient="vertical", command=label_canvas.yview)
        label_scrollable_frame = Frame(label_canvas);
        label_scrollable_frame.bind("<Configure>",
                                    lambda e: label_canvas.configure(scrollregion=label_canvas.bbox("all")))
        label_canvas.create_window((0, 0), window=label_scrollable_frame, anchor="nw");
        label_canvas.configure(yscrollcommand=label_scrollbar.set)

        for col in columns:
            Radiobutton(label_scrollable_frame, text=col, variable=label_var, value=col).pack(anchor='w', padx=10)

        label_canvas.pack(side="left", fill="x", expand=True);
        label_scrollbar.pack(side="right", fill="y")

        # --- 特征列选择框架 ---
        feature_frame = Frame(main_frame, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
        feature_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        tk.Label(feature_frame, text="2. 选择特征列 (X)", font=('Arial', 12, 'bold')).pack()

        feature_vars = {col: BooleanVar(value=True) for col in columns}
        if columns: feature_vars[columns[0]].set(False)

        def sync_selection():
            selected_label = label_var.get()
            for col, var in feature_vars.items():
                var.set(col != selected_label)

        Button(feature_frame, text="根据标签选择默认特征", command=sync_selection).pack(pady=5)

        feature_canvas_frame = Frame(feature_frame)
        feature_canvas_frame.pack(fill=tk.BOTH, expand=True)
        feature_canvas = tk.Canvas(feature_canvas_frame);
        feature_scrollbar = tk.Scrollbar(feature_canvas_frame, orient="vertical", command=feature_canvas.yview)
        feature_scrollable_frame = Frame(feature_canvas);
        feature_scrollable_frame.bind("<Configure>",
                                      lambda e: feature_canvas.configure(scrollregion=feature_canvas.bbox("all")))
        feature_canvas.create_window((0, 0), window=feature_scrollable_frame, anchor="nw");
        feature_canvas.configure(yscrollcommand=feature_scrollbar.set)

        for col in columns:
            Checkbutton(feature_scrollable_frame, text=col, var=feature_vars[col]).pack(anchor='w', padx=10)

        feature_canvas.pack(side="left", fill="both", expand=True);
        feature_scrollbar.pack(side="right", fill="y")

        def confirm():
            selections['label'] = label_var.get()
            selected_features = [col for col, var in feature_vars.items() if var.get()]

            if selections['label'] in selected_features:
                self.log(f"警告: 标签列 '{selections['label']}' 已自动从特征列表中移除。")
                selected_features.remove(selections['label'])

            selections['features'] = selected_features
            if not selections['features']:
                messagebox.showerror("错误", "请至少选择一个特征列！", parent=dialog);
                return
            dialog.destroy()

        # 将确认按钮添加到始终可见的底部框架
        Button(bottom_frame, text="确认选择", command=confirm, font=('Arial', 12, 'bold')).pack()

        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selections


if __name__ == '__main__':
    root = tk.Tk()
    app = IncrementalApp(root)
    root.mainloop()

