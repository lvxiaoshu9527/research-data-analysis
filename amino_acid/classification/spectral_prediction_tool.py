import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Frame, Button, Text, Scrollbar, END, StringVar, Radiobutton
import os
import matplotlib as mpl
import traceback
import threading

# --- Matplotlib全局设置 (与训练脚本保持一致) ---
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("光谱数据预测工具 v2.0")
        master.geometry("700x550")

        self.model_path = ""
        self.data_path = ""

        # --- UI 控件框架 ---
        self.control_frame = Frame(master, padx=10, pady=10)
        self.control_frame.pack(fill=tk.X)

        self.btn_load_model = Button(self.control_frame, text="1. 加载模型包 (.joblib)", font=('Arial', 12),
                                     command=self.load_model)
        self.btn_load_model.pack(fill=tk.X, pady=5)

        self.btn_load_data = Button(self.control_frame, text="2. 加载待预测数据 (.csv)", font=('Arial', 12),
                                    command=self.load_data)
        self.btn_load_data.pack(fill=tk.X, pady=5)

        self.btn_run = Button(self.control_frame, text="开始预测与评估", font=('Arial', 12, 'bold'),
                              command=self.start_prediction_thread, state="disabled")
        self.btn_run.pack(fill=tk.X, pady=10)

        # --- 日志显示框架 (与训练脚本保持一致) ---
        self.log_frame = Frame(master, padx=10, pady=10)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = Text(self.log_frame, wrap=tk.WORD, state='disabled', font=('Courier New', 10))
        self.scrollbar = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.log("欢迎使用光谱数据预测工具！")
        self.log("请按顺序加载模型包和待预测数据。")

    def log(self, message):
        """向GUI的日志文本框中追加消息，线程安全。"""

        def append_log():
            if not self.master.winfo_exists(): return
            self.log_text.config(state='normal')
            self.log_text.insert(END, str(message) + "\n")
            self.log_text.see(END)
            self.log_text.config(state='disabled')

        self.master.after(0, append_log)

    def load_model(self):
        path = filedialog.askopenfilename(parent=self.master, title="选择模型包文件",
                                          filetypes=[("Joblib files", "*.joblib")])
        if path:
            self.model_path = path
            self.log(f"模型已加载: {os.path.basename(self.model_path)}")
            self.check_runnable()

    def load_data(self):
        path = filedialog.askopenfilename(parent=self.master, title="选择待预测的CSV数据文件",
                                          filetypes=[("CSV files", "*.csv")])
        if path:
            self.data_path = path
            self.log(f"数据已加载: {os.path.basename(self.data_path)}")
            self.check_runnable()

    def check_runnable(self):
        """检查是否满足运行条件，并更新按钮状态。"""
        if self.model_path and self.data_path:
            self.btn_run.config(state="normal")
            self.log("\n准备就绪，可以开始预测。")
        else:
            self.btn_run.config(state="disabled")

    def start_prediction_thread(self):
        """使用独立线程运行预测，避免UI卡死 (与训练脚本习惯一致)。"""
        self.btn_run.config(state="disabled", text="预测中...")
        self.log("\n" + "=" * 50)
        self.log("新的预测任务已开始...")
        prediction_thread = threading.Thread(target=self.run_prediction_wrapper)
        prediction_thread.daemon = True
        prediction_thread.start()

    def run_prediction_wrapper(self):
        """包装预测函数，捕获异常并确保UI状态恢复。"""
        try:
            self.run_prediction()
        except Exception as e:
            error_details = traceback.format_exc()
            self.log(f"!!! 发生严重错误 !!!\n错误类型: {type(e).__name__}\n错误信息: {e}")
            self.log(f"详细追溯信息:\n{error_details}")
            messagebox.showerror("程序遇到意外错误", f"发生未处理的错误: {e}\n\n请查看主窗口日志获取详细信息。")
        finally:
            self.master.after(0, lambda: self.btn_run.config(state="normal", text="开始预测与评估"))
            self.log("预测任务结束。")
            self.log("=" * 50 + "\n")

    def run_prediction(self):
        """执行完整的预测和评估流程。"""
        # 步骤 1: 加载模型包
        self.log("\n步骤 1/7: 正在加载模型包...")
        try:
            model_package = joblib.load(self.model_path)
            model = model_package['model']
            scaler = model_package['scaler']
            label_encoder = model_package['label_encoder']
            feature_columns = model_package['feature_columns']
            self.log(f"成功加载模型: {model_package.get('model_name', '未知类型')}")
        except Exception as e:
            self.log(f"错误：加载模型包失败: {e}");
            return

        # 步骤 2: 加载并校验新数据
        self.log("步骤 2/7: 正在加载并校验新数据...")
        try:
            new_data = pd.read_csv(self.data_path)
            if not all(col in new_data.columns for col in feature_columns):
                self.log("错误：新数据文件中缺少训练时所用的特征列。请检查文件。")
                missing_cols = set(feature_columns) - set(new_data.columns)
                self.log(f"缺少的列: {', '.join(missing_cols)}")
                return
        except Exception as e:
            self.log(f"错误：读取新数据文件失败: {e}");
            return

        # 步骤 3: 让用户指定真实标签列
        self.log("步骤 3/7: 请指定真实标签列...")
        true_label_column = self.select_column_dialog(new_data.columns.tolist(), "请选择真实标签列 (用于评估)")
        if not true_label_column:
            self.log("操作取消：未指定真实标签列。");
            return
        if not all(label in label_encoder.classes_ for label in new_data[true_label_column].unique()):
            self.log(f"警告：数据中的标签 '{new_data[true_label_column].unique()}' 含有模型未训练过的类别。")

        # 步骤 4: 数据预处理
        self.log("步骤 4/7: 正在对新数据进行预处理...")
        X_new = new_data[feature_columns]
        y_true_labels = new_data[true_label_column]
        # 关键：必须使用从模型包中加载的scaler进行transform
        X_new_scaled = scaler.transform(X_new)
        self.log("数据预处理完成。")

        # 步骤 5: 执行预测
        self.log("步骤 5/7: 正在执行预测...")
        y_pred_numeric = model.predict(X_new_scaled)
        y_pred_proba = model.predict_proba(X_new_scaled)
        y_pred_labels = label_encoder.inverse_transform(y_pred_numeric)
        self.log("预测完成。")

        # 步骤 6: 评估与可视化
        self.log("\n步骤 6/7: 生成评估报告和混淆矩阵...")
        report = classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_,
                                       zero_division=0)
        self.log("--- 在新测试数据上的模型评估报告 ---")
        self.log(report)
        self.display_confusion_matrix(y_true_labels, y_pred_labels, label_encoder.classes_)

        # 步骤 7: 保存结果
        self.log("\n步骤 7/7: 正在保存详细预测结果...")
        try:
            predictions_df = new_data.copy()
            predictions_df['predicted_label'] = y_pred_labels
            for i, class_name in enumerate(label_encoder.classes_):
                predictions_df[f'proba_{class_name}'] = y_pred_proba[:, i].round(4)

            save_dir = os.path.dirname(self.data_path)
            output_filename = os.path.join(save_dir, 'prediction_results.csv')
            predictions_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            self.log(f"成功！预测结果已保存至: {output_filename}")
            messagebox.showinfo("预测完成", f"预测结果已成功保存至:\n{output_filename}", parent=self.master)
        except Exception as e:
            self.log(f"错误: 保存结果文件失败: {e}")

    def display_confusion_matrix(self, y_true, y_pred, class_names):
        """生成并显示混淆矩阵图。"""
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('在新测试数据上的混淆矩阵', fontsize=20)
        ax.set_ylabel('真实标签 (True Label)', fontsize=16)
        ax.set_xlabel('预测标签 (Predicted Label)', fontsize=16)
        plt.xticks(rotation=45, ha='right');
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def select_column_dialog(self, columns, title):
        """弹出一个对话框让用户选择一列 (复用训练脚本的逻辑)。"""
        dialog = Toplevel(self.master)
        dialog.title(title)
        dialog.geometry("400x450")
        selected_col = StringVar(value=columns[0] if columns else "")

        tk.Label(dialog, text="请选择一个列名:", font=('Arial', 12, 'bold')).pack(pady=10)

        mid_frame = Frame(dialog);
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(mid_frame);
        scrollbar = tk.Scrollbar(mid_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas);
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw");
        canvas.configure(yscrollcommand=scrollbar.set)

        for col in columns:
            Radiobutton(scrollable_frame, text=col, variable=selected_col, value=col, font=('Arial', 12)).pack(
                anchor='w', padx=20)

        canvas.pack(side="left", fill="both", expand=True);
        scrollbar.pack(side="right", fill="y")
        Button(dialog, text="确认", command=dialog.destroy, font=('Arial', 12, 'bold')).pack(pady=10)

        dialog.transient(self.master);
        dialog.grab_set();
        self.master.wait_window(dialog)
        return selected_col.get()


if __name__ == '__main__':
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
