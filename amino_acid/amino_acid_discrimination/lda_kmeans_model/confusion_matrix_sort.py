# 1. 导入库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# 创建Tkinter根窗口，但将其隐藏，因为我们只需要文件对话框
root = tk.Tk()
root.withdraw() # 隐藏主窗口

def select_input_csv_file():
    """弹出文件对话框，让用户选择输入CSV文件"""
    file_path = filedialog.askopenfilename(
        title="选择混淆矩阵CSV文件",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    return file_path

def select_output_directory():
    """弹出目录对话框，让用户选择保存图片的目录"""
    directory_path = filedialog.askdirectory(
        title="选择保存热图的目录"
    )
    return directory_path

def generate_heatmap_with_dialogs():
    """
    使用文件对话框选择文件和目录，然后生成混淆矩阵热图。
    """
    # 提示用户选择CSV文件
    csv_file_path = select_input_csv_file()
    if not csv_file_path:
        messagebox.showinfo("取消操作", "未选择CSV文件，操作已取消。")
        return

    # 2. 读取数据
    try:
        if not os.path.exists(csv_file_path):
            messagebox.showerror("文件错误", f"错误：文件 '{csv_file_path}' 不存在。请检查路径是否正确。")
            return

        df = pd.read_csv(csv_file_path, index_col=0)
        messagebox.showinfo("文件读取", f"成功读取文件: {csv_file_path}")

        matrix = df.values
        try:
            matrix = matrix.astype(int)
        except ValueError:
            messagebox.showerror("数据错误", "警告：CSV文件中的某些数据无法转换为整数。请确保混淆矩阵只包含数字。")
            return

        # 3. 匈牙利算法重排序
        row_idx, col_idx = linear_sum_assignment(matrix, maximize=True)
        ordered_matrix = matrix[row_idx][:, col_idx]
        messagebox.showinfo("重排序完成", "混淆矩阵已按对角线最大化重新排序。")

        # 4. 生成新标签
        new_rows = [df.index[i] for i in row_idx]
        new_cols = [df.columns[i] for i in col_idx]

        # 5. 创建热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            ordered_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=new_cols,
            yticklabels=new_rows,
            cbar_kws={'label': '计数'}
        )
        plt.xlabel("预测聚类")
        plt.ylabel("真实氨基酸")
        plt.title("重新排序的混淆矩阵热图")
        plt.tight_layout()

        # 提示用户选择保存图片的目录
        output_directory = select_output_directory()
        if not output_directory:
            messagebox.showinfo("取消操作", "未选择输出目录，图片将不会保存。")
            plt.show() # 仍然显示图表但不保存
            return

        # 确保输出目录存在
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # 构建完整的输出文件路径
        output_filename = "reordered_confusion_matrix_heatmap.png"
        output_file_path = os.path.join(output_directory, output_filename)

        # 保存高分辨率PNG图片
        plt.savefig(output_file_path, dpi=300)
        messagebox.showinfo("保存成功", f"热图已保存为 '{output_file_path}'。")

        # 显示交互式图表
        plt.show()

    except FileNotFoundError:
        messagebox.showerror("文件错误", f"错误：文件 '{csv_file_path}' 未找到。请确保路径和文件名正确。")
    except pd.errors.EmptyDataError:
        messagebox.showerror("数据错误", "错误：CSV文件为空。请提供包含数据的CSV文件。")
    except pd.errors.ParserError:
        messagebox.showerror("格式错误", "错误：无法解析CSV文件。请检查CSV格式是否正确。")
    except Exception as e:
        messagebox.showerror("运行错误", f"发生未知错误：{e}")

# 运行主功能
if __name__ == "__main__":
    generate_heatmap_with_dialogs()
    root.destroy() # 关闭Tkinter根窗口

