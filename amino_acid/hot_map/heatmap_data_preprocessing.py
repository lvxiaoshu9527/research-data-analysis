import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.stats import zscore
import os


def select_file(title):
    """
    打开文件选择对话框，用于选择单个CSV文件。
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    return file_path


def select_save_path(default_filename):
    """
    打开文件保存对话框。
    """
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        title="请选择保存路径",
        initialfile=default_filename,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    return save_path


def main():
    """
    主函数，执行所有数据处理步骤。
    """
    # 确保必要的库已安装
    try:
        from scipy.stats import zscore
    except ImportError:
        messagebox.showerror("缺少库 (Missing Library)",
                             "需要 scipy 库来进行Z-score标准化。\n"
                             "请在您的终端运行: pip install scipy")
        return

    # --- Step 1: 读取包含所有原始数据的CSV文件 ---
    csv_file_path = select_file("请选择包含全浓度数据的CSV文件")
    if not csv_file_path:
        messagebox.showinfo("提示", "未选择任何文件，程序退出。")
        return

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        messagebox.showerror("文件读取错误", f"读取CSV文件时出错: {e}")
        return

    # --- Step 2: 数据整合 - 计算全浓度平均响应 ---
    print("正在计算每种氨基酸的全浓度平均响应...")

    # 自动识别标签列和特征列
    label_col = 'AA' if 'AA' in df.columns else 'Label'
    if label_col not in df.columns:
        messagebox.showerror("列名错误", "文件中必须包含氨基酸标签列 ('AA' 或 'Label')。")
        return

    feature_columns = [col for col in df.columns if col not in [label_col, '浓度/uM', 'Concentration']]

    # 按氨基酸分组，计算每个传感器通道的平均值
    avg_response_df = df.groupby(label_col)[feature_columns].mean()
    avg_response_df.fillna(0, inplace=True)

    print("平均响应矩阵计算完成。")
    print(avg_response_df.head())

    # --- Step 3: 数据标准化 - 进行行Z-score变换 ---
    print("\n正在对平均响应矩阵进行行Z-score标准化...")

    if avg_response_df.shape[1] > 1:
        # 使用 apply 函数确保在每一行上独立计算 z-score
        # result_type='expand' 确保结果是一个DataFrame
        zscore_matrix = avg_response_df.apply(
            lambda row: zscore(row, nan_policy='omit'),
            axis=1,
            result_type='expand'
        )
        # 如果某行的标准差为0，zscore会产生NaN，这里用0填充
        zscore_matrix.fillna(0, inplace=True)
        # 重新命名列，因为apply可能会丢失它们
        zscore_matrix.columns = feature_columns
    else:
        # 如果只有一个特征列，zscore没有意义，直接返回0矩阵
        zscore_matrix = pd.DataFrame(0, index=avg_response_df.index, columns=avg_response_df.columns)

    print("Z-score标准化完成。")
    print(zscore_matrix.head())

    # --- Step 4: 保存为“Origin就绪”文件 ---
    default_save_name = f"{os.path.splitext(os.path.basename(csv_file_path))[0]}_for_origin.csv"
    save_path = select_save_path(default_save_name)

    if not save_path:
        messagebox.showinfo("提示", "未选择保存路径，程序退出。")
        return

    try:
        # 保存时将索引（氨基酸名称）也作为一列写入
        zscore_matrix.to_csv(save_path, index=True)
        print(f"\n处理完成！可直接导入Origin的文件已保存到:\n{save_path}")
        messagebox.showinfo("成功", f"“Origin就绪”文件已成功生成并保存。")
    except Exception as e:
        messagebox.showerror("保存错误", f"保存文件时出错: {e}")


if __name__ == "__main__":
    main()
