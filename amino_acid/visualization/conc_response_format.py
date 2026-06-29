import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys


def transform_to_origin_wide_format(input_filename, output_filename):
    """
    将输入的CSV文件（长格式）转换为适用于Origin的宽格式。
    - 同一浓度下的重复测量值会（Rep0, Rep1, ...）会平铺到同一行。
    """

    print(f"正在处理: {input_filename}...")

    try:
        # 1. 加载数据
        df = pd.read_csv(input_filename)

        # 2. 确定关键列
        concentration_col = '浓度/uM'
        id_cols = ['AA', concentration_col]
        feature_cols = [col for col in df.columns if col not in id_cols]

        # 3. 为重复组编号
        df['replicate_id'] = df.groupby(id_cols).cumcount()

        # 4. 执行数据透视
        df_pivoted = df.set_index(id_cols + ['replicate_id'])[feature_cols].unstack(level='replicate_id')

        # 5. 展平列名
        df_pivoted.columns = [f'{feature}_Rep{rep_id}' for feature, rep_id in df_pivoted.columns]

        # 6. 恢复索引
        df_wide = df_pivoted.reset_index()

        # 7. 排序
        df_wide = df_wide.sort_values(by=id_cols)

        # 8. 保存到新的 CSV 文件
        df_wide.to_csv(output_filename, index=False)

        print(f"成功！已保存为: {output_filename}\n")
        return True

    except FileNotFoundError:
        print(f"错误: 未找到文件 {input_filename}。")
    except Exception as e:
        print(f"处理 {input_filename} 时发生错误: {e}")

    print("\n")  # 确保出错时也有换行
    return False


def get_file_and_save_paths(title_prefix):
    """
    使用Tkinter弹窗获取输入文件和保存路径
    """
    # 隐藏主窗口
    root = tk.Tk()
    root.withdraw()

    # 1. 弹出窗口选择输入文件
    input_path = filedialog.askopenfilename(
        title=f"请选择 {title_prefix} 数据文件 (例如: {title_prefix.lower()}_data.csv)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not input_path:
        print(f"未选择 {title_prefix} 输入文件，操作取消。")
        return None, None

    # 2. 弹出窗口选择保存路径
    # 从输入路径推断默认的输出文件名
    input_dir, input_name = os.path.split(input_path)
    default_output_name = f"{os.path.splitext(input_name)[0]}_for_origin.csv"

    output_path = filedialog.asksaveasfilename(
        title=f"请选择 {title_prefix} 转换后的保存路径",
        initialdir=input_dir,
        initialfile=default_output_name,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )

    if not output_path:
        print(f"未选择 {title_prefix} 保存路径，操作取消。")
        return None, None

    return input_path, output_path


def main():
    print("--- Origin 数据转换脚本 ---")

    # --- 处理训练数据 ---
    print("--- 步骤 1: 处理训练数据 (Train Data) ---")
    train_input, train_output = get_file_and_save_paths("Train")

    if train_input and train_output:
        transform_to_origin_wide_format(train_input, train_output)
    else:
        print("跳过训练数据处理。\n")

    # --- 处理测试数据 ---
    print("--- 步骤 2: 处理测试数据 (Test Data) ---")
    test_input, test_output = get_file_and_save_paths("Test")

    if test_input and test_output:
        transform_to_origin_wide_format(test_input, test_output)
    else:
        print("跳过测试数据处理。\n")

    print("--- 全部处理完毕 ---")
    # 在某些系统上，可能需要按键才退出
    try:
        # 尝试等待用户输入，以便他们能看到控制台消息
        input("按 Enter 键退出...")
    except EOFError:
        pass  # 如果在非交互式环境中运行，直接退出


# --- 运行主程序 ---
if __name__ == "__main__":
    # 检查是否安装了 pandas
    try:
        import pandas
    except ImportError:
        print("错误: 未找到 'pandas' 库。")
        print("请先安装 pandas: pip install pandas")
        try:
            input("按 Enter 键退出...")
        except EOFError:
            pass
        sys.exit()

    main()