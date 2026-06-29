import pandas as pd
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Optional


def select_csv_file() -> str:
    """
    打开一个文件选择对话框，让用户选择一个CSV文件。
    Returns:
        str: 用户选择的文件路径，如果取消则为空字符串。
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="请选择您的数据文件 (CSV)",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    return file_path


def identify_features_and_labels(df: pd.DataFrame) -> Tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
    """
    从DataFrame中自动识别特征列、标签列和手性列表。

    Args:
        df (pd.DataFrame): 输入的数据框。

    Returns:
        Tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
        - feature_cols: 识别出的特征列名列表。
        - label_col: 识别出的标签列名 ('AA')。
        - chiralities: 从特征列中提取的手性对列表。
        如果关键列不存在，则返回 None。
    """
    # 1. 识别标签列
    label_col = 'AA'
    if label_col not in df.columns:
        messagebox.showerror("错误", f"数据文件中未找到标签列 '{label_col}'。")
        return None, None, None

    # 2. 使用正则表达式识别特征列和手性
    # 匹配模式如: (6,5)_intensity, (7,5)_shift, S7-(6,5)_intensity 等
    feature_pattern = re.compile(r'^\(?S?\d+-?\(?\d+,\d+\)?\)_intensity$|^\(?S?\d+-?\(?\d+,\d+\)?\)_shift$')
    feature_cols = [col for col in df.columns if feature_pattern.match(col)]

    if not feature_cols:
        messagebox.showerror("错误", "未能根据 `(n,m)_intensity/shift` 模式自动识别出任何特征列。")
        return None, None, None

    # 提取手性 (Chiralities)
    chirality_pattern = re.compile(r'^(S?\d+-?\(?\d+,\d+\)?)_')
    chiralities = sorted(list(set(chirality_pattern.match(col).group(1) for col in feature_cols)))

    return feature_cols, label_col, chiralities


def load_and_inspect_data(file_path: str):
    """
    加载CSV数据，进行初步检查，并打印识别出的信息。
    """
    if not file_path:
        print("未选择文件，程序退出。")
        return

    print(f"--- 正在加载数据文件: {file_path} ---")

    try:
        # 使用 pandas 读取 CSV
        df = pd.read_csv(file_path)
        print("✅ 数据加载成功！")

        # --- 数据初步洞察 ---
        print("\n--- 1. 数据概览 ---")
        print(f"数据形状 (行, 列): {df.shape}")
        print("\n前5行数据:")
        print(df.head())

        print("\n数据类型和非空值信息:")
        df.info(verbose=False)  # 使用精简输出

        # --- 特征和标签识别 ---
        print("\n--- 2. 特征与标签自动识别 ---")
        feature_cols, label_col, chiralities = identify_features_and_labels(df)

        if feature_cols:
            print(f"✅ 成功识别到 {len(feature_cols)} 个特征列。")
            # print("部分特征列示例:", feature_cols[:5])

            print(f"\n✅ 成功识别到标签列: '{label_col}'")
            unique_labels = df[label_col].unique()
            print(f"   包含 {len(unique_labels)} 个唯一标签: {', '.join(map(str, unique_labels))}")

            print(f"\n✅ 成功识别到 {len(chiralities)} 个手性对: {chiralities}")

        # --- 缺失值检查 ---
        print("\n--- 3. 缺失值检查 ---")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]  # 只显示有缺失值的列
        if not missing_values.empty:
            print("发现以下列存在缺失值:")
            print(missing_values)
        else:
            print("✅ 数据集中没有发现缺失值。")

    except FileNotFoundError:
        messagebox.showerror("文件未找到", f"无法找到文件: {file_path}")
    except Exception as e:
        messagebox.showerror("读取错误", f"读取或处理文件时发生错误: {e}")


if __name__ == "__main__":
    # 弹出文件选择框让用户选择数据
    selected_file = select_csv_file()

    # 执行加载和检查
    load_and_inspect_data(selected_file)
