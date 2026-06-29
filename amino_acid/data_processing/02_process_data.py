import pandas as pd
import re
import os
import tkinter as tk
from tkinter import filedialog


def process_raw_data_files(file_paths):
    """
    处理原始数据 .xlsx 文件，并将它们整合成一个标准的DataFrame。
    这个版本使用了更灵活的正则表达式来识别各种手性格式。

    Args:
        file_paths (list of str): 用户选择的原始数据文件路径列表。

    Returns:
        tuple: (一个包含所有数据的DataFrame, 检测到的手性字符串)
    """
    all_data = []
    detected_chirality = None

    print("\n开始处理原始数据文件...")
    for file_path in file_paths:
        base_name = os.path.basename(file_path)

        if 'output_data' not in base_name:
            print(f"  - 非原始数据文件，跳过: {base_name}")
            continue

        # --- 关键升级：使用更灵活的正则表达式 ---
        # 它可以识别像 (6,5)-S7 或 (6,5) 等各种格式的手性
        match = re.search(r'(.+?)output_data-([A-Za-z-]+?)(\d+)\.xlsx', base_name)

        if not match:
            print(f"  - 文件名格式不匹配，跳过: {base_name}")
            continue

        # group(1) 现在可以捕获完整的手性字符串
        current_chirality = match.group(1)
        amino_acid = match.group(2)

        if detected_chirality is None:
            detected_chirality = current_chirality
            print(f"  - 检测到本次处理的手性为: {detected_chirality}")

        if current_chirality != detected_chirality:
            print(
                f"  - 警告: 文件 {base_name} 的手性 '{current_chirality}' 与已检测到的 '{detected_chirality}' 不符，已跳过。")
            continue

        print(f"  - 正在处理: {base_name}")
        try:
            max_value_df = pd.read_excel(file_path, sheet_name='Max_Value').rename(columns={'Name': 'Concentration'})
            lorentz_df = pd.read_excel(file_path, sheet_name='Lorentz').rename(columns={'Column': 'Concentration'})
            merged_df = pd.merge(max_value_df, lorentz_df, on='Concentration')

            temp_df = pd.DataFrame({
                'Amino_Acid': amino_acid,
                'Concentration': merged_df['Concentration'],
                '(I-I0)/I0': merged_df['(I-I0)/I0'],
                'delta_lambda': merged_df['Δλ']
            })
            all_data.append(temp_df)

        except Exception as e:
            print(f"    - 错误: 读取文件 '{base_name}' 失败: {e}")

    if not all_data:
        return pd.DataFrame(), None

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df, detected_chirality


def save_as_formatted_excel(df, chirality, output_path):
    """
    将数据按指定格式（排序、空行分隔）分离成训练/测试集，并保存到Excel。
    """
    print(f"\n正在按指定格式准备数据...")

    df.sort_values(by=['Amino_Acid', 'Concentration'], inplace=True)

    train_concentrations = [20, 40, 60, 80, 100, 120]
    test_concentrations = [10, 30, 50, 70, 90, 110]

    train_df = df[df['Concentration'].isin(train_concentrations)]
    test_df = df[df['Concentration'].isin(test_concentrations)]

    def add_blank_rows(dataframe):
        if dataframe.empty:
            return dataframe

        amino_acids = dataframe['Amino_Acid'].unique()
        formatted_dfs = []

        for i, acid in enumerate(amino_acids):
            acid_df = dataframe[dataframe['Amino_Acid'] == acid]
            formatted_dfs.append(acid_df)

            if i < len(amino_acids) - 1:
                blank_row = pd.DataFrame([[None] * len(dataframe.columns)], columns=dataframe.columns)
                formatted_dfs.append(blank_row)

        return pd.concat(formatted_dfs, ignore_index=True)

    formatted_train_df = add_blank_rows(train_df)
    formatted_test_df = add_blank_rows(test_df)

    intensity_col_name = f'(I-I0)/I0-{chirality}'
    wavelength_col_name = f'Δλ-{chirality}'

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for df_to_save, sheet_name in [(formatted_train_df, '训练集 (Training Set)'),
                                       (formatted_test_df, '测试集 (Test Set)')]:
            if df_to_save.empty:
                continue

            df_to_save.rename(columns={
                'Amino_Acid': '氨基酸',
                'Concentration': '浓度/uM',
                '(I-I0)/I0': intensity_col_name,
                'delta_lambda': wavelength_col_name
            }, inplace=True)

            final_columns = ['氨基酸', '浓度/uM', intensity_col_name, wavelength_col_name]
            df_to_save[final_columns].to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  - '{sheet_name}' 已按新格式成功创建。")


def main():
    """主函数，负责整个交互流程。"""
    try:
        import openpyxl
    except ImportError:
        print("错误: 缺少 'openpyxl' 库。请运行: pip install openpyxl")
        return

    root = tk.Tk()
    root.withdraw()

    print("请在弹出的窗口中，选择同一手性的所有原始数据 .xlsx 文件...")
    selected_files = filedialog.askopenfilenames(
        title="选择同一手性的所有原始数据文件",
        filetypes=[("Excel files", "*.xlsx")]
    )

    if not selected_files:
        print("\n您没有选择任何文件，程序已退出。")
        return

    final_df, detected_chirality = process_raw_data_files(selected_files)

    if final_df.empty or detected_chirality is None:
        print("\n数据处理未能生成任何结果。")
        return

    output_path = filedialog.asksaveasfilename(
        title="请选择保存路径",
        initialdir=os.path.dirname(selected_files[0]),
        initialfile=f"{detected_chirality}-汇总.xlsx",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")]
    )

    if output_path:
        save_as_formatted_excel(final_df, detected_chirality, output_path)
        print(f"\n成功！数据已按新格式保存到:\n{output_path}")
    else:
        print("\n您取消了保存操作。")


if __name__ == "__main__":
    main()
