import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import os

def calculate_combined_score(series, alpha=0.5):
    """计算综合评分，考虑平均值和标准差。"""
    mean_val = series.mean()
    std_val = series.std()
    return mean_val - alpha * std_val

def select_top_n_significant_aa(df, n=4, aa_col='AA', intensity_col='(I-I0)/I0', shift_col='Δλ'):
    """为每个手性分别挑选前 N 个响应显著的氨基酸。"""
    if aa_col in df.columns:
        # 计算每个氨基酸的强度和位移的综合评分
        scores = df.groupby(aa_col).apply(lambda x: calculate_combined_score(np.sqrt(x[intensity_col].apply(pd.to_numeric, errors='coerce').fillna(0)**2 +
                                                                                   x[shift_col].apply(pd.to_numeric, errors='coerce').fillna(0)**2)))
        top_n_aa = scores.sort_values(ascending=False).head(n).index.tolist()
        return top_n_aa
    else:
        return []

def process_excel_for_training(file_path):
    """处理 Excel 文件，提取训练数据并保留原始测量值，增加来源信息。"""
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        expected_sheets = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']

        if not all(sheet in sheet_names for sheet in expected_sheets):
            messagebox.showerror("错误", f"Excel 文件缺少预期的工作表：{expected_sheets}")
            return None

        significant_aa_by_handedness = {}
        all_hand_data = {}

        for sheet_name in expected_sheets:
            try:
                df = xls.parse(sheet_name)
                if 'AA' in df.columns and '浓度/uM' in df.columns and '(I-I0)/I0' in df.columns and 'Δλ' in df.columns:
                    all_hand_data[sheet_name] = df.copy()
                    top_n_aa = select_top_n_significant_aa(df.copy(), n=4)
                    significant_aa_by_handedness[sheet_name] = top_n_aa
                    print(f"手性 '{sheet_name}' 中响应最显著的前 4 个氨基酸: {top_n_aa}")
                else:
                    messagebox.showerror("错误", f"工作表 '{sheet_name}' 缺少必要的列 ('AA', '浓度/uM', '(I-I0)/I0', 'Δλ')。")
                    return None
            except Exception as e:
                messagebox.showerror("错误", f"处理工作表 '{sheet_name}' 时发生错误：{e}")
                return None

        all_significant_aa_with_source = {}
        for hand, aa_list in significant_aa_by_handedness.items():
            for aa in aa_list:
                if aa not in all_significant_aa_with_source:
                    all_significant_aa_with_source[aa] = []
                all_significant_aa_with_source[aa].append(hand)

        print("\n所有手性中响应显著的氨基酸及其来源:", all_significant_aa_with_source)

        training_data_list = []
        if all_significant_aa_with_source and all_hand_data:
            # 使用第一个手性的浓度作为基准，假设所有手性具有相同的浓度种类
            concentrations = sorted(all_hand_data[expected_sheets[0]]['浓度/uM'].unique())

            for aa in sorted(list(all_significant_aa_with_source.keys())):
                source_handedness = all_significant_aa_with_source[aa]
                for conc in concentrations:
                    sample = {'AA': aa, '浓度/uM': conc, '来源手性碳管': source_handedness}
                    for hand in expected_sheets:
                        if hand in all_hand_data:
                            aa_df = all_hand_data[hand][(all_hand_data[hand]['AA'] == aa) & (all_hand_data[hand]['浓度/uM'] == conc)]
                            if not aa_df.empty:
                                sample[f'{hand}_intensity'] = aa_df['(I-I0)/I0'].tolist()
                                sample[f'{hand}_shift'] = aa_df['Δλ'].tolist()
                            else:
                                sample[f'{hand}_intensity'] = np.nan
                                sample[f'{hand}_shift'] = np.nan
                        else:
                            sample[f'{hand}_intensity'] = np.nan
                            sample[f'{hand}_shift'] = np.nan

                    # 展开每个样本，处理每个浓度下的原始测量值
                    max_len = 0
                    for key in sample:
                        if isinstance(sample[key], list) and key not in ['AA', '浓度/uM', '来源手性碳管']:
                            max_len = max(max_len, len(sample[key]))

                    if max_len > 0:
                        for i in range(max_len):
                            row = {'AA': aa, '浓度/uM': conc, '来源手性碳管': source_handedness}
                            for col in [f'{h}_intensity' for h in expected_sheets] + [f'{h}_shift' for h in expected_sheets]:
                                if isinstance(sample[col], list) and len(sample[col]) > i:
                                    row[col] = sample[col][i]
                                else:
                                    row[col] = np.nan
                            training_data_list.append(row)
                    elif max_len == 0:
                        training_data_list.append(sample)


        if training_data_list:
            return pd.DataFrame(training_data_list)
        else:
            return None

    except FileNotFoundError:
        messagebox.showerror("错误", f"找不到文件：{file_path}")
        return None
    except Exception as e:
        messagebox.showerror("错误", f"读取 Excel 文件时发生错误：{e}")
        return None

def select_file():
    """弹出文件选择对话框，选择 Excel 文件。"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择包含手性碳管数据的 Excel 文件",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    return file_path

def select_save_path(default_filename="processed_training_data.csv"):
    """弹出文件保存对话框，选择保存路径和文件名。"""
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        title="选择保存训练数据的文件路径和文件名",
        defaultextension=".csv",
        initialfile=default_filename,
        filetypes=[("CSV files", "*.csv")]
    )
    return save_path

def main():
    """主函数，控制脚本的执行流程。"""
    excel_file_path = select_file()

    if excel_file_path:
        processed_df = process_excel_for_training(excel_file_path)

        if processed_df is not None:
            print("\n处理后的训练数据 (前 10 行):")
            print(processed_df.head(10))

            save_file_path = select_save_path()
            if save_file_path:
                try:
                    processed_df.to_csv(save_file_path, index=False, encoding='utf-8')
                    messagebox.showinfo("成功", f"训练数据已保存到：\n{save_file_path}")
                except Exception as e:
                    messagebox.showerror("错误", f"保存文件时发生错误：{e}")
            else:
                messagebox.showinfo("提示", "未选择保存路径，数据未保存。")
        else:
            messagebox.showinfo("提示", "没有生成有效的训练数据。")
    else:
        messagebox.showinfo("提示", "未选择 Excel 文件，程序已退出。")

if __name__ == "__main__":
    main()