import pandas as pd
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

# ================= 1. 锁定列顺序 =================
# 您指定的4种手性，加上其他可能的手性
STANDARD_CHIRALITIES = [
    "(6,5)", "(7,5)", "(8,3)", "(6,5)-S7",  # 优先排这4个
    "(6,4)", "(7,3)", "(8,6)", "(8,7)",
    "(9,1)", "(9,4)", "(9,5)", "(10,2)", "(10,3)"
]


def sort_cols_strict(cols):
    """
    排序规则：
    1. 手性按列表顺序
    2. _intensity 在前, _shift 在后
    """

    def key_func(col_name):
        # 1. 找手性
        chi_rank = 999
        for idx, chi in enumerate(STANDARD_CHIRALITIES):
            if col_name.startswith(chi + "_"):
                chi_rank = idx
                break

        # 2. 找特征
        feat_rank = 1
        if "_intensity" in col_name:
            feat_rank = 0
        elif "_shift" in col_name:
            feat_rank = 1

        return (chi_rank, feat_rank, col_name)

    return sorted(cols, key=key_func)


def main():
    # --- 1. 选择文件 ---
    root = tk.Tk()
    root.withdraw()
    print("请选择 4 个 Excel 文件...")

    file_paths = filedialog.askopenfilenames(
        title="请选中 4 个不同手性的 Excel 文件 (Ctrl+点击)",
        filetypes=[("Excel Files", "*.xlsx;*.xls")]
    )

    if not file_paths:
        print("❌ 未选择文件")
        return

    print(f"选中 {len(file_paths)} 个文件，正在提取...")

    # 用于存放处理后的单手性 DataFrame
    dfs_by_chirality = {}
    # 用于存放浓度列（作为基准）
    base_index_df = None

    # --- 2. 循环读取每个文件 ---
    for file_path in file_paths:
        filename = os.path.basename(file_path)

        # 提取手性
        match = re.match(r"(\([\d,]+\)(?:-[A-Z0-9]+)?|S7-\([\d,]+\))", filename)
        if not match:
            print(f"⚠️ 跳过无法识别的文件: {filename}")
            continue
        chirality = match.group(1)

        try:
            # === 读取数据 ===
            # 读取 Intensity
            df_max = pd.read_excel(file_path, sheet_name="Max_Value")
            # 读取 Shift
            df_lor = pd.read_excel(file_path, sheet_name="Lorentz")

            # 找浓度列 (假设第一列是 Name)
            conc_col_name = df_max.columns[0]

            # 找特征列
            # Intensity: 包含 'ratio' 或 '(I-I0)/I0' 或盲选第4列
            int_col = next((c for c in df_max.columns if 'ratio' in str(c).lower() or 'I0' in str(c)), None)
            if not int_col and df_max.shape[1] >= 4: int_col = df_max.columns[3]

            # Shift: 'Δλ'
            shift_col = 'Δλ' if 'Δλ' in df_lor.columns else None

            if not int_col or not shift_col:
                print(f"❌ {filename} 缺少特征列，跳过")
                continue

            # === 构造该手性的数据块 ===
            # 我们只取纯数据值，不带索引，以便横向强制拼接
            current_df = pd.DataFrame()

            # 1. 浓度列 (只保存一次作为基准)
            concentrations = df_max[conc_col_name].values

            # 如果是第一个处理的文件，把它的浓度作为整个表的基准
            if base_index_df is None:
                base_index_df = pd.DataFrame({'Name': concentrations})
            else:
                # 简单校验行数是否一致
                if len(concentrations) != len(base_index_df):
                    print(f"⚠️ 警告: {filename} 的行数与其他文件不一致，可能会导致错位！")

            # 2. 特征列
            current_df[f"{chirality}_intensity"] = df_max[int_col].values
            current_df[f"{chirality}_shift"] = df_lor[shift_col].values

            # 存入字典
            dfs_by_chirality[chirality] = current_df
            print(f"  ✅ 已提取: {chirality}")

        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")

    # --- 3. 横向拼接 (物理拼接) ---
    if dfs_by_chirality and base_index_df is not None:
        # 把所有手性的 DataFrame 放入列表
        list_to_concat = [base_index_df] + list(dfs_by_chirality.values())

        # 横向拼接 (axis=1)
        # 这一步是纯物理拼接：第1行拼第1行，第2行拼第2行...
        final_df = pd.concat(list_to_concat, axis=1)

        # 设置 Name 为索引
        final_df.set_index('Name', inplace=True)

        # --- 4. 清洗与保存 ---
        # 剔除 0 浓度
        # 先把索引转为数字以便判断
        try:
            # 处理 "10", "10_1" 这种情况，只取数字部分
            index_as_num = final_df.index.astype(str).str.extract(r'(\d+)')[0].astype(float).values
            # 过滤
            mask = (index_as_num != 0)
            final_df = final_df[mask]
            print("✂️ 已剔除 0 浓度样本")
        except:
            # 如果转换失败（比如全是字符串），尝试直接删除索引为 0 或 '0' 的行
            if 0 in final_df.index: final_df = final_df.drop(0)
            if '0' in final_df.index: final_df = final_df.drop('0')

        # 列重排
        final_df = final_df[sort_cols_strict(final_df.columns)]

        # 保存
        base_dir = os.path.dirname(file_paths[0])
        save_path = os.path.join(base_dir, "Merged_8_Features_Matrix.csv")
        final_df.to_csv(save_path, index_label="Name")

        msg = (f"处理完成！\n\n"
               f"文件: {save_path}\n"
               f"特征列数: {len(final_df.columns)} (应为 8)\n"
               f"样本行数: {len(final_df)} (每个浓度均为独立行)")
        messagebox.showinfo("成功", msg)

    else:
        messagebox.showerror("错误", "未能生成数据，请检查文件内容。")


if __name__ == "__main__":
    main()