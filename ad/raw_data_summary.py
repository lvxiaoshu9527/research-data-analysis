import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import os
import sys

# ---------------------------------------------------------
# 配置区域：定义关键列名，方便后续维护
# ---------------------------------------------------------
COL_LORENTZ_TARGET = "Δλ"  # Lorentz 表中的目标列
COL_MAX_TARGET = "ΔI/I0_fit_FBS"  # Max_Value 表中的目标列

# 输出到 Excel 时显示的列名（需保持希腊字母）
OUT_COL_LAMBDA = "Δλ"
OUT_COL_INTENSITY = "ΔI/I0"


# ---------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------

def parse_filename(filename):
    """
    从文件名中解析 chirality 和 batch
    文件名格式示例: output_(7,5)_6.xlsx
    """
    pattern = r"output_(.+)_\s*(\d+)\.xlsx"
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def parse_sample_label(label):
    """
    解析样本标签字符串
    格式: (chirality)-(sample_id)-(replicate_id)__laser
    示例: (7,5)-48-1__660
    返回: sample_id (int), replicate_id (int)
    """
    try:
        # 1. 去掉后缀 __laser
        base = label.split('__')[0]

        # 2. 从右侧分割，获取 replicate_id (1) 和 sample_id (48)
        # 使用 rsplit 确保即使 chirality 部分包含 '-' 也能正确分割后两部分
        parts = base.rsplit('-', 2)

        if len(parts) < 3:
            return None, None

        s_id = int(parts[1])
        r_id = int(parts[2])
        return s_id, r_id
    except Exception:
        return None, None


def process_single_file(file_path):
    """
    处理单个 Excel 文件，返回处理后的 DataFrame 列表（可能为空）
    """
    filename = os.path.basename(file_path)
    chirality, batch = parse_filename(filename)

    if not chirality or not batch:
        messagebox.showerror("文件名错误",
                             f"文件 '{filename}' 命名不符合格式 'output_(chirality)_batch.xlsx'，将被跳过。")
        return None

    try:
        # 读取 Excel (只读取需要的 sheet)
        xls = pd.read_excel(file_path, sheet_name=['Lorentz', 'Max_Value'], engine='openpyxl')
    except ValueError:
        # 通常是 Sheet 不存在引发的错误
        messagebox.showerror("Sheet 缺失", f"文件 '{filename}' 中缺失 'Lorentz' 或 'Max_Value' 工作表，将被跳过。")
        return None
    except Exception as e:
        messagebox.showerror("读取错误", f"无法读取文件 '{filename}':\n{str(e)}")
        return None

    df_l = xls['Lorentz']
    df_m = xls['Max_Value']

    # 校验列是否存在
    # 获取第一列列名（通常是 Label，但不确定具体名字，所以按位置取）
    col_label_l = df_l.columns[0]
    col_label_m = df_m.columns[0]

    if COL_LORENTZ_TARGET not in df_l.columns:
        messagebox.showerror("列缺失", f"文件 '{filename}'\nLorentz 表中缺失关键列: {COL_LORENTZ_TARGET}")
        return None
    if COL_MAX_TARGET not in df_m.columns:
        messagebox.showerror("列缺失", f"文件 '{filename}'\nMax_Value 表中缺失关键列: {COL_MAX_TARGET}")
        return None

    # 提取数据
    # 为了合并，先统一列名为 'Full_Label'
    data_l = df_l[[col_label_l, COL_LORENTZ_TARGET]].rename(
        columns={col_label_l: 'Full_Label', COL_LORENTZ_TARGET: 'val_lambda'})
    data_m = df_m[[col_label_m, COL_MAX_TARGET]].rename(
        columns={col_label_m: 'Full_Label', COL_MAX_TARGET: 'val_intensity'})

    # 确保标签是字符串并去空
    data_l['Full_Label'] = data_l['Full_Label'].astype(str).str.strip()
    data_m['Full_Label'] = data_m['Full_Label'].astype(str).str.strip()

    # 合并数据
    merged = pd.merge(data_l, data_m, on='Full_Label', how='outer', indicator=True)

    # 检查匹配情况
    unmatched = merged[merged['_merge'] != 'both']
    if not unmatched.empty:
        messagebox.showwarning("标签不匹配",
                               f"文件 '{filename}' 中存在无法对齐的样本标签（Lorentz 与 Max_Value 不一致）。\n这些数据将被保留，但存在缺失值。")

    # 解析 Sample ID 和 Replicate ID
    parsed_ids = merged['Full_Label'].apply(parse_sample_label)

    # 赋值新列
    merged['chirality'] = chirality
    merged['batch'] = batch
    merged['sample_id'] = parsed_ids.apply(lambda x: x[0] if x else None)
    merged['replicate_id'] = parsed_ids.apply(lambda x: x[1] if x else None)

    # 过滤掉无法解析标签的行 (如果有)
    valid_data = merged.dropna(subset=['sample_id', 'replicate_id']).copy()

    # 类型转换
    valid_data['sample_id'] = valid_data['sample_id'].astype(int)
    valid_data['replicate_id'] = valid_data['replicate_id'].astype(int)

    return valid_data


def main():
    root = tk.Tk()
    root.withdraw()

    # ---------------------------------------------------------
    # Step 1: 多文件选择
    # ---------------------------------------------------------
    file_paths = filedialog.askopenfilenames(
        title="请选择一个或多个实验 Excel 文件",
        filetypes=[("Excel Files", "*.xlsx")]
    )

    if not file_paths:
        print("未选择文件，程序退出。")
        return

    # ---------------------------------------------------------
    # Step 2 & 3: 逐文件解析与聚合
    # ---------------------------------------------------------
    all_data_list = []

    print("开始处理文件...")
    for f_path in file_paths:
        df_processed = process_single_file(f_path)
        if df_processed is not None:
            all_data_list.append(df_processed)

    if not all_data_list:
        messagebox.showerror("错误", "没有有效的数据被提取，程序退出。")
        return

    # 合并所有文件的 DataFrame
    master_df = pd.concat(all_data_list, ignore_index=True)

    # ---------------------------------------------------------
    # Step 4: 准备 Raw_Replicates 数据
    # ---------------------------------------------------------
    # 按照要求重命名列，并排序
    # 列顺序：原始样本标签, chirality, batch, sample_id, replicate_id, Δλ, ΔI/I0
    raw_output_df = master_df.rename(columns={
        'val_lambda': OUT_COL_LAMBDA,
        'val_intensity': OUT_COL_INTENSITY
    })

    raw_cols_order = ['Full_Label', 'chirality', 'batch', 'sample_id', 'replicate_id', OUT_COL_LAMBDA,
                      OUT_COL_INTENSITY]
    raw_output_df = raw_output_df[raw_cols_order].sort_values(by=['chirality', 'batch', 'sample_id', 'replicate_id'])

    # ---------------------------------------------------------
    # Step 6: 异常检查 (平行组数量)
    # ---------------------------------------------------------
    # 按 chirality, batch, sample_id 分组计数
    counts = raw_output_df.groupby(['chirality', 'batch', 'sample_id']).size()
    bad_counts = counts[counts != 3]

    if not bad_counts.empty:
        error_msg = "以下样本的平行组数量不为 3：\n\n"
        count_items = 0
        for idx, count in bad_counts.items():
            chi, bat, sid = idx
            error_msg += f"手性: {chi} | Batch: {bat} | SampleID: {sid} -> 数量: {count}\n"
            count_items += 1
            if count_items >= 15:
                error_msg += "... (更多异常未显示)"
                break
        messagebox.showwarning("平行组数量异常", error_msg)

    # ---------------------------------------------------------
    # Step 5: 生成 Sample_Mean 数据
    # ---------------------------------------------------------
    # 计算均值
    # 注意：Full_Label 在聚合后会丢失，因为它是唯一的。均值表不需要每行的原始 Label，只需要 Sample ID。
    mean_df = raw_output_df.groupby(['chirality', 'batch', 'sample_id'])[
        [OUT_COL_LAMBDA, OUT_COL_INTENSITY]].mean().reset_index()

    # 重命名列以添加 _mean 后缀
    mean_df.rename(columns={
        OUT_COL_LAMBDA: f"{OUT_COL_LAMBDA}_mean",
        OUT_COL_INTENSITY: f"{OUT_COL_INTENSITY}_mean"
    }, inplace=True)

    # 均值表列顺序：sample_id, chirality, batch, Δλ_mean, ΔI/I0_mean
    mean_cols_order = ['sample_id', 'chirality', 'batch', f"{OUT_COL_LAMBDA}_mean", f"{OUT_COL_INTENSITY}_mean"]
    mean_df = mean_df[mean_cols_order].sort_values(by=['chirality', 'batch', 'sample_id'])

    # ---------------------------------------------------------
    # Step 7: 输出结果
    # ---------------------------------------------------------
    # 选择保存目录
    save_dir = filedialog.askdirectory(title="请选择结果保存目录")
    if not save_dir:
        return

    path_raw = os.path.join(save_dir, "Raw_Replicates.xlsx")
    path_mean = os.path.join(save_dir, "Sample_Mean.xlsx")

    try:
        # 获取所有出现过的手性
        unique_chiralities = raw_output_df['chirality'].unique()

        # 写入 Raw_Replicates.xlsx
        with pd.ExcelWriter(path_raw, engine='openpyxl') as writer:
            for chi in unique_chiralities:
                # 筛选该手性的数据
                sheet_data = raw_output_df[raw_output_df['chirality'] == chi]
                # 写入 Sheet，Sheet 名即为手性
                sheet_name = str(chi).replace('/', '_')  # 防止手性中有非法字符
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

        # 写入 Sample_Mean.xlsx
        with pd.ExcelWriter(path_mean, engine='openpyxl') as writer:
            for chi in unique_chiralities:
                sheet_data = mean_df[mean_df['chirality'] == chi]
                sheet_name = str(chi).replace('/', '_')
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

        messagebox.showinfo("成功", f"处理完成！\n已生成以下文件：\n1. {path_raw}\n2. {path_mean}")

    except Exception as e:
        messagebox.showerror("保存失败", f"写入 Excel 时发生错误：\n{str(e)}\n请检查文件是否被占用。")


if __name__ == "__main__":
    main()