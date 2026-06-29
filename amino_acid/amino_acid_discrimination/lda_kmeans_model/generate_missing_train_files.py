import os
import re
import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
import glob
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from typing import List, Tuple, Dict, Any, Optional


# --- GUI 功能函数 ---

def select_directory(title: str) -> str:
    """打开一个对话框让用户选择一个文件夹。"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path


def select_file(title: str, filetypes: list) -> str:
    """打开一个对话框让用户选择一个文件。"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


# --- 从全面聚类优化.txt 复制的必要函数 ---
# 确保与原始训练脚本完全一致，尤其是标准化和特征工程的顺序。

def preprocess_data(df: pd.DataFrame, feature_cols: List[str], label_col: str = 'AA', missing_threshold: float = 0.2) -> \
        Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    预处理数据：处理缺失值，获取编码标签和原始标签名称。
    """
    required_cols = [label_col] + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 中缺少预期的列: {', '.join(missing_cols)}")

    df.dropna(subset=[label_col], inplace=True)
    df_features_numeric = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df_filtered_missing = df[df_features_numeric.isnull().sum(axis=1) / len(feature_cols) <= missing_threshold]
    rows_dropped_missing = df.shape[0] - df_filtered_missing.shape[0]
    if rows_dropped_missing > 0:
        print(f"因超过 {missing_threshold * 100}% 的特征缺失，丢弃了 {rows_dropped_missing} 个样本。")
    df = df_filtered_missing.copy()

    if df.empty:
        raise ValueError("丢弃缺失值后 DataFrame 为空。")

    y_original_labels = df[label_col].copy()

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y_original_labels), index=y_original_labels.index)
    unique_original_labels = list(label_encoder.classes_)

    if len(unique_original_labels) < 2:
        raise ValueError("处理后的数据包含少于 2 个独特的氨基酸。无法进行聚类。")

    return df.index, y_encoded, unique_original_labels


def engineer_features(df_original_features_raw: pd.DataFrame, chiralities: List[str]) -> pd.DataFrame:
    """
    根据原始强度和位移数据生成新特征，并**内部进行标准化**。
    此函数与原始训练脚本中的 engineer_features 完全一致。
    """
    df_engineered = df_original_features_raw.copy()

    for i in range(len(chiralities)):
        for j in range(i + 1, len(chiralities)):
            c1, c2 = chiralities[i], chiralities[j]
            col1_int = f'{c1}_intensity'
            col2_int = f'{c2}_intensity'
            if col1_int in df_engineered.columns and col2_int in df_engineered.columns:
                df_engineered[f'{c1}I_div_{c2}I'] = df_engineered[col1_int] / df_engineered[col2_int].replace(0, np.nan)
                df_engineered[f'{c2}I_div_{c1}I'] = df_engineered[col2_int] / df_engineered[col1_int].replace(0, np.nan)

    for i in range(len(chiralities)):
        for j in range(i + 1, len(chiralities)):
            c1, c2 = chiralities[i], chiralities[j]
            col1_shift = f'{c1}_shift'
            col2_shift = f'{c2}_shift'
            if col1_shift in df_engineered.columns and col2_shift in df_engineered.columns:
                df_engineered[f'{c1}S_minus_{c2}S'] = df_engineered[col1_shift] - df_engineered[col2_shift]

    for c in chiralities:
        intensity_col = f'{c}_intensity'
        shift_col = f'{c}_shift'
        if intensity_col in df_engineered.columns and shift_col in df_engineered.columns:
            df_engineered[f'{c}I_div_{c}S'] = df_engineered[intensity_col] / df_engineered[shift_col].replace(0, np.nan)

    intensity_cols_present = [f'{c}_intensity' for c in chiralities if f'{c}_intensity' in df_engineered.columns]
    if intensity_cols_present:
        max_intensity_per_row = df_engineered[intensity_cols_present].max(axis=1)
        max_intensity_per_row = max_intensity_per_row.replace(0, np.nan)
        for col in intensity_cols_present:
            df_engineered[f'{col}_norm'] = df_engineered[col] / max_intensity_per_row
            df_engineered[f'{col}_norm'].replace([np.inf, -np.inf], np.nan, inplace=True)

    df_engineered.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_engineered_scaled = pd.DataFrame(scaler.fit_transform(df_engineered), columns=df_engineered.columns,
                                       index=df_engineered.index)
    return X_engineered_scaled


def remap_labels_for_subset(y_encoded: pd.Series, unique_original_labels: List[str],
                            amino_acid_subset_names: List[str]) -> Tuple[pd.Series, List[str]]:
    """
    为给定的氨基酸子集重新映射标签。
    """
    subset_encoded_values = [
        unique_original_labels.index(aa_name)
        for aa_name in amino_acid_subset_names
        if aa_name in unique_original_labels
    ]
    filtered_indices = y_encoded[y_encoded.isin(subset_encoded_values)].index
    y_filtered_encoded = y_encoded.loc[filtered_indices]
    actual_subset_original_labels = [unique_original_labels[val] for val in sorted(y_filtered_encoded.unique())]
    new_label_mapping = {old_val: new_val for new_val, old_val in enumerate(sorted(y_filtered_encoded.unique()))}
    y_remapped = y_filtered_encoded.map(new_label_mapping)
    return y_remapped, actual_subset_original_labels


# --- 主逻辑 ---

def main():
    print("--- 辅助脚本: 生成缺失的训练可视化文件 ---")
    print("此脚本将加载您的原始训练数据和已保存的模型，重建训练数据在 LDA 空间中的投影和标签，并保存它们。")

    # 1. 引导用户选择路径
    model_dir = select_directory(
        title="请选择包含模型文件 (maximally_separable_combination.json 和 .pkl 文件) 的文件夹")
    if not model_dir:
        messagebox.showerror("错误", "未选择模型文件夹，程序退出。")
        return

    original_train_data_path = select_file(title="请选择原始训练数据 CSV 文件", filetypes=[("CSV files", "*.csv")])
    if not original_train_data_path:
        messagebox.showerror("错误", "未选择原始训练数据文件，程序退出。")
        return

    output_dir = select_directory(
        title="请选择保存生成文件 (X_train_lda_for_plot.npy, y_train_for_plot.csv, feature_columns.csv) 的文件夹")
    if not output_dir:
        messagebox.showerror("错误", "未选择结果保存文件夹，程序退出。")
        return

    json_path = os.path.join(model_dir, "maximally_separable_combination.json")
    if not os.path.exists(json_path):
        messagebox.showerror("错误",
                             f"在所选文件夹中未找到 'maximally_separable_combination.json' 文件。\n请确保该文件存在，它是识别正确模型和特征的关键。")
        return

    try:
        # 2. 加载 maximally_separable_combination.json 获取模型路径和组合信息
        print(f"\n🔄 正在加载模型元数据: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            model_metadata = json.load(f)

        dim_reduction_model_path = os.path.join(model_dir, os.path.basename(model_metadata['dim_reduction_model_path']))
        amino_acid_list_for_combo = model_metadata['amino_acid_list']
        expected_feature_names_for_lda_input = model_metadata['feature_names_for_subset']

        print(f"✅ 模型元数据加载成功。组合氨基酸: {amino_acid_list_for_combo}")
        print(f"预计工程化并标准化后的特征数 (LDA输入): {len(expected_feature_names_for_lda_input)}")

        # 3. 加载 LDA 模型
        print("\n🔄 正在加载 LDA 模型...")
        lda_model = joblib.load(dim_reduction_model_path)
        print("✅ LDA 模型加载成功。")

    except Exception as e:
        messagebox.showerror("加载模型/元数据错误", f"加载文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # 4. 加载原始训练数据并重新进行预处理和特征工程 (与原始训练脚本完全一致)
        print(f"\n🔄 正在加载并处理原始训练数据: {original_train_data_path}")
        df_original = pd.read_csv(original_train_data_path)

        chiralities = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']
        original_feature_columns = [f'{c}_intensity' for c in chiralities] + [f'{c}_shift' for c in chiralities]
        label_column = 'AA'

        # 预处理：处理缺失值并获取 y_full_encoded 和 all_amino_acids_names
        filtered_df_indices, y_full_encoded, all_amino_acids_names = preprocess_data(
            df_original.copy(), original_feature_columns, label_column, missing_threshold=0.2)

        df_filtered_by_labels = df_original.loc[filtered_df_indices].copy()

        print("\n--- 重新执行特征工程 (包含内部标准化) ---")

        df_original_features_raw_for_engineering = df_filtered_by_labels[original_feature_columns].copy()

        X_full_scaled = engineer_features(df_original_features_raw_for_engineering, chiralities)

        if not X_full_scaled.columns.tolist() == expected_feature_names_for_lda_input:
            messagebox.showwarning("警告", "重新生成的特征列与模型期望的特征列不完全匹配。尝试对齐。")

            X_aligned_for_lda = pd.DataFrame(index=X_full_scaled.index, columns=expected_feature_names_for_lda_input)

            common_cols = [col for col in expected_feature_names_for_lda_input if col in X_full_scaled.columns]
            X_aligned_for_lda[common_cols] = X_full_scaled[common_cols]

            X_aligned_for_lda.fillna(0, inplace=True)

            X_full_scaled = X_aligned_for_lda

            if X_full_scaled.shape[
                1] != lda_model.n_features_in_ or not X_full_scaled.columns.tolist() == expected_feature_names_for_lda_input:
                messagebox.showerror("严重错误",
                                     "特征对齐失败，无法继续 LDA 转换。请检查原始训练脚本中的特征工程和保存的 feature_names_for_subset 是否完全一致。")
                return

        print(f"✅ 原始训练数据已加载并重新处理。最终工程化+标准化特征维度: {X_full_scaled.shape[1]}")
        print(f"总样本数: {X_full_scaled.shape[0]}")

    except Exception as e:
        messagebox.showerror("数据处理错误", f"处理原始训练数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # 5. 选择组合子集并进行 LDA 转换
        print(f"\n🔄 正在为组合 {amino_acid_list_for_combo} 选择子集并进行 LDA 转换...")

        y_subset_remapped, original_subset_labels_names = remap_labels_for_subset(
            y_full_encoded, all_amino_acids_names, list(amino_acid_list_for_combo)
        )
        X_subset_scaled = X_full_scaled.loc[y_subset_remapped.index]

        X_train_lda_for_plot = lda_model.transform(X_subset_scaled)
        print("✅ LDA 转换完成。")

        # 6. 保存生成的文件
        print("\n💾 正在保存生成的文件...")

        output_npy_path = os.path.join(output_dir, "X_train_lda_for_plot.npy")
        np.save(output_npy_path, X_train_lda_for_plot)
        print(f"✅ X_train_lda_for_plot.npy 已保存到: {output_npy_path}")

        # **关键修改：y_train_for_plot.csv 中保存编码后的数字标签**
        y_train_for_plot_df = y_subset_remapped.to_frame(name='AA_Label_Encoded')  # 保存数字标签
        output_y_csv_path = os.path.join(output_dir, "y_train_for_plot.csv")
        y_train_for_plot_df.to_csv(output_y_csv_path, index=False)
        print(f"✅ y_train_for_plot.csv 已保存到 (包含编码后的数字标签): {output_y_csv_path}")

        # 同时保存一个映射文件，以便在需要时查找原始名称
        label_mapping_df = pd.DataFrame(
            {'Encoded_ID': range(len(all_amino_acids_names)), 'Amino_Acid_Name': all_amino_acids_names})
        label_mapping_df.to_csv(os.path.join(output_dir, "amino_acid_label_mapping_for_plot.csv"), index=False)
        print(f"✅ 氨基酸标签映射已保存到: {os.path.join(output_dir, 'amino_acid_label_mapping_for_plot.csv')}")

        feature_columns_df = pd.DataFrame({'Feature': expected_feature_names_for_lda_input})
        output_feature_csv_path = os.path.join(output_dir, "feature_columns.csv")
        feature_columns_df.to_csv(output_feature_csv_path, index=False)
        print(f"✅ feature_columns.csv 已保存到: {output_feature_csv_path}")

        messagebox.showinfo("成功",
                            f"所有缺失文件已成功生成并保存到:\n{output_dir}\n\n现在您可以运行预测脚本了！")

    except Exception as e:
        messagebox.showerror("生成文件错误", f"生成文件时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
