import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
import glob
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import adjusted_rand_score, silhouette_score  # For metrics calculation
from typing import List, Tuple, Dict, Any, Optional
import seaborn as sns


# Suppress all warnings for cleaner output (optional, for final presentation)
# from warnings import filterwarnings
# filterwarnings("ignore", category=UserWarning)
# filterwarnings("ignore", category=FutureWarning)
# filterwarnings("ignore", category=DeprecationWarning)

# --- GUI Utility Functions ---

def select_directory(title: str) -> str:
    """Opens a dialog for the user to select a directory."""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path


def select_file(title: str, filetypes: list) -> str:
    """Opens a dialog for the user to select a file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


# --- Feature Engineering Function (copied from original training script) ---
# This function is crucial as it includes the internal scaling logic used during training.
def engineer_features(df_original_features_raw: pd.DataFrame, chiralities: List[str]) -> pd.DataFrame:
    """
    Generates new features from raw intensity and shift data, and performs internal standardization.
    This function must be identical to the one used in the original training script.
    """
    df_engineered = df_original_features_raw.copy()

    for i in range(len(chiralities)):
        for j in range(i + 1, len(chiralities)):
            c1, c2 = chiralities[i], chiralities[j]
            col1_int = f'{c1}_intensity'
            col2_int = f'{c2}_intensity'
            if col1_int in df_engineered.columns and col2_int in df_engineered.columns:
                # Handle division by zero carefully
                denominator_1 = df_engineered[col2_int].replace(0, np.nan)
                df_engineered[f'{c1}I_div_{c2}I'] = df_engineered[col1_int] / denominator_1

                denominator_2 = df_engineered[col1_int].replace(0, np.nan)
                df_engineered[f'{c2}I_div_{c1}I'] = df_engineered[col2_int] / denominator_2

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
            denominator = df_engineered[shift_col].replace(0, np.nan)
            df_engineered[f'{c}I_div_{c}S'] = df_engineered[intensity_col] / denominator

    intensity_cols_present = [f'{c}_intensity' for c in chiralities if f'{c}_intensity' in df_engineered.columns]
    if intensity_cols_present:
        max_intensity_per_row = df_engineered[intensity_cols_present].max(axis=1)
        max_intensity_per_row = max_intensity_per_row.replace(0, np.nan)  # Avoid division by zero
        for col in intensity_cols_present:
            df_engineered[f'{col}_norm'] = df_engineered[col] / max_intensity_per_row
            # Replace inf/ -inf values with NaN for consistency before final fillna
            df_engineered[f'{col}_norm'] = df_engineered[f'{col}_norm'].replace([np.inf, -np.inf], np.nan)

    df_engineered.fillna(0, inplace=True)  # Fill any remaining NaNs (e.g., from divisions) with 0

    # Key: Perform internal standardization as done in the training script
    scaler = StandardScaler()
    X_engineered_scaled = pd.DataFrame(scaler.fit_transform(df_engineered), columns=df_engineered.columns,
                                       index=df_engineered.index)
    return X_engineered_scaled


# --- Plotting Function for Test Data Cluster Visualization ---

def plot_test_prediction_results(
        X_test_lda: np.ndarray,
        predicted_labels: np.ndarray,
        output_filename: str
):
    """
    Visualizes the prediction results for test data in LDA space.
    Only shows test data, as requested by the user.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))  # Adjusted figure size

    # Use Tab20 colormap for clusters
    unique_clusters = np.unique(predicted_labels)
    colors = sns.color_palette('tab20', n_colors=max(len(unique_clusters), 20))

    # Plot test data scatter points
    if X_test_lda.shape[0] > 0:
        scatter_colors_test = [colors[int(label) % len(colors)] for label in predicted_labels]

        scatter = plt.scatter(
            X_test_lda[:, 0],
            X_test_lda[:, 1],
            c=scatter_colors_test,
            marker='o',  # Changed marker to circle for clarity in test-only plot
            s=80,  # Adjusted size
            linewidth=0.8,
            edgecolor='black',
            alpha=0.9
        )
    else:
        print("Warning: No test samples available for visualization.")
        messagebox.showwarning("Warning", "No test samples available for visualization. Plot will not be generated.")
        return  # Exit if no data to plot

    plt.title("Test Samples Clustered in LDA Space", fontsize=16, pad=20)
    plt.xlabel("LDA Component 1", fontsize=12)
    plt.ylabel("LDA Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Create custom legend for clusters
    handles = []
    labels = []
    for cluster_id in sorted(unique_clusters):
        handle = plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=colors[int(cluster_id) % len(colors)],
                            markeredgecolor='black', markeredgewidth=0.8,
                            markersize=10, label=f'Predicted Cluster {cluster_id}')
        handles.append(handle)
        labels.append(f'Predicted Cluster {cluster_id}')

    plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left',
               borderaxespad=0.)  # Place legend outside plot
    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)
    plt.savefig(output_filename.replace(".png", ".svg"), format='svg')  # Save as SVG as well
    plt.close()

    print(f"✅ Prediction visualization saved to: {output_filename}")


# --- Main Execution Logic ---

def main():
    """Main function to perform the prediction workflow."""
    print("--- LDA + KMeans Test Data Prediction Tool ---")

    # 1. Guide user to select paths
    model_dir = select_directory(title="请选择包含已训练模型的文件夹")
    if not model_dir:
        messagebox.showerror("错误", "未选择模型文件夹，程序退出。")
        return

    test_data_path = select_file(title="请选择要预测的测试数据 (CSV)", filetypes=[("CSV files", "*.csv")])
    if not test_data_path:
        messagebox.showerror("错误", "未选择测试数据文件，程序退出。")
        return

    output_dir = select_directory(title="请选择保存结果的文件夹")
    if not output_dir:
        messagebox.showerror("错误", "未选择结果保存文件夹，程序退出。")
        return

    # Initialize variables
    lda_model = None
    kmeans_model = None
    expected_features_for_lda_input = None
    amino_acid_list_from_model = None  # To store the list of amino acids the model was trained on

    try:
        # 2. Load models and expected feature list
        print(f"\n🔄 正在从 '{model_dir}' 加载模型...")

        # Load maximally_separable_combination.json for model metadata
        json_path_for_metadata = os.path.join(model_dir, "maximally_separable_combination.json")
        if not os.path.exists(json_path_for_metadata):
            messagebox.showerror("Error",
                                 f"未找到 'maximally_separable_combination.json'。\n请确保该文件存在，它是识别正确模型和特征的关键。")
            return

        with open(json_path_for_metadata, 'r', encoding='utf-8') as f:
            model_metadata = json.load(f)

        # Get paths for LDA and KMeans models from metadata, and expected feature list
        dim_reduction_model_path = os.path.join(model_dir, os.path.basename(model_metadata['dim_reduction_model_path']))
        expected_features_for_lda_input = model_metadata['feature_names_for_subset']
        amino_acid_list_from_model = model_metadata.get('amino_acid_list',
                                                        [])  # Get amino acid list, default to empty list

        # Load KMeans model using glob pattern to find the latest .pkl file
        kmeans_files = glob.glob(os.path.join(model_dir, "*_kmeans_clusterer.pkl"))
        if kmeans_files:
            kmeans_model = joblib.load(max(kmeans_files, key=os.path.getmtime))
            print(f"✅ 已加载 KMeans 模型: {os.path.basename(max(kmeans_files, key=os.path.getmtime))}")
        else:
            messagebox.showerror("文件缺失", "未找到 KMeans 聚类器 (.pkl) 文件。预测无法进行。")
            return

        # Load LDA model from its specific path
        if os.path.exists(dim_reduction_model_path):
            lda_model = joblib.load(dim_reduction_model_path)
            print(f"✅ 已加载 LDA 模型: {os.path.basename(dim_reduction_model_path)}")
        else:
            messagebox.showerror("文件缺失",
                                 f"未找到 LDA 降维模型文件: {os.path.basename(dim_reduction_model_path)}。预测无法进行。")
            return

        print(f"模型期望 {len(expected_features_for_lda_input)} 个特征作为 LDA 输入。")
        if amino_acid_list_from_model:
            print(f"模型训练时的氨基酸列表: {amino_acid_list_from_model}")

    except Exception as e:
        messagebox.showerror("模型加载错误", f"加载模型文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # 3. Load and Preprocess Test Data
        print(f"\n🔄 正在加载并处理测试数据: {test_data_path}")

        # --- FIX: Try common encodings for pd.read_csv ---
        try:
            df_test_raw = pd.read_csv(test_data_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_test_raw = pd.read_csv(test_data_path, encoding='GBK')
                print("尝试GBK编码成功。")
            except UnicodeDecodeError:
                try:
                    df_test_raw = pd.read_csv(test_data_path, encoding='latin1')
                    print("尝试latin1编码成功。")
                except Exception as encoding_e:
                    messagebox.showerror("编码错误",
                                         f"无法使用UTF-8、GBK或latin1编码读取文件。请检查文件编码: {encoding_e}")
                    import traceback
                    traceback.print_exc()
                    return
        except Exception as read_e:
            messagebox.showerror("文件读取错误", f"读取CSV文件时发生错误: {read_e}")
            import traceback
            traceback.print_exc()
            return
        # --- END FIX ---

        # --- Debugging: Print actual column names loaded from test data ---
        print("\n--- 调试信息：加载的原始测试数据列名 ---")
        print(df_test_raw.columns.tolist())
        print("-----------------------------------------------------")
        # --- Debug Info End ---

        df_test_processed = df_test_raw.copy()  # Start with a copy for processing

        # --- Filter test data by amino acid list from model ---
        if 'AA' in df_test_processed.columns and amino_acid_list_from_model:
            initial_sample_count = len(df_test_processed)
            # Drop rows where 'AA' is NaN or empty string before filtering
            df_test_processed.dropna(subset=['AA'], inplace=True)
            samples_after_aa_dropna = len(df_test_processed)

            df_test_processed = df_test_processed[
                df_test_processed['AA'].isin(amino_acid_list_from_model)
            ].copy()
            aa_filtered_count = len(df_test_processed)
            print(f"原始测试样本数: {initial_sample_count}")
            print(f"去除AA缺失样本后: {samples_after_aa_dropna}")
            print(
                f"根据模型训练列表过滤氨基酸后: {aa_filtered_count} (过滤了 {samples_after_aa_dropna - aa_filtered_count} 个样本)")

            if aa_filtered_count == 0:
                messagebox.showwarning("无可用数据", "所有测试样本都因氨基酸类型不在模型训练列表内而被过滤。请检查数据。")
                return

        # Handle 'Sample_ID' or 'AA' column for indexing for the *processed* df
        # This ensures 'AA' column remains available for True_Label if it's not the unique ID.
        if 'AA' in df_test_processed.columns:
            # If 'AA' is unique, use it as Sample_ID. Otherwise, generate new Sample_ID.
            if df_test_processed['AA'].duplicated().any():
                df_test_processed['Sample_ID'] = [f'TestSample_{i + 1}' for i in range(len(df_test_processed))]
                # Keep 'AA' column for True_Label access later
                df_test_final = df_test_processed.set_index('Sample_ID')
            else:
                # 'AA' is unique, rename it to Sample_ID and use as index
                df_test_processed.rename(columns={'AA': 'Sample_ID'}, inplace=True)
                df_test_final = df_test_processed.set_index('Sample_ID')
        elif 'Sample_ID' in df_test_processed.columns:
            # If Sample_ID already exists, just set it as index
            df_test_final = df_test_processed.set_index('Sample_ID')
        else:
            # No 'AA' or 'Sample_ID', generate new Sample_ID
            df_test_processed['Sample_ID'] = [f'TestSample_{i + 1}' for i in range(len(df_test_processed))]
            df_test_final = df_test_processed.set_index('Sample_ID')

        # From now on, use df_test_final which has Sample_ID as index and potentially 'AA' as a column.

        # Define expected raw feature column groups (must match training script)
        chiralities = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']
        original_intensity_cols_expected = [f'{c}_intensity' for c in chiralities]
        original_shift_cols_expected = [f'{c}_shift' for c in chiralities]
        all_original_feature_columns_expected = original_intensity_cols_expected + original_shift_cols_expected

        # Create a DataFrame for raw features, initialized with expected columns
        X_test_raw_features = pd.DataFrame(columns=all_original_feature_columns_expected, index=df_test_final.index)

        # Populate with data from df_test_final where columns exist
        actual_present_raw_features = [col for col in all_original_feature_columns_expected if
                                       col in df_test_final.columns]
        X_test_raw_features[actual_present_raw_features] = df_test_final[actual_present_raw_features]

        missing_raw_features_in_test_df = [col for col in all_original_feature_columns_expected if
                                           col not in df_test_final.columns]
        if missing_raw_features_in_test_df:
            messagebox.showwarning("警告",
                                   f"测试数据中缺少模型期望的原始特征列: {missing_raw_features_in_test_df}。这些将填充 0 进行特征工程。")

        # Ensure all raw feature columns are numeric and fill any remaining NaNs with 0
        X_test_raw_features = X_test_raw_features.apply(pd.to_numeric, errors='coerce').fillna(0)

        # --- Perform Feature Engineering (includes internal standardization) ---
        print("\n🔬 正在对测试数据进行特征工程 (含内部标准化)...")
        X_test_engineered_and_scaled = engineer_features(X_test_raw_features, chiralities)
        print("✅ 特征工程完成。")

        # --- Feature Alignment for LDA Input ---
        print("\n🔬 正在将测试数据特征与模型期望的最终特征进行对齐...")

        # Create an empty DataFrame with columns in the exact order expected by the LDA model
        X_test_aligned_for_lda = pd.DataFrame(columns=expected_features_for_lda_input,
                                              index=X_test_engineered_and_scaled.index)

        # Populate with data from X_test_engineered_and_scaled where columns match
        common_features_for_lda = [col for col in expected_features_for_lda_input if
                                   col in X_test_engineered_and_scaled.columns]
        X_test_aligned_for_lda[common_features_for_lda] = X_test_engineered_and_scaled[common_features_for_lda]

        # Fill any newly introduced NaNs (e.g., if a feature expected by the model wasn't generated)
        X_test_aligned_for_lda.fillna(0, inplace=True)

        print(f"测试数据中包含 {len(common_features_for_lda)} 个模型期望的 LDA 输入特征。")
        missing_from_test_for_lda = len(expected_features_for_lda_input) - len(common_features_for_lda)
        if missing_from_test_for_lda > 0:
            print(f"测试数据缺少 {missing_from_test_for_lda} 个 LDA 输入特征。这些将填充 0。")

        # Ensure column order is exactly as expected by the LDA model
        X_test_processed_for_lda = X_test_aligned_for_lda[expected_features_for_lda_input]

        # Filter out samples with any remaining NaN values (should be minimal after fillna(0))
        original_sample_count_before_final_drop = len(X_test_processed_for_lda)
        X_test_complete = X_test_processed_for_lda.dropna(axis=0, how='any')
        retained_sample_count = len(X_test_complete)

        print(f"\n样本筛选结果:")
        print(f"  - 原始样本数 (经氨基酸过滤和初始处理后): {original_sample_count_before_final_drop}")
        print(f"  - 特征完整的样本数: {retained_sample_count}")
        print(f"  - 因特征缺失被丢弃的样本数: {original_sample_count_before_final_drop - retained_sample_count}")

        if retained_sample_count == 0:
            messagebox.showwarning("无可用数据", "所有测试样本都因缺少或不完整特征而被过滤。请检查原始测试数据和列名。")
            return

        # 4. Perform Prediction
        print("\n🤖 正在对完整样本执行降维和聚类预测...")

        try:
            X_test_lda = lda_model.transform(X_test_complete)
            predicted_labels = kmeans_model.predict(X_test_lda)
            print("✅ 预测完成！")
        except Exception as pred_e:
            messagebox.showerror("预测错误", f"执行预测时出错，请检查模型和数据是否兼容: {pred_e}")
            import traceback
            traceback.print_exc()
            return

        # 5. Save Core Prediction Results
        print("\n💾 正在保存核心预测结果 (LDA投影坐标和聚类标签)...")
        test_lda_projection_path = os.path.join(output_dir, "test_lda_projection.npy")
        np.save(test_lda_projection_path, X_test_lda)
        print(f"✅ 测试样本 LDA 投影坐标已保存到: {test_lda_projection_path}")

        test_cluster_labels_df = pd.DataFrame({
            "SampleID": X_test_complete.index,
            "Cluster": predicted_labels
        })
        test_cluster_labels_csv_path = os.path.join(output_dir, "test_cluster_labels.csv")
        test_cluster_labels_df.to_csv(test_cluster_labels_csv_path, index=False)
        print(f"✅ 测试样本聚类预测标签已保存到: {test_cluster_labels_csv_path}")

        # --- Part 1: Generate Sample Assignment Table ---
        print("\n📊 正在生成单样本归属分析表格...")
        sample_assignment_df = pd.DataFrame({
            'SampleID': X_test_complete.index,
            'Predicted_Cluster': predicted_labels
        })

        # Get True_Label for the complete samples (ensuring 'AA' column exists)
        if 'AA' in df_test_final.columns:
            true_labels_for_complete_samples = df_test_final.loc[X_test_complete.index, 'AA']
            sample_assignment_df['True_Label'] = true_labels_for_complete_samples

            # Calculate Match column
            # Find the mode of True_Label for each cluster
            cluster_modes = sample_assignment_df.groupby('Predicted_Cluster')['True_Label'].agg(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan  # Use mode[0] for first mode if multiple
            ).to_dict()

            sample_assignment_df['Match'] = sample_assignment_df.apply(
                lambda row: row['True_Label'] == cluster_modes.get(row['Predicted_Cluster']), axis=1
            )

            sample_assignment_table_path = os.path.join(output_dir, "sample_assignment_table.csv")
            sample_assignment_df.to_csv(sample_assignment_table_path, index=False)
            print(f"✅ 单样本归属分析表格已保存到: {sample_assignment_table_path}")
        else:
            print("警告: 无法生成 'True_Label' 和 'Match' 列，因为测试数据中没有 'AA' 列。")
            messagebox.showwarning("警告", "无法生成 'True_Label' 和 'Match' 列，因为测试数据中没有 'AA' 列。")
            # Save without True_Label and Match if 'AA' is missing
            sample_assignment_table_path = os.path.join(output_dir, "sample_assignment_table_no_AA.csv")
            sample_assignment_df.to_csv(sample_assignment_table_path, index=False)
            print(f"✅ 单样本归属分析表格 (无AA列) 已保存到: {sample_assignment_table_path}")

        # --- Part 2: Generate Cluster Composition & Physico-chemical Property Analysis ---
        print("\n📊 正在生成簇组成和理化属性分析...")

        # Combine predicted labels with original data for analysis
        analysis_df = df_test_final.loc[X_test_complete.index].copy()  # Ensure df_test_final has 'AA' and properties
        analysis_df['Predicted_Cluster'] = predicted_labels

        # 2.1 Cluster Label Composition Plot
        if 'AA' in analysis_df.columns:
            print("正在绘制每个簇的真实氨基酸标签组成图...")
            unique_predicted_clusters = sorted(analysis_df['Predicted_Cluster'].unique())

            # Decide on grid size for subplots
            n_clusters = len(unique_predicted_clusters)
            n_cols = min(n_clusters, 3)  # Max 3 columns for subplots
            n_rows = int(np.ceil(n_clusters / n_cols))

            plt.figure(figsize=(n_cols * 5, n_rows * 4))  # Adjust figure size dynamically
            for i, cluster_id in enumerate(unique_predicted_clusters):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                cluster_data = analysis_df[analysis_df['Predicted_Cluster'] == cluster_id]
                label_counts = cluster_data['AA'].value_counts().sort_index()

                # FIX: Pass hue explicitly to avoid FutureWarning
                sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax, palette='viridis',
                            hue=label_counts.index, legend=False)
                ax.set_title(f'Cluster {cluster_id} Label Composition ({len(cluster_data)} samples)')
                ax.set_xlabel('Amino Acid Label')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            cluster_composition_plot_path = os.path.join(output_dir, "cluster_label_composition.png")
            plt.savefig(cluster_composition_plot_path, dpi=300)
            plt.close()
            print(f"✅ 簇真实标签组成图已保存到: {cluster_composition_plot_path}")
        else:
            print("警告: 无法绘制簇组成图，因为测试数据中没有 'AA' 列。")

        # 2.2 Physico-chemical properties statistics and plot
        # Define expected physico-chemical property columns (customize as needed)
        # Please ensure these columns exist in your test_data.csv if you want this analysis
        phys_chem_properties = ['Hydrophobicity', 'pI', 'Molecular_Weight']

        # Check which properties are actually in the test data
        available_properties = [prop for prop in phys_chem_properties if prop in analysis_df.columns]

        if available_properties:
            print("正在计算理化属性统计量...")
            # Calculate mean and std dev for available properties per cluster
            # Ensure properties are numeric
            for prop in available_properties:
                analysis_df[prop] = pd.to_numeric(analysis_df[prop], errors='coerce')

            property_stats = analysis_df.groupby('Predicted_Cluster')[available_properties].agg(
                ['mean', 'std']).reset_index()
            # Flatten multi-level columns
            property_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in
                                      property_stats.columns.values]

            cluster_property_stats_path = os.path.join(output_dir, "cluster_property_stats.csv")
            property_stats.to_csv(cluster_property_stats_path, index=False)
            print(f"✅ 理化属性统计表格已保存到: {cluster_property_stats_path}")

            print("正在绘制理化属性均值±误差棒图...")
            # Plotting mean with error bars
            plt.figure(figsize=(12, 6))
            bar_width = 0.2

            cluster_ids = property_stats['Predicted_Cluster'].values

            for i, prop in enumerate(available_properties):
                means = property_stats[f'{prop}_mean'].values
                stds = property_stats[f'{prop}_std'].values

                # Adjust x position for bars
                x_pos = np.arange(len(cluster_ids)) + i * bar_width

                plt.bar(x_pos, means, yerr=stds, width=bar_width, label=prop, capsize=5,
                        color=sns.color_palette('pastel')[i % len(sns.color_palette('pastel'))],
                        edgecolor='black')

            plt.xlabel("Predicted Cluster")
            plt.ylabel("Value")
            plt.title("Cluster Physico-chemical Property Summary (Mean ± Std Dev)")
            plt.xticks(np.arange(len(cluster_ids)) + (len(available_properties) - 1) * bar_width / 2, cluster_ids)
            plt.legend(title="Property", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            cluster_property_summary_path = os.path.join(output_dir, "cluster_property_summary.png")
            plt.savefig(cluster_property_summary_path, dpi=300)
            plt.close()
            print(f"✅ 理化属性均值图已保存到: {cluster_property_summary_path}")

        else:
            print("警告: 测试数据中未找到指定的理化属性列，跳过理化属性分析和绘图。")
            messagebox.showwarning("警告", "测试数据中未找到指定的理化属性列，跳过理化属性分析和绘图。")

        # --- Part 3: Calculate ARI and Silhouette Score ---
        print("\n📈 正在计算聚类评价指标 (ARI 和 Silhouette Score)...")

        metrics_data = []  # Use a list of dicts to build metrics_df

        # Check if 'AA' column is available for True_Label for ARI calculation
        if 'AA' in df_test_final.columns:
            # Only include samples that have complete features and valid AA labels
            true_labels_for_metrics = df_test_final.loc[X_test_complete.index, 'AA']

            # Encode True_Labels to numerical format for ARI calculation
            label_encoder_for_ari = LabelEncoder()
            # Handle potential NaNs or non-string types in true_labels_for_metrics before encoding
            true_labels_for_metrics_clean = true_labels_for_metrics.astype(str).dropna()

            # Filter predicted_labels to match the cleaned true_labels_for_metrics index
            predicted_labels_for_ari = pd.Series(predicted_labels, index=X_test_complete.index).loc[
                true_labels_for_metrics_clean.index]

            if len(true_labels_for_metrics_clean) > 1 and len(
                    np.unique(predicted_labels_for_ari)) > 1:  # Need at least 2 unique labels/clusters
                true_labels_encoded_for_ari = label_encoder_for_ari.fit_transform(true_labels_for_metrics_clean)
                ari_score = adjusted_rand_score(true_labels_encoded_for_ari, predicted_labels_for_ari)
                metrics_data.append({'Metric': 'ARI', 'Value': ari_score})
                print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
            else:
                print("警告: 真实标签或预测簇数量不足，无法计算 ARI。")
                messagebox.showwarning("警告", "真实标签或预测簇数量不足，无法计算 ARI。")
        else:
            print("警告: 无法计算 ARI，因为测试数据中没有 'AA' 列用于真实标签。")
            messagebox.showwarning("警告", "无法计算 ARI，因为测试数据中没有 'AA' 列用于真实标签。")

        # Silhouette Score calculation
        if X_test_complete.shape[0] > 1 and len(
                np.unique(predicted_labels)) > 1:  # Silhouette needs at least 2 samples and 2 clusters
            try:
                silhouette = silhouette_score(X_test_lda, predicted_labels)
                metrics_data.append({'Metric': 'Silhouette', 'Value': silhouette})
                print(f"Silhouette Score: {silhouette:.4f}")
            except Exception as sil_e:
                print(f"警告: 计算 Silhouette Score 时出错: {sil_e}")
                messagebox.showwarning("警告", f"计算 Silhouette Score 时出错: {sil_e}")
        else:
            print("警告: 样本数不足或聚类数少于2，无法计算 Silhouette Score。")
            messagebox.showwarning("警告", "样本数不足或聚类数少于2，无法计算 Silhouette Score。")

        metrics_df = pd.DataFrame(metrics_data)
        if not metrics_df.empty:
            test_cluster_metrics_path = os.path.join(output_dir, "test_cluster_metrics.csv")
            metrics_df.to_csv(test_cluster_metrics_path, index=False)
            print(f"✅ 聚类评价指标已保存到: {test_cluster_metrics_path}")
        else:
            print("未计算任何评价指标，未保存 metrics.csv。")

        messagebox.showinfo("成功",
                            f"预测流程完成！\n\n成功处理了 {retained_sample_count} 个有效样本。\n所有结果已保存至:\n{output_dir}")

    except Exception as e:
        messagebox.showerror("运行时错误", f"处理数据时发生意外错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
