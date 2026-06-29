import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox


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


# --- 可视化函数 ---

def plot_prediction_results(
        X_train_lda: np.ndarray,
        y_train: pd.Series,
        X_test_lda: np.ndarray,
        predicted_labels: np.ndarray,
        output_filename: str
):
    """
    可视化训练数据和测试数据的预测结果。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))

    # 绘制训练数据散点图
    plt.scatter(
        X_train_lda[:, 0],
        X_train_lda[:, 1],
        c=y_train,
        cmap='tab20',
        alpha=0.35,
        edgecolor='none',
        label='Train Data'
    )

    # 绘制测试数据散点图
    if X_test_lda.shape[0] > 0:
        plt.scatter(
            X_test_lda[:, 0],
            X_test_lda[:, 1],
            c=predicted_labels,
            cmap='tab20',
            marker='x',
            s=120,
            linewidth=1.5,
            edgecolor='black',
            label='Predicted Test Data (Complete Features)'
        )

    plt.title("LDA + KMeans: Test Data Prediction vs. Training Data", fontsize=16, pad=20)
    plt.xlabel("LDA Component 1", fontsize=12)
    plt.ylabel("LDA Component 2", fontsize=12)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)
    plt.savefig(output_filename.replace(".png", ".svg"), format='svg')
    plt.close()

    print(f"✅ 预测结果可视化图像已保存到: {output_filename}")


# --- 主执行逻辑 ---

def main():
    """
    主函数，执行整个预测流程。
    """
    print("--- LDA+KMeans 模型预测工具 (可处理不完整特征) ---")

    # 1. 引导用户选择路径
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

    try:
        # 2. 加载模型和必要文件
        print(f"\n🔄 正在从 '{model_dir}' 加载模型和数据...")

        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        lda_model = joblib.load(os.path.join(model_dir, "lda_model.joblib"))
        kmeans_model = joblib.load(os.path.join(model_dir, "kmeans_model.joblib"))

        X_train_lda_for_plot = np.load(os.path.join(model_dir, "X_train_lda_for_plot.npy"))
        y_train_for_plot = pd.read_csv(os.path.join(model_dir, "y_train_for_plot.csv")).iloc[:, 0]
        # 加载训练时使用的最终特征列表
        expected_features = pd.read_csv(os.path.join(model_dir, "feature_columns.csv"))['Feature'].tolist()

        print("✅ 模型和数据加载成功。")

    except FileNotFoundError as e:
        messagebox.showerror("文件未找到", f"加载模型文件时出错: {e}")
        return
    except Exception as e:
        messagebox.showerror("加载错误", f"加载文件时发生未知错误: {e}")
        return

    try:
        # 3. 加载并处理测试数据
        print(f"\n🔄 正在加载测试数据: {test_data_path}")
        df_test = pd.read_csv(test_data_path)

        # 提取样本ID列
        if 'Sample_ID' in df_test.columns:
            sample_ids = df_test['Sample_ID']
        else:
            sample_ids = pd.Series([f'TestSample_{i + 1}' for i in range(len(df_test))], name='Sample_ID')
            df_test['Sample_ID'] = sample_ids
            print("未找到 'Sample_ID' 列，已自动创建。")

        X_test = df_test.set_index('Sample_ID')

        # --- 新增：特征对齐与样本筛选逻辑 ---
        print("\n🔬 正在对齐特征并筛选完整样本...")

        # 补全测试数据中缺失的特征列，并填充为NaN
        for col in expected_features:
            if col not in X_test.columns:
                X_test[col] = np.nan

        # 按照训练时的顺序对齐特征列
        X_test_aligned = X_test[expected_features]

        # 筛选出所有特征列都不是NaN的行（即完整的样本）
        original_sample_count = len(X_test_aligned)
        X_test_complete = X_test_aligned.dropna(axis=0, how='any')
        retained_sample_count = len(X_test_complete)

        print(f"原始样本数: {original_sample_count}")
        print(f"特征齐全的样本数: {retained_sample_count}")
        print(f"因特征缺失而被丢弃的样本数: {original_sample_count - retained_sample_count}")

        if retained_sample_count == 0:
            messagebox.showwarning("无可用数据", "所有测试样本都因缺少必要的特征而被丢弃，无法进行预测。")
            return

        # 获取被保留样本的ID
        sample_ids_complete = X_test_complete.index

        # 4. 对筛选后的完整数据执行预测
        print("\n🤖 正在对完整样本执行标准化、降维和预测...")

        # 标准化 (使用已加载的scaler)
        X_test_scaled = scaler.transform(X_test_complete)

        # LDA 降维
        X_test_lda = lda_model.transform(X_test_scaled)

        # KMeans 预测
        predicted_labels = kmeans_model.predict(X_test_lda)
        print("✅ 预测完成！")

        # 5. 保存结果
        print("\n💾 正在保存预测结果...")
        results_df = pd.DataFrame({
            'Sample_ID': sample_ids_complete,
            'Predicted_Cluster': predicted_labels
        })
        for i in range(X_test_lda.shape[1]):
            results_df[f'LDA_{i + 1}'] = X_test_lda[:, i]

        output_csv_path = os.path.join(output_dir, "test_data_predictions.csv")
        results_df.to_csv(output_csv_path, index=False)
        print(f"✅ 预测结果已保存到: {output_csv_path}")

        # 6. 生成可视化图表
        output_plot_path = os.path.join(output_dir, "test_prediction_visualization.png")
        plot_prediction_results(
            X_train_lda_for_plot,
            y_train_for_plot,
            X_test_lda,
            predicted_labels,
            output_plot_path
        )

        messagebox.showinfo("成功",
                            f"预测流程完成！\n\n处理了 {retained_sample_count} 个有效样本。\n结果已保存至:\n{output_dir}")

    except ValueError as e:
        messagebox.showerror("数据错误", str(e))
    except Exception as e:
        messagebox.showerror("运行时错误", f"处理数据时发生错误: {e}")


if __name__ == "__main__":
    main()
