import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification


def generate_dummy_models_and_data_v2():
    """
    生成一套虚拟模型和数据。
    生成的测试数据会故意缺少一些特征列，以测试预测脚本的稳健性。
    """
    print("--- 正在生成虚拟模型和数据文件 (V2) ---")

    # 定义参数
    N_SAMPLES = 200
    N_FEATURES = 15  # 增加特征数量以更好地模拟
    N_CLASSES = 5
    MODEL_DIR = "dummy_model_files_v2"

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 生成虚拟数据并训练模型
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_FEATURES,
        n_redundant=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=42
    )
    feature_columns = [f'Feature_{i + 1}' for i in range(N_FEATURES)]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    lda = LDA(n_components=N_CLASSES - 1).fit(X_scaled, y)
    X_lda = lda.transform(X_scaled)

    kmeans = KMeans(n_clusters=N_CLASSES, random_state=42, n_init=10).fit(X_lda)

    print("✅ 模型训练完成。")

    # 2. 保存所有必要文件
    print(f"💾 正在将文件保存到 '{MODEL_DIR}' 文件夹...")
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(lda, os.path.join(MODEL_DIR, "lda_model.joblib"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.joblib"))
    np.save(os.path.join(MODEL_DIR, "X_train_lda_for_plot.npy"), X_lda)
    pd.DataFrame(y).to_csv(os.path.join(MODEL_DIR, "y_train_for_plot.csv"), index=False)
    pd.DataFrame({'Feature': feature_columns}).to_csv(os.path.join(MODEL_DIR, "feature_columns.csv"), index=False)

    # 3. 创建不完整的虚拟测试数据
    print("\n🔧 正在创建不完整的测试数据文件...")
    X_test, _ = make_classification(n_samples=50, n_features=N_FEATURES, n_informative=N_FEATURES, n_redundant=0,
                                    n_classes=N_CLASSES, random_state=101)
    df_test = pd.DataFrame(X_test, columns=feature_columns)

    # 故意删除几列特征
    features_to_drop = ['Feature_2', 'Feature_5', 'Feature_10']
    df_test.drop(columns=features_to_drop, inplace=True)
    print(f"从测试数据中移除了以下特征列: {features_to_drop}")

    # 插入ID列
    df_test.insert(0, 'Sample_ID', [f'NewSample_{i + 1}' for i in range(50)])

    test_file_path = "dummy_test_data_incomplete.csv"
    df_test.to_csv(test_file_path, index=False)

    print("\n🎉 全部完成！")
    print(f"  - 模型和辅助文件已保存在 '{MODEL_DIR}' 文件夹中。")
    print(f"  - 一个不完整的虚拟测试文件 '{test_file_path}' 已在当前目录创建。")
    print("\n现在您可以运行更新后的预测脚本了。")


if __name__ == "__main__":
    generate_dummy_models_and_data_v2()
