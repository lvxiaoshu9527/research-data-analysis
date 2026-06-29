import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 全局配置
# ==========================================
TEST_SIZE = 0.33
RFECV_SCORING = 'f1_weighted'
RANDOM_STATE = 42
N_ESTIMATORS = 1000  # 树的数量
RFECV_CV_FOLDS = 10  # 特征筛选折数

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class AnalysisLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def select_files_gui():
    root = tk.Tk()
    root.withdraw()
    print(">>> 等待用户选择文件...")
    train_path = filedialog.askopenfilename(title="选择训练集 (train_data.csv)", filetypes=[("CSV", "*.csv")])
    if not train_path: return None, None, None
    test_path = filedialog.askopenfilename(title="选择测试集 (test_data.csv)", filetypes=[("CSV", "*.csv")])
    if not test_path: return None, None, None
    save_dir = filedialog.askdirectory(title="选择结果保存目录")
    root.destroy()
    return train_path, test_path, save_dir


# ==========================================
# 1. 核心预处理：聚合 (Mean + Std)
# ==========================================
def aggregate_replicates(df):
    """
    将平行样压缩为 1 个高信噪比样本。
    保留 Mean (强度) 和 Std (稳定性)。
    """
    print(f"  -> [预处理] 正在聚合平行样 (原始行数: {len(df)})...")
    if 'AA' in df.columns:
        df['AA'] = df['AA'].replace({'L-Lle': 'L-Ile', 'Lle': 'L-Ile'})

    non_feature_cols = ['AA', '浓度/uM']
    if 'Date' in df.columns: non_feature_cols.append('Date')
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # GroupBy Mean & Std
    grouped = df.groupby(['AA', '浓度/uM'])[feature_cols]
    df_mean = grouped.mean().add_suffix('_mean')
    df_std = grouped.std().fillna(0).add_suffix('_std')

    # 直接合并，这就是我们所有的特征
    df_agg = pd.concat([df_mean, df_std], axis=1).reset_index()
    print(f"     聚合完成！样本数: {len(df_agg)}")
    print(f"     特征源: 仅使用 Mean + Std (不生成 Ratio)")
    return df_agg


# ==========================================
# 2. 数据划分
# ==========================================
def stratified_split_aggregated(df, test_size=0.2):
    print(f"\n[数据划分] 执行 Stratified Split ({test_size:.0%} Test)...")

    X_indices = np.arange(len(df))
    y = df['AA'].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X_indices, y))

    return train_idx, test_idx


# ==========================================
# 3. 全局筛选与训练
# ==========================================
def train_global_model(X, y, feature_names):
    print(f"\n  >>> 开始训练 Global Model (RFECV + RF)...")

    # --- 阶段1: 特征筛选 ---
    print(f"     正在进行特征筛选 (CV={RFECV_CV_FOLDS}, Scoring={RFECV_SCORING})...")

    rf_selector = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    # 动态调整 CV
    min_samples = pd.Series(y).value_counts().min()
    curr_cv = min(RFECV_CV_FOLDS, min_samples) if min_samples > 1 else 2

    rfecv = RFECV(estimator=rf_selector, step=1, cv=StratifiedKFold(curr_cv),
                  scoring=RFECV_SCORING, n_jobs=-1)
    rfecv.fit(X, y)

    selected_mask = rfecv.support_
    X_selected = X[:, selected_mask]
    selected_feats = np.array(feature_names)[selected_mask]

    print(f"     [筛选结果] 从 {len(feature_names)} 个特征中选中了 {len(selected_feats)} 个")

    # 打印 Top 重要特征
    if len(selected_feats) > 0:
        imps = rfecv.estimator_.feature_importances_
        indices = np.argsort(imps)[::-1]
        top_n = min(10, len(selected_feats))
        print(f"     [Top {top_n} 关键特征]:")
        for i in range(top_n):
            print(f"       {i + 1}. {selected_feats[indices[i]]} ({imps[indices[i]]:.4f})")

    # --- 阶段2: 最终模型训练 ---
    print(f"     正在训练最终模型 (n_estimators={N_ESTIMATORS})...")
    final_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_selected, y)

    return rfecv, final_model


# ==========================================
# 主流程
# ==========================================
def process_and_train(train_path, test_path, save_dir):
    log_path = os.path.join(save_dir, 'v18_global_native_log.txt')
    sys.stdout = AnalysisLogger(log_path)

    print("=" * 60)
    print("DNA-SWCNT 分析 (V18: Global Native)")
    print("策略: 聚合降噪 (Mean+Std) -> 全局 RFECV 筛选 -> 全局 RF")
    print("=" * 60)

    try:
        df_train_raw = pd.read_csv(train_path)
        df_test_raw = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    # 1. 聚合 (Mean + Std)
    df_feat = aggregate_replicates(df_all)

    # 不再调用 generate_ratio_features

    feature_names = [c for c in df_feat.columns if c not in ['AA', '浓度/uM', 'Date']]
    X = df_feat[feature_names].values
    y = df_feat['AA'].values

    # 2. 划分
    train_idx, test_idx = stratified_split_aggregated(df_feat, test_size=TEST_SIZE)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  -> 训练集大小: {len(X_train)}")
    print(f"  -> 测试集大小: {len(X_test)}")

    # 3. 训练全局模型
    selector, model = train_global_model(X_train, y_train, feature_names)

    # 4. 预测
    print("\n[最终评估]...")
    X_test_sel = X_test[:, selector.support_]
    y_pred = model.predict(X_test_sel)

    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"【V18 Global Native 结果】")
    print(f"Accuracy: {acc:.4f}")
    print("=" * 60)
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    plt.figure(figsize=(12, 10))
    labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"V18 Global Native (Mean+Std) (Acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_v18.png'))

    # 保存特征重要性
    imp_df = pd.DataFrame({
        'Feature': np.array(feature_names)[selector.support_],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    imp_df.to_csv(os.path.join(save_dir, 'global_native_importance.csv'), index=False)
    print(f"特征重要性已保存至 global_native_importance.csv")

    sys.stdout = sys.stdout.terminal
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("完成", f"V18 运行结束！\nAccuracy: {acc:.4f}")
    root.destroy()
    sys.exit(0)


if __name__ == "__main__":
    tf, ttf, sd = select_files_gui()
    if tf:
        process_and_train(tf, ttf, sd)