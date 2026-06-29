import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter

# 机器学习库
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 全局配置
# ==========================================
TEST_SIZE = 0.33
RFECV_SCORING = 'f1_weighted'
RANDOM_STATE = 42
N_ESTIMATORS = 1000  # 树的数量 (越多越稳)
RFECV_CV_FOLDS = 10  # 特征筛选的折数 (越雷越准)

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
# 1. 核心预处理：聚合降噪 (Mean + Std)
# ==========================================
def aggregate_replicates(df):
    """
    将平行样压缩为 1 个高信噪比样本。
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

    df_agg = pd.concat([df_mean, df_std], axis=1).reset_index()
    print(f"     聚合完成！样本数: {len(df_agg)}")
    return df_agg


# ==========================================
# 2. 特征工程：生成比值 (Ratio Features)
# ==========================================
def generate_ratio_features(df):
    print("  -> [特征工程] 生成 Ratio 特征...")
    cols = [c for c in df.columns if c.endswith('_mean')]
    intensity_cols = [c for c in cols if 'intensity' in c.lower()]
    shift_cols = [c for c in cols if 'shift' in c.lower()]

    epsilon = 1e-6
    new_feats = {}

    # 强度比值
    for c1, c2 in combinations(intensity_cols, 2):
        n1 = c1.replace('_intensity_mean', '').strip('"')
        n2 = c2.replace('_intensity_mean', '').strip('"')
        new_feats[f"R_I_{n1}/{n2}"] = df[c1] / (df[c2] + epsilon)

    # 位移比值
    for c1, c2 in combinations(shift_cols, 2):
        n1 = c1.replace('_shift_mean', '').strip('"')
        n2 = c2.replace('_shift_mean', '').strip('"')
        new_feats[f"R_S_{n1}/{n2}"] = df[c1] / (df[c2] + epsilon)

    df_new = pd.concat([df, pd.DataFrame(new_feats)], axis=1)
    return df_new


# ==========================================
# 3. 数据划分：基于聚合后的唯一浓度点
# ==========================================
def stratified_split_aggregated(df, test_size=0.2):
    """
    因为 df 已经是聚合后的 (每个浓度只有一行)，直接用 StratifiedShuffleSplit 即可。
    这等同于对原始数据的 GroupShuffleSplit。
    """
    print(f"\n[数据划分] 执行 Stratified Split ({test_size:.0%} Test)...")
    from sklearn.model_selection import StratifiedShuffleSplit

    # 创建唯一ID用于追踪
    df['Sample_ID'] = df['AA'].astype(str) + "_" + df['浓度/uM'].astype(str)

    X_indices = np.arange(len(df))
    y = df['AA'].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X_indices, y))

    return train_idx, test_idx


# ==========================================
# 4. 训练核心：Global RFECV
# ==========================================
def train_global_model(X, y, feature_names):
    print(f"\n  >>> 开始训练 Global Model (RFECV + RF)...")

    # --- 阶段1: 特征筛选 ---
    print(f"     正在进行特征筛选 (CV={RFECV_CV_FOLDS}, Scoring={RFECV_SCORING})...")
    # 使用较少的树进行筛选以节省时间，但要足够稳定
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

    # 打印 Top 10 重要特征
    if len(selected_feats) > 0:
        imps = rfecv.estimator_.feature_importances_
        # 排序
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
    log_path = os.path.join(save_dir, 'v16_global_log.txt')
    sys.stdout = AnalysisLogger(log_path)

    print("=" * 60)
    print("DNA-SWCNT 分析 (V16: Global Direct)")
    print("策略: 聚合降噪 -> Ratio特征 -> 全局 RFECV 筛选 -> 全局 RF")
    print("=" * 60)

    try:
        df_train_raw = pd.read_csv(train_path)
        df_test_raw = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    # 1. 聚合
    df_agg = aggregate_replicates(df_all)

    # 2. 特征
    df_feat = generate_ratio_features(df_agg)

    feature_names = [c for c in df_feat.columns if c not in ['AA', '浓度/uM', 'Date', 'Sample_ID']]
    X = df_feat[feature_names].values
    y = df_feat['AA'].values

    # 3. 划分
    train_idx, test_idx = stratified_split_aggregated(df_feat, test_size=TEST_SIZE)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  -> 训练集大小: {len(X_train)}")
    print(f"  -> 测试集大小: {len(X_test)}")

    # 4. 训练全局模型
    selector, model = train_global_model(X_train, y_train, feature_names)

    # 5. 预测
    print("\n[最终评估]...")
    # 先筛选特征
    X_test_sel = X_test[:, selector.support_]
    # 再预测
    y_pred = model.predict(X_test_sel)

    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"【V16 Global Direct 结果】")
    print(f"Accuracy: {acc:.4f}")
    print("=" * 60)
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    plt.figure(figsize=(12, 10))
    labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"V16 Global Model (Acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_v16.png'))

    # 保存特征重要性
    imp_df = pd.DataFrame({
        'Feature': np.array(feature_names)[selector.support_],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    imp_df.to_csv(os.path.join(save_dir, 'global_feature_importance.csv'), index=False)
    print(f"特征重要性已保存至 global_feature_importance.csv")

    sys.stdout = sys.stdout.terminal
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("完成", f"V16 Global 分析结束！\nAccuracy: {acc:.4f}")
    root.destroy()
    sys.exit(0)


if __name__ == "__main__":
    tf, ttf, sd = select_files_gui()
    if tf:
        process_and_train(tf, ttf, sd)