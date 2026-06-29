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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 全局配置
# ==========================================
TEST_SIZE = 0.33
RFECV_SCORING = 'f1_weighted'
RANDOM_STATE = 42
N_ESTIMATORS_FINAL = 1000  # 最终模型树数量

# 【统计学安全标准】
CORRELATION_THRESHOLD = 0.95  # Pearson 相关系数阈值 (去共线性)
COMPACT_N_FEATURES = 10  # Group 1 强制保留特征数 (确保 N/d > 10)

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
# 1. 预处理与特征工程
# ==========================================
def aggregate_replicates(df):
    print(f"  -> [预处理] 聚合平行样 (Mean + Std)...")
    if 'AA' in df.columns:
        df['AA'] = df['AA'].replace({'L-Lle': 'L-Ile', 'Lle': 'L-Ile'})
    non_feature_cols = ['AA', '浓度/uM']
    if 'Date' in df.columns: non_feature_cols.append('Date')
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    grouped = df.groupby(['AA', '浓度/uM'])[feature_cols]
    df_mean = grouped.mean().add_suffix('_mean')
    df_std = grouped.std().fillna(0).add_suffix('_std')
    return pd.concat([df_mean, df_std], axis=1).reset_index()


def generate_ratio_features(df):
    print("  -> [特征工程] 生成 Ratio 特征...")
    cols = [c for c in df.columns if c.endswith('_mean')]
    intensity_cols = [c for c in cols if 'intensity' in c.lower()]
    shift_cols = [c for c in cols if 'shift' in c.lower()]
    epsilon = 1e-6
    new_feats = {}
    for c1, c2 in combinations(intensity_cols, 2):
        n1, n2 = c1.replace('_intensity_mean', '').strip('"'), c2.replace('_intensity_mean', '').strip('"')
        new_feats[f"R_I_{n1}/{n2}"] = df[c1] / (df[c2] + epsilon)
    for c1, c2 in combinations(shift_cols, 2):
        n1, n2 = c1.replace('_shift_mean', '').strip('"'), c2.replace('_shift_mean', '').strip('"')
        new_feats[f"R_S_{n1}/{n2}"] = df[c1] / (df[c2] + epsilon)
    return pd.concat([df, pd.DataFrame(new_feats)], axis=1)


def stratified_group_split(df, target_col='AA', group_col='Group_ID', test_size=0.2):
    print(f"\n[数据划分] Stratified Group Split ({test_size:.0%} Test)...")
    train_indices, test_indices = [], []
    unique_classes = df[target_col].unique()
    for label in unique_classes:
        sub_df = df[df[target_col] == label]
        unique_groups = sub_df[group_col].unique()
        n_groups = len(unique_groups)
        n_test = int(np.ceil(n_groups * test_size))
        if n_groups > 1 and n_groups - n_test < 1: n_test = n_groups - 1
        np.random.seed(RANDOM_STATE)
        shuffled_groups = np.random.permutation(unique_groups)
        test_rows = sub_df[sub_df[group_col].isin(shuffled_groups[:n_test])].index.tolist()
        train_rows = sub_df[sub_df[group_col].isin(shuffled_groups[n_test:])].index.tolist()
        test_indices.extend(test_rows)
        train_indices.extend(train_rows)
    return train_indices, test_indices


# ==========================================
# 2. 核心：紧凑模型训练函数
# ==========================================

def remove_correlated_features(X, feature_names, threshold=0.95):
    """
    预筛选：计算 Pearson 相关系数，移除冗余特征。
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 找出相关系数大于阈值的列
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    keep_indices = [i for i, f in enumerate(feature_names) if f not in to_drop]

    # print(f"     [Pearson] 发现 {len(to_drop)} 个高相关特征 (> {threshold})，予以剔除。")
    return X[:, keep_indices], np.array(feature_names)[keep_indices]


def train_compact_model(X, y, feature_names, task_name="Task", force_compact=False):
    """
    训练函数：
    1. Pearson 去重
    2. 如果 force_compact=True -> 使用 RFE 强制选 10 个特征
    3. 否则 -> 使用 RFECV 自动选
    """
    print(f"\n  >>> 训练 [{task_name}]...")

    # 1. Pearson 预筛选 (减少计算量，去共线性)
    X_clean, feats_clean = remove_correlated_features(X, feature_names, threshold=CORRELATION_THRESHOLD)
    print(f"     [预处理] 原始特征 {X.shape[1]} -> Pearson去重后 {X_clean.shape[1]}")

    rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

    # 2. 特征选择 (分支逻辑)
    if force_compact:
        print(f"     [⚠️ 紧凑模式] 强制执行 RFE (n={COMPACT_N_FEATURES}) 以满足 N/d > 10 安全标准")
        # 使用 RFE (非 CV)，强制指定特征数
        selector = RFE(estimator=rf_base, n_features_to_select=COMPACT_N_FEATURES, step=1)
        selector.fit(X_clean, y)
    else:
        # 正常模式：RFECV
        min_samples = pd.Series(y).value_counts().min()
        curr_cv = 5 if min_samples > 5 else 2
        selector = RFECV(estimator=rf_base, step=1, cv=StratifiedKFold(curr_cv),
                         scoring=RFECV_SCORING, n_jobs=-1)
        selector.fit(X_clean, y)
        print(f"     [标准模式] RFECV 自动选中 {selector.n_features_} 个特征")

    selected_mask = selector.support_
    X_selected = X_clean[:, selected_mask]
    final_feats = feats_clean[selected_mask]

    # 打印最终选中的特征 (如果是紧凑模式，这10个非常重要)
    if force_compact or len(final_feats) < 15:
        print(f"     [最终特征列表]: {list(final_feats)}")

    # 3. 训练最终模型 (暴力加树)
    final_model = RandomForestClassifier(n_estimators=N_ESTIMATORS_FINAL, random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_selected, y)

    # 这里我们需要把 selector 和 feature 映射关系存好
    # 注意：因为我们先做了 Pearson，所以 selector 是针对 feats_clean 的
    return {
        'selector': selector,
        'model': final_model,
        'clean_features': feats_clean,  # 必须保存去重后的特征名列表
        'clean_indices': [i for i, f in enumerate(feature_names) if f in feats_clean],  # 原始索引映射
        'classes': final_model.classes_
    }


def predict_compact(bundle, X_test_raw):
    """
    预测逻辑需要适配两步特征筛选
    """
    # 1. 提取 Pearson 筛选后的列
    clean_indices = bundle['clean_indices']
    X_clean = X_test_raw[:, clean_indices]

    # 2. 提取 RFE/RFECV 筛选后的列
    selector = bundle['selector']
    X_final = X_clean[:, selector.support_]

    # 3. 预测概率 (用于软投票)
    return bundle['model'].predict_proba(X_final)


# ==========================================
# 主流程
# ==========================================
def process_and_train(train_path, test_path, save_dir):
    log_path = os.path.join(save_dir, 'v17_compact_log.txt')
    sys.stdout = AnalysisLogger(log_path)

    print("=" * 60)
    print("DNA-SWCNT 分析 (V17: Compact Model Edition)")
    print(f"统计学约束: Pearson < {CORRELATION_THRESHOLD}, Group 1 Max Features = {COMPACT_N_FEATURES}")
    print("=" * 60)

    try:
        df_train_raw = pd.read_csv(train_path)
        df_test_raw = pd.read_csv(test_path)
    except:
        return

    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
    df_agg = aggregate_replicates(df_all)
    df_feat = generate_ratio_features(df_agg)
    df_feat['Group_ID'] = df_feat['AA'].astype(str) + "_" + df_feat['浓度/uM'].astype(str)

    train_idx, test_idx = stratified_group_split(df_feat, target_col='AA', group_col='Group_ID', test_size=TEST_SIZE)

    feature_names = [c for c in df_feat.columns if c not in ['AA', '浓度/uM', 'Date', 'Group_ID']]
    X = df_feat[feature_names].values
    y = df_feat['AA'].values

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    global_le = LabelEncoder()
    y_train_enc = global_le.fit_transform(y_train)
    n_classes = len(global_le.classes_)

    # 1. 聚类 (V3 Logic)
    print("\n[Step 2] 自然分层...")
    base_mean_indices = [i for i, f in enumerate(feature_names) if f.endswith('_mean') and 'Ratio' not in f]
    unique_classes_str = np.unique(y_train)
    class_centers = [X_train[y_train == cls][:, base_mean_indices].mean(axis=0) for cls in unique_classes_str]

    clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = clustering.fit_predict(class_centers)
    aa_to_superclass = {aa: lbl for aa, lbl in zip(unique_classes_str, labels)}
    groups_dict = {}
    for aa, lbl in aa_to_superclass.items(): groups_dict.setdefault(lbl, []).append(aa)

    for lbl, aas in groups_dict.items():
        print(f"  Group {lbl}: {aas}")

    # 2. Level 1 训练 (Global)
    y_train_l1 = np.array([aa_to_superclass[aa] for aa in y_train])
    # Level 1 任务相对简单，允许自动筛选
    model_l1 = train_compact_model(X_train, y_train_l1, feature_names, "Level 1", force_compact=False)

    # 3. Level 2 训练 (Sub-models)
    models_l2 = {}
    print("\n[Step 3] 训练子模型 (应用紧凑约束)...")

    for lbl, aas in groups_dict.items():
        target_indices = global_le.transform(aas)
        mask = np.isin(y_train_enc, target_indices)
        X_sub, y_sub = X_train[mask], y_train_enc[mask]

        if len(np.unique(y_sub)) <= 1:
            models_l2[lbl] = None
        else:
            # 【关键逻辑】判断是否需要强制紧凑
            # 通常 Group 1 (弱信号组) 包含大量氨基酸，且混淆严重，需要强制降维
            # 这里我们简单粗暴：只要是 Group 1 就强制 Compact
            # (请根据上面的聚类打印确认 Group 1 是否为您想要限制的那个大组)
            is_compact_target = (lbl == 1)

            bundle = train_compact_model(X_sub, y_sub, feature_names, f"Group {lbl}", force_compact=is_compact_target)
            models_l2[lbl] = bundle

    # 4. Soft Voting
    print("\n[Step 4] 软投票预测...")
    # L1 Probs
    l1_probs = predict_compact(model_l1, X_test)

    final_probs = np.zeros((len(X_test), n_classes))

    for lbl, bundle in models_l2.items():
        if bundle is None:
            # 单样本组处理
            target_aa = groups_dict[lbl][0]
            target_idx = global_le.transform([target_aa])[0]
            final_probs[:, target_idx] += l1_probs[:, lbl]
        else:
            # 子模型预测
            sub_probs = predict_compact(bundle, X_test)
            weight = l1_probs[:, lbl].reshape(-1, 1)

            for loc_i, glob_i in enumerate(bundle['classes']):
                final_probs[:, glob_i] += (sub_probs[:, loc_i] * weight[:, 0])

    final_pred_indices = np.argmax(final_probs, axis=1)
    y_pred_hier = global_le.inverse_transform(final_pred_indices)

    # 5. 评估
    print("\n" + "=" * 60)
    print("【紧凑模型评估报告】")
    acc = accuracy_score(y_test, y_pred_hier)
    print(f"Overall Accuracy: {acc:.4f}")
    print("=" * 60)
    print(classification_report(y_test, y_pred_hier))

    # 重点检查 Ile 和 Leu
    print("\n[关键检查] 易混淆氨基酸性能:")
    report = classification_report(y_test, y_pred_hier, output_dict=True)
    for aa in ['L-Ile', 'L-Leu', 'L-Val', 'L-Phe']:
        if aa in report:
            f1 = report[aa]['f1-score']
            status = "PASS" if f1 >= 0.9 else "WARNING"
            print(f"  {aa}: F1 = {f1:.4f} [{status}]")

    # 混淆矩阵
    plt.figure(figsize=(12, 10))
    labels = np.unique(np.concatenate([y_test, y_pred_hier]))
    cm = confusion_matrix(y_test, y_pred_hier, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"V17 Compact Model (Acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_v17.png'))

    sys.stdout = sys.stdout.terminal
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("完成", f"V17 运行结束！\nAcc: {acc:.4f}")
    root.destroy()
    sys.exit(0)


if __name__ == "__main__":
    tf, ttf, sd = select_files_gui()
    if tf: process_and_train(tf, ttf, sd)