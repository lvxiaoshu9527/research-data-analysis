import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# 机器学习库
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, \
    recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV

# 尝试导入贝叶斯优化库
try:
    from skopt import BayesSearchCV

    HAS_BAYES = True
except ImportError:
    HAS_BAYES = False

# ==========================================
# 全局配置
# ==========================================
TEST_SIZE = 0.33
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = 'f1_weighted'

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class DualLogger:
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

    file_types = [("Data Files", "*.csv *.xlsx *.xls")]
    file_paths = filedialog.askopenfilenames(title="请选择数据文件 (CSV/Excel)", filetypes=file_types)
    if not file_paths: return None, None, None

    save_dir = filedialog.askdirectory(title="选择结果保存目录")
    if not save_dir: return None, None, None

    do_selection = messagebox.askyesno("配置", "是否执行 RFECV 特征筛选？")

    root.destroy()
    return list(file_paths), save_dir, do_selection


# ==========================================
# 1. 数据预处理 (带详细 Debug 信息)
# ==========================================
def load_and_merge_data(file_paths):
    dfs = []
    print("\n[Debug] 开始读取文件...")
    for f in file_paths:
        try:
            if f.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(f)
            else:
                # 尝试多种编码读取 CSV
                try:
                    df = pd.read_csv(f, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(f, encoding='gbk')

            print(f"  -> 读取 {os.path.basename(f)}: {len(df)} 行")
            # 检查 HCl 数量
            if 'AA' in df.columns:
                hcl_count = df[df['AA'] == 'HCl'].shape[0]
                print(f"     (包含 HCl 样本: {hcl_count} 个)")

            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs: return None
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"[Debug] 合并后总行数: {len(df_full)}")
    print(f"        总 HCl 样本数: {df_full[df_full['AA'] == 'HCl'].shape[0]}")
    return df_full


def aggregate_replicates(df):
    print(f"\n[Debug] 执行聚合 (Mean+Std)...")
    if 'AA' in df.columns:
        df['AA'] = df['AA'].replace({'L-Lle': 'L-Ile', 'Lle': 'L-Ile'})

    non_feature_cols = ['AA', '浓度/uM', 'Date']
    numeric_cols = [c for c in df.columns if c not in non_feature_cols and df[c].dtype in [np.float64, np.int64]]

    grouped = df.groupby(['AA', '浓度/uM'])[numeric_cols]

    df_mean = grouped.mean().add_suffix('_mean')
    df_std = grouped.std().fillna(0).add_suffix('_std')

    df_agg = pd.concat([df_mean, df_std], axis=1).reset_index()

    print(f"[Debug] 聚合后总样本数 (浓度点数): {len(df_agg)}")
    hcl_agg_count = df_agg[df_agg['AA'] == 'HCl'].shape[0]
    print(f"        聚合后 HCl 样本数: {hcl_agg_count} (预期应为浓度梯度数量)")

    return df_agg


def stratified_split_aggregated(df, test_size=0.33):
    print(f"\n[Debug] 执行 Stratified Split ({test_size:.0%} Test)...")
    X_indices = np.arange(len(df))
    y = df['AA'].values

    # 打印类别分布
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"  -> 聚合后各类别样本数: {dist}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X_indices, y))

    # 检查测试集里的 HCl
    y_test = y[test_idx]
    hcl_test_count = np.sum(y_test == 'HCl')
    print(f"  -> 测试集总数: {len(test_idx)}")
    print(f"  -> 测试集中 HCl 数量: {hcl_test_count}")

    return train_idx, test_idx


# ==========================================
# 2. 模型相关
# ==========================================
def get_model_config():
    models = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_jobs=-1)
    }

    if HAS_BAYES:
        params = {
            'RandomForest': {'n_estimators': (100, 1000), 'max_depth': (3, 25)},
            'SVM': {'clf__C': (0.1, 100.0, 'log-uniform'), 'clf__gamma': (1e-4, 1.0, 'log-uniform')},
            'XGBoost': {'n_estimators': (100, 500), 'learning_rate': (0.01, 0.3, 'log-uniform')},
            'KNN': {'clf__n_neighbors': (3, 15)}
        }
    else:
        params = {
            'RandomForest': {'n_estimators': [100, 500]},
            'SVM': {'clf__C': [1, 10]},
            'XGBoost': {'n_estimators': [100, 300]},
            'KNN': {'clf__n_neighbors': [5, 7]}
        }
    return models, params


def optimize_and_train(model_name, model, param_space, X_train, y_train):
    print(f"     正在优化 {model_name}...", end="")
    if model_name in ['SVM', 'KNN']:
        estimator = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    else:
        estimator = model

    start = time()
    if HAS_BAYES:
        opt = BayesSearchCV(estimator, param_space, n_iter=10, cv=3, scoring=SCORING, random_state=RANDOM_STATE,
                            n_jobs=-1, verbose=0)
    else:
        opt = RandomizedSearchCV(estimator, param_space, n_iter=10, cv=3, scoring=SCORING, random_state=RANDOM_STATE,
                                 n_jobs=-1, verbose=0)

    try:
        opt.fit(X_train, y_train)
        print(f" 完成 ({time() - start:.1f}s) | CV: {opt.best_score_:.4f}")
        return opt.best_estimator_
    except Exception as e:
        print(f" 失败: {e}")
        return None


def perform_feature_selection(X, y, feature_names):
    print("\n  >>> [RFECV] 特征筛选...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    min_samples = pd.Series(y).value_counts().min()
    curr_cv = 5 if min_samples > 5 else 2

    selector = RFECV(estimator=rf, step=1, cv=StratifiedKFold(curr_cv), scoring=SCORING, n_jobs=-1)
    selector.fit(X, y)
    print(f"     选中: {np.sum(selector.support_)}/{len(feature_names)}")
    return selector.support_


def plot_confusion_matrix_percent(y_true, y_pred, classes, model_name, save_dir):
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_percent = cm.astype('float') / row_sums * 100

    sns.heatmap(cm_percent, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"{model_name} Confusion Matrix (%)")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_cm_percent.png'), dpi=300)
    plt.close()


# ==========================================
# 主流程
# ==========================================
def process_and_train(file_paths, save_dir, do_selection):
    log_path = os.path.join(save_dir, 'debug_report.txt')
    sys.stdout = DualLogger(log_path)

    print("=" * 60)
    print("DNA-SWCNT 分析 (Debug Edition)")
    print(
        f"配置: Test Size={TEST_SIZE}, 贝叶斯={'ON' if HAS_BAYES else 'OFF'}, RFECV={'ON' if do_selection else 'OFF'}")
    print("=" * 60)

    # 1. 加载
    df_all = load_and_merge_data(file_paths)
    if df_all is None: return

    # 2. 聚合 (Mean+Std)
    df_agg = aggregate_replicates(df_all)

    feature_names = [c for c in df_agg.columns if c not in ['AA', '浓度/uM', 'Date', 'Sample_ID']]
    X = df_agg[feature_names].values
    y = df_agg['AA'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print(f"[Debug] 类别列表 ({len(classes)}): {classes}")

    # 3. 划分
    train_idx, test_idx = stratified_split_aggregated(df_agg, test_size=TEST_SIZE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    # 4. 训练
    models, params = get_model_config()
    feature_sets = {'Mean+Std (Full)': np.ones(X.shape[1], dtype=bool)}

    if do_selection:
        mask_selected = perform_feature_selection(X_train, y_train, feature_names)
        feature_sets['Mean+Std (RFECV)'] = mask_selected
        pd.DataFrame({'Feature': np.array(feature_names)[mask_selected]}).to_csv(
            os.path.join(save_dir, 'rfecv_features.csv'), index=False)

    print("\n" + "=" * 40)
    print("开始多模型对比训练")
    print("=" * 40)

    results = []
    for model_name, model_inst in models.items():
        for f_set_name, mask in feature_sets.items():
            print(f"\n>> 处理: {model_name} [{f_set_name}]")

            X_tr_sub = X_train[:, mask]
            X_te_sub = X_test[:, mask]

            best_model = optimize_and_train(model_name, model_inst, params[model_name], X_tr_sub, y_train)

            if best_model:
                y_pred = best_model.predict(X_te_sub)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                print(f"   -> Acc: {acc:.4f}, F1: {f1:.4f}")

                results.append({
                    'Model': model_name,
                    'Feature_Set': f_set_name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1_Score': f1
                })

                # 画图
                safe_name = f"{model_name}_{f_set_name}".replace(' ', '_')
                plot_confusion_matrix_percent(y_test, y_pred, classes, f"{model_name} ({f_set_name})", save_dir)

    # 导出结果
    df_res = pd.DataFrame(results)
    res_path = os.path.join(save_dir, 'final_comparison.csv')
    df_res.to_csv(res_path, index=False)

    print("\n" + "=" * 60)
    print("【最终对比摘要】")
    if not df_res.empty:
        print(
            df_res[['Model', 'Feature_Set', 'Accuracy', 'F1_Score']].sort_values('Accuracy', ascending=False).to_string(
                index=False))
    print("=" * 60)

    sys.stdout = sys.stdout.terminal
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("完成", f"分析结束！\n日志已生成: debug_report.txt")
    root.destroy()
    sys.exit(0)


if __name__ == "__main__":
    res = select_files_gui()
    if res[0]: process_and_train(*res)