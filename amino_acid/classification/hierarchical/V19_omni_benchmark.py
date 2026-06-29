import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from time import time

# 机器学习基础库
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# 模型库
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV

# 尝试导入贝叶斯优化库，如果没有则降级
try:
    from skopt import BayesSearchCV

    HAS_BAYES = True
except ImportError:
    HAS_BAYES = False

# ==========================================
# 全局配置 (User Requirements)
# ==========================================
TEST_SIZE = 0.33  # 【修改点】测试集占比改为 0.33
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
    # 支持多选，方便一次性导入 train 和 test
    file_paths = filedialog.askopenfilenames(title="请选择所有数据文件 (train.csv 和 test.csv)",
                                             filetypes=[("CSV", "*.csv")])
    if not file_paths: return None, None, None

    save_dir = filedialog.askdirectory(title="选择结果保存目录")
    if not save_dir: return None, None, None

    # 询问是否进行特征筛选
    do_selection = messagebox.askyesno("配置",
                                       "是否执行 RFECV 特征筛选？\n\nYes: 对比 [Mean+Std全集] vs [RFECV筛选集]\nNo: 仅运行 [Mean+Std全集]")

    root.destroy()
    return list(file_paths), save_dir, do_selection


# ==========================================
# 1. 数据预处理 (仅 Mean + Std)
# ==========================================
def load_and_merge_data(file_paths):
    dfs = []
    for f in file_paths:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if not dfs: return None
    return pd.concat(dfs, ignore_index=True)


def aggregate_replicates(df):
    print(f"  -> [预处理] 聚合平行样 (Mean + Std)...")
    if 'AA' in df.columns:
        df['AA'] = df['AA'].replace({'L-Lle': 'L-Ile', 'Lle': 'L-Ile'})

    non_feature_cols = ['AA', '浓度/uM', 'Date']
    # 仅选择数值类型的列进行聚合
    numeric_cols = [c for c in df.columns if c not in non_feature_cols and df[c].dtype in [np.float64, np.int64]]

    grouped = df.groupby(['AA', '浓度/uM'])[numeric_cols]

    df_mean = grouped.mean().add_suffix('_mean')
    df_std = grouped.std().fillna(0).add_suffix('_std')

    # 【修改点】不再生成 Ratio，直接返回 Mean + Std
    return pd.concat([df_mean, df_std], axis=1).reset_index()


def stratified_split_aggregated(df, test_size=0.33):
    print(f"\n[数据划分] Stratified Split ({test_size:.0%} Test)...")
    X_indices = np.arange(len(df))
    y = df['AA'].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    return next(sss.split(X_indices, y))


# ==========================================
# 2. 模型定义与优化器
# ==========================================
def get_model_config():
    """定义模型及其参数搜索空间"""
    models = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_jobs=-1)
    }

    if HAS_BAYES:
        # scikit-optimize 空间
        params = {
            'RandomForest': {
                'n_estimators': (100, 1000),
                'max_depth': (3, 25),
                'min_samples_split': (2, 10)
            },
            'SVM': {
                'clf__C': (0.1, 100.0, 'log-uniform'),
                'clf__gamma': (1e-4, 1.0, 'log-uniform'),
                'clf__kernel': ['rbf']
            },
            'XGBoost': {
                'n_estimators': (100, 500),
                'learning_rate': (0.01, 0.3, 'log-uniform'),
                'max_depth': (3, 10)
            },
            'KNN': {
                'clf__n_neighbors': (3, 15),
                'clf__weights': ['uniform', 'distance']
            }
        }
    else:
        # RandomizedSearchCV 空间
        params = {
            'RandomForest': {
                'n_estimators': [100, 200, 500, 800],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'SVM': {
                'clf__C': [0.1, 1, 10, 100],
                'clf__gamma': ['scale', 0.1, 0.01],
                'clf__kernel': ['rbf']
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'KNN': {
                'clf__n_neighbors': [3, 5, 7, 9, 11],
                'clf__weights': ['uniform', 'distance']
            }
        }
    return models, params


def optimize_and_train(model_name, model, param_space, X_train, y_train):
    """执行超参数优化"""
    print(f"     正在优化 {model_name}...", end="")

    # SVM 和 KNN 需要标准化 pipeline
    if model_name in ['SVM', 'KNN']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        estimator = pipeline
    else:
        # RF 和 XGBoost 不需要标准化
        estimator = model

    start = time()

    if HAS_BAYES:
        opt = BayesSearchCV(
            estimator, param_space, n_iter=15, cv=StratifiedKFold(3),
            scoring=SCORING, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
        )
    else:
        opt = RandomizedSearchCV(
            estimator, param_space, n_iter=15, cv=StratifiedKFold(3),
            scoring=SCORING, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
        )

    try:
        opt.fit(X_train, y_train)
        print(f" 完成 ({time() - start:.1f}s) | CV最佳分: {opt.best_score_:.4f}")
        return opt.best_estimator_
    except Exception as e:
        print(f" 失败: {e}")
        return None


# ==========================================
# 3. 特征筛选 (RFECV)
# ==========================================
def perform_feature_selection(X, y, feature_names):
    print("\n  >>> 执行 RFECV 特征筛选 (Mean+Std)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

    min_samples = pd.Series(y).value_counts().min()
    curr_cv = 5 if min_samples > 5 else 2

    rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(curr_cv), scoring=SCORING, n_jobs=-1)
    rfecv.fit(X, y)

    selected_mask = rfecv.support_
    print(f"     选中特征数: {np.sum(selected_mask)}/{len(feature_names)}")
    return selected_mask


# ==========================================
# 主流程
# ==========================================
def process_and_train(file_paths, save_dir, do_selection):
    log_path = os.path.join(save_dir, 'process_log_mean_std.txt')
    sys.stdout = DualLogger(log_path)

    print("=" * 60)
    print("DNA-SWCNT 分析 (V19 Modified: Mean+Std Only)")
    print(
        f"配置: Test Size={TEST_SIZE}, 贝叶斯优化={'ON' if HAS_BAYES else 'OFF'}, RFECV={'ON' if do_selection else 'OFF'}")
    print("=" * 60)

    # 1. 加载与合并
    df_all = load_and_merge_data(file_paths)
    if df_all is None: return

    # 2. 聚合 (Mean + Std) - 不生成 Ratio
    df_agg = aggregate_replicates(df_all)

    # 剔除非特征列
    feature_names = [c for c in df_agg.columns if c not in ['AA', '浓度/uM', 'Date', 'Sample_ID']]
    X = df_agg[feature_names].values
    y = df_agg['AA'].values

    # 标签编码
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    # 3. 划分 (0.33)
    train_idx, test_idx = stratified_split_aggregated(df_agg, test_size=TEST_SIZE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    print(f"  -> Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  -> 特征总数: {len(feature_names)} (Mean + Std)")

    # 4. 准备实验队列
    models, params = get_model_config()
    results = []

    # 特征集合准备
    feature_sets = {'Mean+Std (Full)': np.ones(X.shape[1], dtype=bool)}  # 全选

    if do_selection:
        mask_selected = perform_feature_selection(X_train, y_train, feature_names)
        feature_sets['Mean+Std (RFECV)'] = mask_selected

        # 记录选中的特征
        sel_feats = np.array(feature_names)[mask_selected]
        print(f"     [Log] RFECV 保留特征示例: {list(sel_feats[:5])}...")
        pd.DataFrame({'Feature': sel_feats}).to_csv(os.path.join(save_dir, 'rfecv_selected_features.csv'), index=False)

    # 5. 循环训练
    print("\n" + "=" * 40)
    print("开始多模型对比训练")
    print("=" * 40)

    for model_name, model_inst in models.items():
        for f_set_name, mask in feature_sets.items():
            print(f"\n>> 正在处理: {model_name} [{f_set_name}]")

            X_train_sub = X_train[:, mask]
            X_test_sub = X_test[:, mask]

            # 优化与训练
            best_model = optimize_and_train(model_name, model_inst, params[model_name], X_train_sub, y_train)

            if best_model:
                # 预测
                y_pred = best_model.predict(X_test_sub)

                # 记录指标
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                print(f"   -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

                results.append({
                    'Model': model_name,
                    'Feature_Set': f_set_name,
                    'Accuracy': acc,
                    'F1_Score': f1,
                    'Best_Params': str(best_model.get_params())
                })

                # 保存混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                safe_name = f"{model_name}_{f_set_name}".replace(' ', '_').replace('+', '').replace('(', '').replace(
                    ')', '')
                cm_df.to_csv(os.path.join(save_dir, f'cm_{safe_name}.csv'))

                plt.figure(figsize=(10, 8))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{model_name} ({f_set_name})\nAcc={acc:.4f}')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'plot_cm_{safe_name}.png'))
                plt.close()

    # 6. 导出结果
    df_res = pd.DataFrame(results)
    res_path = os.path.join(save_dir, 'model_comparison_results.csv')
    df_res.to_csv(res_path, index=False)

    print("\n" + "=" * 60)
    print("【最终对比摘要】")
    if not df_res.empty:
        print(df_res[['Model', 'Feature_Set', 'Accuracy']].sort_values('Accuracy', ascending=False).to_string(
            index=False))
    print("=" * 60)

    # 绘图
    if not df_res.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_res, x='Model', y='Accuracy', hue='Feature_Set', palette='viridis')
        plt.title(f'Model Comparison (Mean+Std Only, Test={TEST_SIZE})')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_benchmark_plot.png'))

    sys.stdout = sys.stdout.terminal
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("完成", f"全流程分析结束！\n结果已保存至: {save_dir}")
    root.destroy()
    sys.exit(0)


if __name__ == "__main__":
    file_paths, save_dir, do_sel = select_files_gui()
    if file_paths:
        process_and_train(file_paths, save_dir, do_sel)