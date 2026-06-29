import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
# --- Regression Models ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
# --- Classification Models ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
# --- Metrics ---
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report, confusion_matrix, \
    accuracy_score, f1_score
# --- Other Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Checkbutton, Button, BooleanVar, Frame, Radiobutton, StringVar, \
    Text, Scrollbar, END, simpledialog
import os
import matplotlib as mpl
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import traceback
import sys
import threading
import joblib

try:
    from scipy.signal import savgol_filter
    from scipy.spatial.distance import pdist, squareform  # for Kennard-Stone
except ImportError:
    messagebox.showerror("缺少库", "需要 'scipy' 库来执行高级功能。\n请在命令行运行: pip install scipy")
    sys.exit()

mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


# --- Preprocessing Functions ---
def standard_normal_variate(X_input):
    if X_input.empty: return X_input
    X = np.asarray(X_input, dtype=float)
    mean_row, std_row = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    std_row[std_row == 0] = 1
    X_snv = (X - mean_row) / std_row
    return pd.DataFrame(X_snv, index=X_input.index, columns=X_input.columns) if isinstance(X_input,
                                                                                           pd.DataFrame) else X_snv


# --- Kennard-Stone Algorithm Implementation ---
def kennard_stone_group_splitter(X_groups_mean, n_train_groups):
    if n_train_groups >= len(X_groups_mean):
        return X_groups_mean.index.tolist(), []

    scaler_ks = StandardScaler()
    X_groups_scaled = scaler_ks.fit_transform(X_groups_mean)

    dists = squareform(pdist(X_groups_scaled, metric='euclidean'))
    i, j = np.unravel_index(np.argmax(dists), dists.shape)

    selected_indices = [i, j]
    remaining_indices = list(range(len(X_groups_mean)))
    remaining_indices.remove(i);
    remaining_indices.remove(j)

    while len(selected_indices) < n_train_groups:
        min_dists_to_selected = [np.min(dists[rem_idx, selected_indices]) for rem_idx in remaining_indices]
        farthest_idx_in_remaining = np.argmax(min_dists_to_selected)
        farthest_idx_original = remaining_indices.pop(farthest_idx_in_remaining)
        selected_indices.append(farthest_idx_original)

    all_group_ids = X_groups_mean.index.tolist()
    train_group_ids = [all_group_ids[i] for i in selected_indices]
    test_group_ids = [all_group_ids[i] for i in remaining_indices]
    return train_group_ids, test_group_ids


class SpectralAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("光谱分析工具 (最终科学版)")
        master.geometry("800x600")
        self.control_frame = Frame(master, padx=10, pady=10);
        self.control_frame.pack(fill=tk.X)
        self.start_button = Button(self.control_frame, text="开始分析", font=('Arial', 14, 'bold'),
                                   command=self.prepare_and_start_analysis)
        self.start_button.pack(pady=10)
        self.log_frame = Frame(master, padx=10, pady=10);
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_frame.grid_rowconfigure(0, weight=1);
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = Text(self.log_frame, wrap=tk.WORD, state='disabled', font=('Courier New', 11))
        self.scrollbar = Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew");
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.log("欢迎使用光谱分析最终科学版！")
        self.log("KS算法已优化，确保在整个浓度范围内均匀选择。")

    def log(self, message):
        def append_log():
            if not self.master.winfo_exists(): return
            self.log_text.config(state='normal')
            self.log_text.insert(END, str(message) + "\n");
            self.log_text.see(END)
            self.log_text.config(state='disabled')

        self.master.after(0, append_log)

    def prepare_and_start_analysis(self):
        self.log("\n" + "=" * 50)
        try:
            file_path1 = filedialog.askopenfilename(parent=self.master, title="选择第一个数据文件",
                                                    filetypes=[("CSV files", "*.csv")])
            if not file_path1: self.log("操作取消。"); return
            file_path2 = filedialog.askopenfilename(parent=self.master, title="选择第二个数据文件",
                                                    filetypes=[("CSV files", "*.csv")])
            if not file_path2: self.log("操作取消。"); return
            save_base_dir = filedialog.askdirectory(parent=self.master, title="选择主结果保存目录")
            if not save_base_dir: self.log("操作取消。"); return

            df_for_ui = pd.concat([pd.read_csv(file_path1), pd.read_csv(file_path2)], ignore_index=True)
            task_type = self.select_task_type()
            if not task_type: return
            category_column = self.select_column_role(df_for_ui.columns.tolist(), "请选择物质类别列")
            if not category_column: return
            target_column = self.select_column_role(df_for_ui.columns.tolist(), "请选择目标浓度列")
            if not target_column: return
            df_for_ui[target_column] = pd.to_numeric(df_for_ui[target_column], errors='raise')

            binned_column_name = self.define_concentration_bins(df_for_ui,
                                                                target_column) if task_type == 'classification' else None
            if task_type == 'classification' and not binned_column_name: return

            grouping_column = self.select_column_role(df_for_ui.columns.tolist(), "请选择组内划分依据列 (浓度)")
            if not grouping_column: return

            selected_categories = self.select_categories_to_model(df_for_ui[category_column].dropna().unique().tolist())
            if not selected_categories: return

            all_features = [c for c in df_for_ui.columns if
                            c not in [target_column, grouping_column, category_column, binned_column_name]]
            selected_features = self.select_features(all_features)
            if not selected_features: return

            preprocessing_method = self.select_preprocessing_method()
            if not preprocessing_method: return

            split_strategy, split_config = self.select_split_strategy(df_for_ui, grouping_column)
            if not split_strategy: return

            params = {'df': df_for_ui, 'save_dir': save_base_dir, 'target_col': target_column,
                      'group_col': grouping_column,
                      'features': selected_features, 'cat_col': category_column, 'sel_cat': selected_categories,
                      'split_strat': split_strategy, 'split_cfg': split_config, 'preproc': preprocessing_method,
                      'task': task_type, 'binned_col': binned_column_name}

            self.start_button.config(state="disabled", text="分析中...")
            threading.Thread(target=self.run_analysis_wrapper, args=(params,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("准备错误", f"分析准备阶段出错: {e}");
            self.log(f"错误: {e}")

    def run_analysis_wrapper(self, params):
        try:
            self.run_batch_analysis(params)
        except Exception as e:
            self.log(f"!!! 发生严重错误 !!!\n{e}\n{traceback.format_exc()}")
            self.master.after(0, lambda err=e: messagebox.showerror("程序遇到意外错误", f"发生未处理的错误: {err}"))
        finally:
            self.master.after(0, lambda: self.start_button.config(state="normal", text="开始分析"))
            self.log("所有分析任务结束。")

    def run_batch_analysis(self, params):
        overall_summary = []
        for category in params['sel_cat']:
            self.log("\n" + "#" * 60 + f"\n### 开始为类别: [{category}] 进行独立分析 ###")
            df_current = params['df'][params['df'][params['cat_col']] == category].copy()
            if len(df_current) < 10:
                self.log(f"警告: 类别 [{category}] 样本数 ({len(df_current)}) 过少，跳过。");
                continue

            category_save_dir = os.path.join(params['save_dir'], str(category))
            os.makedirs(category_save_dir, exist_ok=True)

            model_info = self.run_single_analysis(df_current, category, category_save_dir, params)
            if model_info: overall_summary.append(model_info)

        self.log("\n" + "#" * 60 + "\n### 所有类别分析完成，生成总汇总报告... ###")
        if overall_summary:
            summary_df = pd.DataFrame(overall_summary)
            summary_path = os.path.join(params['save_dir'], "overall_summary_report.xlsx")
            summary_df.to_excel(summary_path, index=False)
            self.log(f"成功！总性能汇总报告已保存至: {summary_path}")

    def run_single_analysis(self, df, category_name, save_dir, params):
        self.log(f"步骤 1: [{category_name}] 准备与预处理 (方法: {params['preproc']})...");
        X = df[params['features']].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df[params['binned_col']] if params['task'] == 'classification' else df[params['target_col']]
        groups = df[params['group_col']]
        valid_indices = y.notna()
        X, y, groups = X.loc[valid_indices], y.loc[valid_indices], groups.loc[valid_indices]

        method = params['preproc']
        if method != '无': X = standard_normal_variate(X)
        if 'SG' in method:
            deriv = 1 if '1st' in method else 2
            win_len = min(11, len(X.columns) - (1 if len(X.columns) % 2 == 0 else 0))
            if win_len >= 3: X = pd.DataFrame(savgol_filter(X, window_length=win_len, polyorder=2, deriv=deriv),
                                              index=X.index, columns=X.columns)

        self.log(f"步骤 2: [{category_name}] 应用统一划分策略 ({params['split_strat']})...");
        train_groups, test_groups = [], []
        if params['split_strat'] == 'Kennard-Stone':
            mean_spectra = X.groupby(groups).mean()
            if params['split_cfg'] >= len(mean_spectra):
                self.log(f"警告: 训练集组数 >= 总组数，无法划分，跳过 [{category_name}]。");
                return None
            train_groups, test_groups = kennard_stone_group_splitter(mean_spectra, params['split_cfg'])
            self.log(f"KS算法选择训练组: {sorted(train_groups)}")
            self.log(f"KS算法选择测试组: {sorted(test_groups)}")
        elif params['split_strat'] == '手动按组划分':
            test_groups = [g for g in params['split_cfg'] if g in groups.unique()]
            if not test_groups: self.log(f"错误: 为 [{category_name}] 指定的测试组均不存在，跳过。"); return None
        elif params['split_strat'] == '随机按组划分':
            gss = GroupShuffleSplit(n_splits=1, test_size=params['split_cfg'], random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            test_groups = groups.iloc[test_idx].unique().tolist()

        if test_groups:
            test_indices = groups.isin(test_groups)
        else:
            test_indices = ~groups.isin(train_groups)  # For KS

        X_train, X_test, y_train, y_test = X.loc[~test_indices], X.loc[test_indices], y.loc[~test_indices], y.loc[
            test_indices]
        if X_train.empty or X_test.empty: self.log(f"错误: 划分后训练集或测试集为空! 跳过。"); return None

        self.log(f"划分成功！训练集: {len(X_train)}, 测试集: {len(X_test)}。")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X.columns)
        le = LabelEncoder() if params['task'] == 'classification' else None
        if le: y_train, y_test = le.fit_transform(y_train), le.transform(y_test)

        self.log(f"步骤 3: [{category_name}] 超参数优化...");
        results = self.run_hyperopt_and_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, save_dir,
                                                   category_name, params['task'], le)
        if not results: return None
        best_model_name, best_params, best_model_report = results

        self.log(f"\n步骤 6: [{category_name}] 重训练并导出专属模型...")
        scaler_final = StandardScaler().fit(X)
        final_model = self.get_models(params['task'])[best_model_name].set_params(**best_params)
        if params['task'] == 'classification':
            le_final = LabelEncoder().fit(y)
            final_model.fit(scaler_final.transform(X), le_final.transform(y))
            self.export_best_model(best_model_name, final_model, scaler_final, X.columns, params['preproc'], save_dir,
                                   le_final)
        else:
            final_model.fit(scaler_final.transform(X), y)
            self.export_best_model(best_model_name, final_model, scaler_final, X.columns, params['preproc'], save_dir)

        summary_info = {'Category': category_name, 'Task': params['task'], 'Preprocessing': params['preproc'],
                        'Best_Model': best_model_name.upper()}
        summary_info.update(best_model_report)
        return summary_info

    def run_hyperopt_and_evaluation(self, X_train, y_train, X_test, y_test, save_dir, category_name, task_type,
                                    le=None):
        models, hp_space = self.get_models(task_type), self.get_hp_space(task_type)
        final_models, all_params, all_reports, all_preds = {}, {}, [], {}
        original_stdout, sys.stdout = sys.stdout, self

        for name, model in models.items():
            scoring = 'neg_root_mean_squared_error' if task_type == 'regression' else 'f1_macro'

            def objective(params):
                p = {k: int(v) if k in ['max_depth', 'n_estimators'] else v for k, v in params.items()}
                model.set_params(**p)
                n_s = min(4, len(np.unique(y_train)) if task_type == 'classification' and len(
                    np.unique(y_train)) > 1 else len(X_train))
                if n_s <= 1: return 999
                score = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=n_s, shuffle=True, random_state=42),
                                        scoring=scoring, n_jobs=-1).mean()
                return {'loss': -score, 'status': STATUS_OK}

            trials = Trials();
            best_params_raw = fmin(fn=objective, space=hp_space[name], algo=tpe.suggest, max_evals=30, trials=trials,
                                   rstate=np.random.default_rng(42), show_progressbar=False)
            best_params = {k: int(v) if k in ['max_depth', 'n_estimators'] else v for k, v in
                           space_eval(hp_space[name], best_params_raw).items()}
            all_params[name] = best_params;
            final_model = models[name].set_params(**best_params).fit(X_train, y_train);
            final_models[name] = final_model
            y_pred = final_model.predict(X_test);
            all_preds[name] = y_pred

            if task_type == 'regression':
                r2, rmse, mae = r2_score(y_test, y_pred), np.sqrt(
                    mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)
                all_reports.append({
                                       'str': f"--- {name.upper()} ---\nParams: {best_params}\nR²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}\n",
                                       'name': name, 'Test_R2_Score': r2, 'Test_RMSE': rmse, 'Test_MAE': mae})
            else:
                acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro', zero_division=0)
                all_reports.append(
                    {'str': f"--- {name.upper()} ---\nParams: {best_params}\nAccuracy: {acc:.4f}, Macro F1: {f1:.4f}\n",
                     'name': name, 'Test_Accuracy': acc, 'Test_Macro_F1': f1})

        sys.stdout = original_stdout
        self.log(f"步骤 4: [{category_name}] 在测试集上评估所有模型...")
        if task_type == 'classification':
            self.export_classification_report_to_txt(y_test, all_preds, all_reports, le, save_dir)
        else:
            self.export_predictions_to_excel(pd.Series(y_test, name="True_Value"), all_preds, save_dir)

        best_name, best_report = self.find_best_model(all_reports, category_name, task_type)
        if not best_name: return None

        self.log(f"步骤 5: [{category_name}] 生成图表和报告...")
        for name, model in final_models.items():
            if task_type == 'regression':
                self.plot_predictions_vs_true(pd.Series(y_test, name="True_Value"), all_preds[name], name, save_dir,
                                              category_name)
            else:
                self.plot_confusion_matrix(y_test, all_preds[name], name, le.classes_, save_dir, category_name)
            save_path = os.path.join(save_dir,
                                     f'BEST_MODEL_{best_name}_feature_importance.png') if name == best_name else None
            self.display_feature_importance(model, X_train.columns, name, save_path)

        return best_name, all_params.get(best_name, {}), best_report

    # --- Helper Functions Below ---
    def get_models(self, task_type):
        if task_type == 'regression':
            return {'dt': DecisionTreeRegressor(random_state=42),
                    'rf': RandomForestRegressor(random_state=42, n_jobs=-1), 'svr': SVR(),
                    'xgb': XGBRegressor(random_state=42, n_jobs=-1), 'ridge': Ridge(random_state=42)}
        else:  # classification
            return {'dt': DecisionTreeClassifier(random_state=42),
                    'rf': RandomForestClassifier(random_state=42, n_jobs=-1), 'svc': SVC(random_state=42),
                    'xgb': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')}

    def get_hp_space(self, task_type):
        shared_space = {'dt': {'max_depth': hp.quniform('max_depth', 3, 20, 1)},
                        'rf': {'n_estimators': hp.quniform('n_estimators', 50, 400, 10),
                               'max_depth': hp.quniform('max_depth', 5, 50, 1)},
                        'xgb': {'n_estimators': hp.quniform('n_estimators', 50, 400, 10),
                                'max_depth': hp.quniform('max_depth', 3, 15, 1)}}
        if task_type == 'regression':
            shared_space.update({'svr': {'C': hp.loguniform('C', np.log(0.1), np.log(100)),
                                         'gamma': hp.loguniform('gamma', np.log(0.001), np.log(1))},
                                 'ridge': {'alpha': hp.loguniform('alpha', np.log(0.01), np.log(100))}})
        else:  # classification
            shared_space.update({'svc': {'C': hp.loguniform('C', np.log(0.1), np.log(100)),
                                         'gamma': hp.loguniform('gamma', np.log(0.001), np.log(1))}})
        return shared_space

    def export_best_model(self, name, model, scaler, features, preproc, save_dir, le=None):
        pkg = {'model_name': name, 'model': model, 'scaler': scaler, 'feature_columns': features.tolist(),
               'preprocessing_method': preproc, 'label_encoder': le}
        try:
            joblib.dump(pkg, os.path.join(save_dir, f'best_model_{name}.joblib'))
        except Exception as e:
            self.log(f"错误：导出模型失败 - {e}")

    def plot_predictions_vs_true(self, y_true, y_pred, model_name, save_dir, cat_name):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
        lims = [min(y_true.min(), y_pred.min()) * 0.95, max(y_true.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x')
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f'[{cat_name}] {model_name.upper()}: 真实 vs. 预测', fontsize=18)
        ax.set_xlabel('真实值', fontsize=14)
        ax.set_ylabel('预测值', fontsize=14)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{model_name}_preds_vs_true.png'))
            plt.close(fig)
        else:
            plt.show(block=False)

    def plot_confusion_matrix(self, y_true, y_pred, model_name, class_names, save_dir, cat_name):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'[{cat_name}] {model_name.upper()}: 混淆矩阵', fontsize=18)
        ax.set_xlabel('预测', fontsize=14)
        ax.set_ylabel('真实', fontsize=14)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
            plt.close(fig)
        else:
            plt.show(block=False)

    def export_classification_report_to_txt(self, y_true, preds_dict, reports, le, save_dir):
        path = os.path.join(save_dir, "classification_summary.txt")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("=" * 20 + " 模型评估 " + "=" * 20 + "\n\n")
                for name, y_pred in preds_dict.items():
                    f.write(
                        f"--- {name.upper()} 分类报告 ---\n{classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)}\n{'-' * 50}\n\n")
                f.write("\n\n" + "=" * 20 + " 性能汇总 " + "=" * 20 + "\n\n")
                summary_df = pd.DataFrame(reports)[['name', 'Test_Accuracy', 'Test_Macro_F1']].rename(
                    columns={'name': 'Model'}).sort_values(by='Test_Macro_F1', ascending=False).reset_index(drop=True)
                f.write(summary_df.to_string(float_format="%.4f"))
        except Exception as e:
            self.log(f"导出TXT失败: {e}")

    def export_predictions_to_excel(self, y_true, preds_dict, save_dir):
        path = os.path.join(save_dir, "detailed_prediction_results.xlsx")
        try:
            df = pd.DataFrame({'True_Value': y_true})
            for name, y_pred in preds_dict.items():
                df[f'Predicted_{name.upper()}'] = pd.Series(y_pred, index=y_true.index)
                df[f'Error_{name.upper()}'] = df['True_Value'] - df[f'Predicted_{name.upper()}']
            df.to_excel(path, index=True)
        except Exception as e:
            self.log(f"导出Excel失败: {e}")

    def select_task_type(self):
        d = Toplevel(self.master);
        d.title("选择分析任务");
        v = StringVar(value="regression");
        r = {'t': ''}
        tk.Label(d, text="选择分析任务:", font=('Arial', 16, 'bold')).pack(pady=10)
        tk.Radiobutton(d, text="定量分析 (回归)", variable=v, value="regression", font=('Arial', 12)).pack(anchor='w',
                                                                                                           padx=30)
        tk.Radiobutton(d, text="浓度范围验证 (分类)", variable=v, value="classification", font=('Arial', 12)).pack(
            anchor='w', padx=30)

        def c(): r['t'] = v.get(); d.destroy()

        Button(d, text="确认", command=c, font=('Arial', 12, 'bold')).pack(pady=15);
        d.transient(self.master);
        d.grab_set();
        self.master.wait_window(d)
        return r['t']

    def define_concentration_bins(self, df, target_col):
        d = Toplevel(self.master);
        d.title("定义浓度范围");
        v = StringVar();
        binned_name = f"{target_col}_Range"

        def c():
            try:
                edges = sorted([float(e.strip()) for e in v.get().split(',')])
                if not edges: messagebox.showerror("输入错误", "请输入至少一个分割点。", parent=d); return
                full_edges = [-np.inf] + edges + [np.inf];
                labels = [f"<= {edges[0]}"] + [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)] + [
                    f"> {edges[-1]}"]
                df[binned_name] = pd.cut(df[target_col], bins=full_edges, labels=labels)
                self.log(f"成功创建浓度范围列 '{binned_name}'.\n{df[binned_name].value_counts().to_string()}");
                d.destroy()
            except Exception as e:
                messagebox.showerror("输入错误", f"无法解析分割点, 请确保输入正确。\n例如: 30, 80\n错误: {e}", parent=d)

        tk.Label(d, text="定义浓度范围分割点", font=('Arial', 14, 'bold')).pack(pady=10)
        tk.Label(d, text=f"当前数据范围: {df[target_col].min():.2f} - {df[target_col].max():.2f}",
                 font=('Arial', 10)).pack()
        tk.Label(d, text="输入分割点, 用逗号隔开 (例如: 30, 80)", font=('Arial', 10)).pack(pady=5)
        tk.Entry(d, textvariable=v, width=50).pack(pady=5);
        Button(d, text="创建范围", command=c, font=('Arial', 12, 'bold')).pack(pady=10)
        d.transient(self.master);
        d.grab_set();
        self.master.wait_window(d)
        return binned_name if binned_name in df.columns else None

    def select_preprocessing_method(self):
        d = Toplevel(self.master);
        d.title("选择光谱预处理方法");
        v = StringVar(value="SNV");
        r = {'m': ''}
        tk.Label(d, text="选择预处理方法:", font=('Arial', 16, 'bold')).pack(pady=10)
        for opt in ["无", "SNV", "SNV + SG 1st Der", "SNV + SG 2nd Der"]:
            tk.Radiobutton(d, text=opt, variable=v, value=opt, font=('Arial', 12)).pack(anchor='w', padx=30)

        def c(): r['m'] = v.get(); d.destroy()

        Button(d, text="确认", command=c, font=('Arial', 12, 'bold')).pack(pady=15);
        d.transient(self.master);
        d.grab_set();
        self.master.wait_window(d)
        return r['m']

    def select_split_strategy(self, df, group_col):
        d = Toplevel(self.master);
        d.title("选择统一的数据划分策略");
        v = StringVar(value="Kennard-Stone");
        r = {'s': '', 'c': None}
        ug = sorted(df[group_col].dropna().unique());
        tk.Label(d, text="选择统一的划分策略:", font=('Arial', 16, 'bold')).pack(pady=10)
        for opt in ["Kennard-Stone", "随机按组划分", "手动按组划分"]:
            tk.Radiobutton(d, text=opt, variable=v, value=opt, font=('Arial', 12)).pack(anchor='w', padx=20)
        f = Frame(d, bd=1, relief=tk.SUNKEN);
        f.pack(pady=10, padx=20, fill=tk.BOTH, expand=True);
        tk.Label(f, text="配置 (根据上方选择):", font=('Arial', 11, 'bold')).pack(pady=5)
        ca = tk.Canvas(f);
        sc = tk.Scrollbar(f, orient="vertical", command=ca.yview);
        inf = Frame(ca);
        inf.bind("<Configure>", lambda e: ca.configure(scrollregion=ca.bbox("all")))
        ca.create_window((0, 0), window=inf, anchor="nw");
        ca.configure(yscrollcommand=sc.set);
        gv = {g: BooleanVar(value=False) for g in ug}
        for g in ug: Checkbutton(inf, text=str(g), var=gv[g], font=('Arial', 11)).pack(anchor='w', padx=10)
        ca.pack(side="left", fill="both", expand=True);
        sc.pack(side="right", fill="y")

        def c():
            r['s'] = v.get()
            if r['s'] == '手动按组划分':
                tg = [g for g, v_ in gv.items() if v_.get()];
                if not tg: messagebox.showerror("错误", "请至少选择一个组！", parent=d); return
                r['c'] = tg
            else:
                prompt = "随机选择几个浓度组作测试集?" if r['s'] == '随机按组划分' else "选择几个浓度组作训练集 (KS)?"
                max_val = len(ug) - 1;
                unit = "测试集" if r['s'] == '随机按组划分' else "训练集"
                num = simpledialog.askinteger(f"设置{unit}数量", f"{prompt} (1-{max_val})", parent=d, minvalue=1,
                                              maxvalue=max_val if max_val > 0 else 1)
                if num is None: return
                r['c'] = num
            d.destroy()

        Button(d, text="确认", command=c, font=('Arial', 12, 'bold')).pack(pady=15);
        d.transient(self.master);
        d.grab_set();
        self.master.wait_window(d)
        return r['s'], r['c']

    def find_best_model(self, reports, cat_name, task_type):
        if not reports: return None, None
        metric = 'Test_R2_Score' if task_type == 'regression' else 'Test_Macro_F1'
        best_report = max(reports, key=lambda x: x[metric])
        self.log(f"类别 [{cat_name}] 的最佳模型是: {best_report['name'].upper()} ({metric}: {best_report[metric]:.4f})")
        return best_report['name'], {k: v for k, v in best_report.items() if k.startswith('Test_')}

    def write(self, text):
        if text.strip(): self.log(text.strip())

    def flush(self):
        pass

    def select_column_role(self, columns, title):
        d = Toplevel(self.master);
        d.title(title);
        v = StringVar(value=columns[0] if columns else "");
        r = {'c': ''}
        tk.Label(d, text=title, font=('Arial', 14, 'bold')).pack(pady=10);
        f = Frame(d);
        f.pack(fill=tk.BOTH, expand=True, padx=10)
        ca = tk.Canvas(f);
        sc = tk.Scrollbar(f, orient="vertical", command=ca.yview);
        inf = Frame(ca);
        inf.bind("<Configure>", lambda e: ca.configure(scrollregion=ca.bbox("all")))
        ca.create_window((0, 0), window=inf, anchor="nw");
        ca.configure(yscrollcommand=sc.set)
        for col in columns: Radiobutton(inf, text=col, variable=v, value=col, font=('Arial', 12)).pack(anchor='w',
                                                                                                       padx=20)
        ca.pack(side="left", fill="both", expand=True);
        sc.pack(side="right", fill="y")

        def c(): r['c'] = v.get(); d.destroy()

        Button(d, text="确认", command=c, font=('Arial', 14, 'bold')).pack(pady=10);
        d.transient(self.master);
        d.grab_set();
        self.master.wait_window(d)
        return r['c']

    def select_categories_to_model(self, all_cat):
        d = Toplevel(self.master);
        d.title("选择要建模的物质类别");
        sl = [];
        v = {c: BooleanVar(value=True) for c in all_cat}

        def c():
            nonlocal sl;
            sl = [c for c, v_ in v.items() if v_.get()]
            if not sl: messagebox.showerror("错误", "请至少选择一个类别！", parent=d); return
            d.destroy()

        tf = Frame(d, pady=5);
        tf.pack(fill=tk.X, padx=10)
        Button(tf, text="全选", command=lambda: [v_.set(True) for v_ in v.values()]).pack(side=tk.LEFT, expand=True)
        Button(tf, text="全不选", command=lambda: [v_.set(False) for v_ in v.values()]).pack(side=tk.LEFT, expand=True)
        mf = Frame(d);
        mf.pack(fill=tk.BOTH, expand=True, padx=10)
        ca = tk.Canvas(mf);
        sc = tk.Scrollbar(mf, orient="vertical", command=ca.yview);
        inf = Frame(ca);
        inf.bind("<Configure>", lambda e: ca.configure(scrollregion=ca.bbox("all")))
        ca.create_window((0, 0), window=inf, anchor="nw");
        ca.configure(yscrollcommand=sc.set)
        for cat in all_cat: Checkbutton(inf, text=cat, var=v[cat], font=('Arial', 12)).pack(anchor='w', padx=20)
        ca.pack(side="left", fill="both", expand=True);
        sc.pack(side="right", fill="y")
        Button(d, text="确认选择", command=c, font=('Arial', 14, 'bold')).pack(pady=10);
        d.transient(self.master);
        d.grab_set();
        self.master.wait_window(d)
        return sl

    def select_features(self, all_feat):
        d = Toplevel(self.master);
        d.title("请选择特征列 (X)");
        sl = [];
        v = {f: BooleanVar(value=True) for f in all_feat}

        def c():
            nonlocal sl;
            sl = [f for f, v_ in v.items() if v_.get()]
            if not sl: messagebox.showerror("错误", "请至少选择一个特征列！", parent=d); return
            d.destroy()

        tf = Frame(d, pady=5);
        tf.pack(fill=tk.X, padx=10)
        Button(tf, text="全选", command=lambda: [v_.set(True) for v_ in v.values()]).pack(side=tk.LEFT, expand=True)
        Button(tf, text="全不选", command=lambda: [v_.set(False) for v_ in v.values()]).pack(side=tk.LEFT, expand=True)
        mf = Frame(d);
        mf.pack(fill=tk.BOTH, expand=True, padx=10)
        ca = tk.Canvas(mf);
        sc = tk.Scrollbar(mf, orient="vertical", command=ca.yview);
        inf = Frame(ca);
        inf.bind("<Configure>", lambda e: ca.configure(scrollregion=ca.bbox("all")))
        ca.create_window((0, 0), window=inf, anchor="nw");
        ca.configure(yscrollcommand=sc.set)
        for feat in all_feat: Checkbutton(inf, text=feat, var=v[feat], font=('Arial', 12)).pack(anchor='w', padx=20)
        ca.pack(side="left", fill="both", expand=True);
        sc.pack(side="right", fill="y")
        Button(d, text="确认选择", command=c, font=('Arial', 14, 'bold')).pack(pady=10);
        d.transient(self.master);
        d.grab_set();
        self.master.wait_window(d)
        return sl

    def display_feature_importance(self, model, columns, model_name, save_path=None):
        imp = None
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        if imp is None: return
        df_imp = pd.DataFrame({'feature': columns, 'importance': imp}).sort_values('importance', ascending=False).head(
            20)
        self.log(f"\n--- {model_name} Top 20 Importances ---\n{df_imp.to_string()}")
        fig, ax = plt.subplots(figsize=(12, 8));
        sns.barplot(x='importance', y='feature', data=df_imp, ax=ax, palette='viridis')
        ax.set_title(f'{model_name.upper()} Feature Importance', fontsize=22);
        ax.set_xlabel('Importance', fontsize=18);
        ax.set_ylabel('Feature', fontsize=18)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path);
            self.log(f"已保存特征图: {os.path.basename(save_path)}");
            plt.close(fig)
        else:
            plt.show(block=False)


if __name__ == '__main__':
    root = tk.Tk()
    app = SpectralAnalysisApp(root)
    root.mainloop()
