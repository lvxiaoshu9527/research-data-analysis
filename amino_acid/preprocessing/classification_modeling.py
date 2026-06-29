import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier # 导入 VotingClassifier 和 StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Import HyperOpt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Set Matplotlib default font to support English and ensure Arial and other English fonts are available
mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


def select_file():
    """Opens a file selection dialog to choose the training data CSV file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Training Data CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    return file_path


def get_save_directory():
    """Opens a folder selection dialog for the user to choose a directory for saving files."""
    root = tk.Tk()
    root.withdraw()
    save_dir = filedialog.askdirectory(
        title="Select Directory to Save Results"
    )
    return save_dir


def save_scatter_data(data, labels, filename, save_dir):
    """Saves scatter plot data to a TXT file in the specified directory."""
    if save_dir:
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'w') as f:
            f.write("Component 1\tComponent 2\tLabel\n")
            for i in range(data.shape[0]):
                f.write(f"{data[i, 0]:.6f}\t{data[i, 1]:.6f}\t{labels[i]}\n")
        print(f"Scatter plot data saved to: {file_path}")
    else:
        print("No save path selected, scatter plot data not saved.")


def save_feature_importance(feature_importance_df, filename, save_dir):
    """Saves feature importance data to a TXT file in the specified directory."""
    if save_dir:
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'w') as f:
            f.write("Feature\tImportance\n")
            for index, row in feature_importance_df.iterrows():
                f.write(f"{row['feature']}\t{row['importance']:.6f}\n")
        print(f"Feature importance data saved to: {file_path}")
    else:
        print("No save path selected, feature importance data not saved.")


def main():
    """Main function to control the script's execution flow."""
    input_file_path = select_file()

    if input_file_path:
        try:
            df_original = pd.read_csv(input_file_path)
        except FileNotFoundError:
            messagebox.showerror("Error", f"File {input_file_path} not found.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Error reading file: {e}")
            return

        print("Original Data Info:")
        print(df_original.info())
        print("\nFirst 5 Rows of Original Data:")
        print(df_original.head())

        # Define chirality list and corresponding intensity and shift column names
        chiralities = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']
        intensity_cols = [f'{c}_intensity' for c in chiralities]
        shift_cols = [f'{c}_shift' for c in chiralities]

        target_col = 'AA'

        # Check if necessary columns exist in the original data
        required_cols = ['AA', '浓度/uM'] + intensity_cols + shift_cols
        if not all(col in df_original.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_original.columns]
            messagebox.showerror("Error",
                                 f"Missing required columns in original data: {missing_cols}\nPlease ensure the CSV file contains all nanotube intensity and shift features, as well as AA and 浓度/uM.")
            return

        # --- Wide to Long format conversion starts ---
        long_format_data = []

        for index, row in df_original.iterrows():
            aa_label = row['AA']
            # We still need '浓度/uM' for the long format conversion, but it won't be a feature later.
            concentration = row['浓度/uM']

            for chirality in chiralities:
                intensity_col_name = f'{chirality}_intensity'
                shift_col_name = f'{chirality}_shift'

                new_record = {
                    'AA': aa_label,
                    '浓度/uM': concentration,
                    '碳管手性': chirality,
                    'intensity': row[intensity_col_name],
                    'shift': row[shift_col_name]
                }
                long_format_data.append(new_record)

        df_long = pd.DataFrame(long_format_data)
        # --- Wide to Long format conversion ends ---

        print("\nTransformed Long Format Data Info:")
        print(df_long.info())
        print("\nFirst 5 Rows of Transformed Long Format Data:")
        print(df_long.head())
        print(f"\nTotal samples after transformation: {len(df_long)} (Original samples * 4)")

        # --- Feature Selection Adjustment ---
        # Only retain 'intensity' and 'shift' as features
        feature_cols_final = ['intensity', 'shift']
        X = df_long[feature_cols_final].copy()
        y_labels = df_long[target_col].copy()

        # No label encoding for '碳管手性' as it's not a feature anymore.

        # Label Encoding for AA
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)
        unique_labels = label_encoder.classes_
        original_labels_for_viz = y_labels.values

        X.fillna(0, inplace=True)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Select a directory for saving all TXT files upfront
        save_base_dir = get_save_directory()
        if not save_base_dir:
            messagebox.showinfo("Info", "No save directory selected, data and reports will not be saved to files.")
            save_base_dir = None

        all_models_report = []
        print("\n===== Model Training and Evaluation =====\n")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # --- HyperOpt Bayesian Optimization ---
        MAX_EVALS = 50

        # Define choices lists explicitly to use for both space definition and mapping
        dtc_max_depth_options = list(range(1, 20))
        dtc_min_samples_split_options = list(range(2, 20))
        dtc_min_samples_leaf_options = list(range(1, 20))
        dtc_criterion_options = ['gini', 'entropy']

        lr_penalty_options = ['l1', 'l2']

        mlp_hidden_layers_options = [(50,), (100,), (50, 50), (100, 50)]
        mlp_activation_options = ['relu', 'tanh']
        mlp_learning_rate_options = ['constant', 'adaptive']

        rfc_n_estimators_options = list(range(50, 200, 10))
        rfc_max_depth_options = list(range(1, 20))
        rfc_min_samples_split_options = list(range(2, 20))
        rfc_min_samples_leaf_options = list(range(1, 20))
        rfc_criterion_options = ['gini', 'entropy']

        svc_kernel_options = ['linear', 'rbf']

        xgb_n_estimators_options = list(range(50, 200, 10))
        xgb_max_depth_options = list(range(1, 10))


        # Define the objective function for HyperOpt
        def objective(params, model_type, X_train_opt, y_train_opt, choices_map):
            mapped_params = params.copy()

            # Apply mapping for parameters that use hp.choice and return indices
            for param_name, options_list in choices_map.get(model_type, {}).items():
                if param_name in mapped_params and isinstance(mapped_params[param_name], (int, np.integer)):
                    idx = int(mapped_params[param_name])
                    if 0 <= idx < len(options_list):
                        mapped_params[param_name] = options_list[idx]
                    else:
                        print(f"Warning: For {model_type}, parameter '{param_name}' got out-of-range index {idx}. Using default/first choice.")
                        mapped_params[param_name] = options_list[0] if options_list else None

            # Instantiate model with mapped parameters
            if model_type == "DecisionTree":
                model = DecisionTreeClassifier(**mapped_params, random_state=42)
            elif model_type == "LogisticRegression":
                solver = 'liblinear'
                if 'penalty' in mapped_params and mapped_params['penalty'] == 'elasticnet':
                    solver = 'saga'
                model = LogisticRegression(solver=solver, **mapped_params, random_state=42)
            elif model_type == "MLPClassifier":
                model = MLPClassifier(**mapped_params, random_state=42, max_iter=500)
            elif model_type == "RandomForest":
                model = RandomForestClassifier(**mapped_params, random_state=42)
            elif model_type == "SVC":
                model = SVC(**mapped_params, random_state=42, probability=True) # SVC需要probability=True支持soft voting
            elif model_type == "XGBoost":
                model = XGBClassifier(**mapped_params, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
            else:
                raise ValueError("Unknown model type")

            try:
                # 使用 cross_validate 获取多个指标
                cv_results = cross_validate(model, X_train_opt, y_train_opt, cv=skf,
                                            scoring=['accuracy', 'f1_weighted', 'f1_macro'], # 添加 f1_macro
                                            error_score='raise')
                mean_accuracy = np.mean(cv_results['test_accuracy'])
                mean_f1_weighted = np.mean(cv_results['test_f1_weighted'])
                mean_f1_macro = np.mean(cv_results['test_f1_macro']) # 获取 macro-F1

                # 定义权重：这里可以根据需要调整，示例使用 0.4 准确率 + 0.4 加权F1 + 0.2 Macro-F1
                weight_accuracy = 0.4
                weight_f1_weighted = 0.4
                weight_f1_macro = 0.2 # 为 macro-F1 分配权重

                # 计算加权和
                combined_score = (weight_accuracy * mean_accuracy) + \
                                 (weight_f1_weighted * mean_f1_weighted) + \
                                 (weight_f1_macro * mean_f1_macro)

                # HyperOpt 最小化损失，所以我们返回 1 - score
                loss = 1 - combined_score
                return {'loss': loss, 'status': STATUS_OK}
            except Exception as e:
                # 如果模型训练失败（例如参数组合无效），返回一个大损失值
                print(f"Model training failed for {model_type} with params {mapped_params}: {e}")
                return {'loss': 1.0, 'status': STATUS_OK} # 返回最大损失，表示这个参数组合很差


        # Define hyperparameter search spaces (unchanged)
        space_dtc = {
            'max_depth': hp.choice('max_depth', dtc_max_depth_options),
            'min_samples_split': hp.choice('min_samples_split', dtc_min_samples_split_options),
            'min_samples_leaf': hp.choice('min_samples_leaf', dtc_min_samples_leaf_options),
            'criterion': hp.choice('criterion', dtc_criterion_options)
        }

        space_lr = {
            'C': hp.loguniform('C', np.log(0.001), np.log(100)),
            'penalty': hp.choice('penalty', lr_penalty_options)
        }

        space_mlp = {
            'hidden_layer_sizes': hp.choice('hidden_layer_sizes', mlp_hidden_layers_options),
            'activation': hp.choice('activation', mlp_activation_options),
            'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(0.1)),
            'learning_rate': hp.choice('learning_rate', mlp_learning_rate_options)
        }

        space_rfc = {
            'n_estimators': hp.choice('n_estimators', rfc_n_estimators_options),
            'max_depth': hp.choice('max_depth', rfc_max_depth_options),
            'min_samples_split': hp.choice('min_samples_split', rfc_min_samples_split_options),
            'min_samples_leaf': hp.choice('min_samples_leaf', rfc_min_samples_leaf_options),
            'criterion': hp.choice('criterion', rfc_criterion_options)
        }

        space_svc = {
            'C': hp.loguniform('C', np.log(0.1), np.log(100)),
            'kernel': hp.choice('kernel', svc_kernel_options),
            'gamma': hp.loguniform('gamma', np.log(0.001), np.log(0.1))
        }

        space_xgb = {
            'n_estimators': hp.choice('n_estimators', xgb_n_estimators_options),
            'max_depth': hp.choice('max_depth', xgb_max_depth_options),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
        }

        # Centralized choices_map for clarity and reuse (unchanged)
        global_choices_map = {
            "DecisionTree": {
                "max_depth": dtc_max_depth_options,
                "min_samples_split": dtc_min_samples_split_options,
                "min_samples_leaf": dtc_min_samples_leaf_options,
                "criterion": dtc_criterion_options
            },
            "LogisticRegression": {
                "penalty": lr_penalty_options
            },
            "MLPClassifier": {
                "hidden_layer_sizes": mlp_hidden_layers_options,
                "activation": mlp_activation_options,
                "learning_rate": mlp_learning_rate_options
            },
            "RandomForest": {
                "n_estimators": rfc_n_estimators_options,
                "max_depth": rfc_max_depth_options,
                "min_samples_split": rfc_min_samples_split_options,
                "min_samples_leaf": rfc_min_samples_leaf_options,
                "criterion": rfc_criterion_options
            },
            "SVC": {
                "kernel": svc_kernel_options
            },
            "XGBoost": {
                "n_estimators": xgb_n_estimators_options,
                "max_depth": xgb_max_depth_options
            }
        }


        optimized_models_configs = { # 更改变量名以避免与下面的 final_models 混淆
            "DecisionTree": {"model_class": DecisionTreeClassifier, "space": space_dtc},
            "LogisticRegression": {"model_class": LogisticRegression, "space": space_lr},
            "MLPClassifier": {"model_class": MLPClassifier, "space": space_mlp},
            "RandomForest": {"model_class": RandomForestClassifier, "space": space_rfc},
            "SVC": {"model_class": SVC, "space": space_svc},
            "XGBoost": {"model_class": XGBClassifier, "space": space_xgb}
        }

        final_models = {}
        model_performance_metrics = {} # 存储所有模型的性能指标，用于后续对比图

        for name, config in optimized_models_configs.items():
            print(f"\n===== Optimizing {name} =====")
            trials = Trials()
            best_raw = fmin(
                fn=lambda params: objective(params, name, X_train, y_train, global_choices_map),
                space=config["space"],
                algo=tpe.suggest,
                max_evals=MAX_EVALS,
                trials=trials
            )

            # Apply mapping to best_raw to get actual parameter values for final model instantiation
            final_model_params = best_raw.copy()
            for param_name, options_list in global_choices_map.get(name, {}).items():
                if param_name in final_model_params and isinstance(final_model_params[param_name], (int, np.integer)):
                    idx = int(final_model_params[param_name])
                    if 0 <= idx < len(options_list):
                        final_model_params[param_name] = options_list[idx]
                    else:
                        print(f"Warning: For final model {name}, parameter '{param_name}' got out-of-range index {idx}. Using default/first choice.")
                        final_model_params[param_name] = options_list[0] if options_list else None

            # Instantiate the final model with correctly mapped parameters
            if name == "LogisticRegression":
                solver = 'liblinear'
                if 'penalty' in final_model_params and final_model_params['penalty'] == 'elasticnet':
                    solver = 'saga'
                final_model = LogisticRegression(random_state=42, solver=solver, **final_model_params)
            elif name == "DecisionTree":
                final_model = DecisionTreeClassifier(random_state=42, **final_model_params)
            elif name == "MLPClassifier":
                final_model = MLPClassifier(random_state=42, max_iter=500, **final_model_params)
            elif name == "RandomForest":
                final_model = RandomForestClassifier(random_state=42, **final_model_params)
            elif name == "SVC":
                final_model = SVC(random_state=42, probability=True, **final_model_params)
            elif name == "XGBoost":
                final_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', **final_model_params)

            print(f"Best hyperparameters for {name}: {final_model_params}")
            final_models[name] = final_model

        # Evaluate the optimized models
        for name, model in final_models.items():
            print(f"\n===== Evaluating {name} =====") # 修改了标题，去掉了"Optimized"
            all_models_report.append(f"===== {name} =====") # 修改了报告标题

            # --- Cross-validation part ---
            cv_results = cross_validate(model, X_scaled, y, cv=skf, scoring=['accuracy', 'f1_weighted', 'f1_macro'], error_score='raise')
            mean_cv_accuracy = np.mean(cv_results['test_accuracy'])
            mean_cv_f1_weighted = np.mean(cv_results['test_f1_weighted'])
            mean_cv_f1_macro = np.mean(cv_results['test_f1_macro'])

            print(f"Cross-validation (5-fold) accuracy scores: {cv_results['test_accuracy']}")
            print(f"Mean cross-validation accuracy: {mean_cv_accuracy:.4f}")
            print(f"Cross-validation (5-fold) F1-weighted scores: {cv_results['test_f1_weighted']}")
            print(f"Mean cross-validation F1-weighted: {mean_cv_f1_weighted:.4f}")
            print(f"Cross-validation (5-fold) F1-macro scores: {cv_results['test_f1_macro']}")
            print(f"Mean cross-validation F1-macro: {mean_cv_f1_macro:.4f}\n")

            all_models_report.append(f"Cross-validation (5-fold) accuracy scores: {cv_results['test_accuracy']}")
            all_models_report.append(f"Mean cross-validation accuracy: {mean_cv_accuracy:.4f}")
            all_models_report.append(f"Cross-validation (5-fold) F1-weighted scores: {cv_results['test_f1_weighted']}")
            all_models_report.append(f"Mean cross-validation F1-weighted: {mean_cv_f1_weighted:.4f}")
            all_models_report.append(f"Cross-validation (5-fold) F1-macro scores: {cv_results['test_f1_macro']}")
            all_models_report.append(f"Mean cross-validation F1-macro: {mean_cv_f1_macro:.4f}\n")

            # 存储性能指标
            model_performance_metrics[name] = {
                'accuracy': mean_cv_accuracy,
                'f1_weighted': mean_cv_f1_weighted,
                'f1_macro': mean_cv_f1_macro
            }

            # --- Original test set evaluation part ---
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            class_report = classification_report(y_test, y_pred, target_names=unique_labels)
            conf_matrix = confusion_matrix(y_test, y_pred)

            print(f"Test set accuracy: {accuracy:.4f}")
            print(f"Test set F1-score (weighted): {f1:.4f}\n")
            all_models_report.append(f"Test set accuracy: {accuracy:.4f}")
            all_models_report.append(f"Test set F1-score (weighted): {f1:.4f}\n")

            print("Classification Report:")
            print(class_report)
            all_models_report.append("Classification Report:")
            all_models_report.append(class_report)

            print("\nConfusion Matrix:")
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                             xticklabels=unique_labels, yticklabels=unique_labels,
                             annot_kws={"size": 14})

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)

            plt.xlabel('Predicted Label', fontname='Arial', fontsize=22)
            plt.ylabel('True Label', fontname='Arial', fontsize=22)
            plt.xticks(fontsize=18, rotation=45, ha='right')
            plt.yticks(fontsize=18, rotation=0)
            plt.title(f'{name} Confusion Matrix', fontsize=24) # 修改了标题，去掉了"Optimized"
            plt.tight_layout()
            plt.show()
            all_models_report.append("Confusion Matrix:\n" + str(conf_matrix) + "\n\n")

        # --- 模型融合操作 ---
        print("\n===== Performing Ensemble Learning =====")
        ensemble_models_info = {}

        # 1. VotingClassifier (Soft Voting)
        # 确保基模型已经优化过，并且支持 predict_proba
        # 我们使用之前优化好的 SVC, RandomForest, MLPClassifier
        estimators_for_voting = [
            ('svc', final_models['SVC']),
            ('rf', final_models['RandomForest']),
            ('mlp', final_models['MLPClassifier'])
        ]
        voting_clf = VotingClassifier(estimators=estimators_for_voting, voting='soft', n_jobs=-1) # n_jobs=-1 使用所有核心
        ensemble_models_info['VotingClassifier'] = {
            'model': voting_clf,
            'params': {
                'estimators': [e[0] for e in estimators_for_voting],
                'voting': 'soft'
            }
        }
        print("\nVotingClassifier configured:")
        print(f"  Estimators: {[e[0] for e in estimators_for_voting]}")
        print(f"  Voting: soft")

        # 2. StackingClassifier
        # 基模型同上，元学习器使用 LogisticRegression
        estimators_for_stacking = [
            ('svc', final_models['SVC']),
            ('rf', final_models['RandomForest']),
            ('mlp', final_models['MLPClassifier'])
        ]
        stacking_clf = StackingClassifier(estimators=estimators_for_stacking,
                                          final_estimator=LogisticRegression(random_state=42, solver='liblinear'), # 使用逻辑回归作为元学习器
                                          cv=skf, # 指定用于元学习器训练的交叉验证策略
                                          n_jobs=-1 # 使用所有核心
                                         )
        ensemble_models_info['StackingClassifier'] = {
            'model': stacking_clf,
            'params': {
                'base_estimators': [e[0] for e in estimators_for_stacking],
                'final_estimator': 'LogisticRegression'
            }
        }
        print("\nStackingClassifier configured:")
        print(f"  Base Estimators: {[e[0] for e in estimators_for_stacking]}")
        print(f"  Final Estimator: LogisticRegression")

        # 评估融合模型
        for ensemble_name, info in ensemble_models_info.items():
            model = info['model']
            params = info['params']
            print(f"\n===== Evaluating Ensemble: {ensemble_name} =====")
            all_models_report.append(f"===== Ensemble: {ensemble_name} =====")
            all_models_report.append(f"Parameters: {params}")

            cv_results = cross_validate(model, X_scaled, y, cv=skf,
                                        scoring=['accuracy', 'f1_weighted', 'f1_macro'],
                                        error_score='raise', n_jobs=-1) # 交叉验证也使用多核

            mean_cv_accuracy = np.mean(cv_results['test_accuracy'])
            mean_cv_f1_weighted = np.mean(cv_results['test_f1_weighted'])
            mean_cv_f1_macro = np.mean(cv_results['test_f1_macro'])

            print(f"Cross-validation (5-fold) accuracy scores: {cv_results['test_accuracy']}")
            print(f"Mean cross-validation accuracy: {mean_cv_accuracy:.4f}")
            print(f"Cross-validation (5-fold) F1-weighted scores: {cv_results['test_f1_weighted']}")
            print(f"Mean cross-validation F1-weighted: {mean_cv_f1_weighted:.4f}")
            print(f"Cross-validation (5-fold) F1-macro scores: {cv_results['test_f1_macro']}")
            print(f"Mean cross-validation F1-macro: {mean_cv_f1_macro:.4f}\n")

            all_models_report.append(f"Cross-validation (5-fold) accuracy scores: {cv_results['test_accuracy']}")
            all_models_report.append(f"Mean cross-validation accuracy: {mean_cv_accuracy:.4f}")
            all_models_report.append(f"Cross-validation (5-fold) F1-weighted scores: {cv_results['test_f1_weighted']}")
            all_models_report.append(f"Mean cross-validation F1-weighted: {mean_cv_f1_weighted:.4f}")
            all_models_report.append(f"Cross-validation (5-fold) F1-macro scores: {cv_results['test_f1_macro']}")
            all_models_report.append(f"Mean cross-validation F1-macro: {mean_cv_f1_macro:.4f}\n")

            # 存储融合模型性能指标
            model_performance_metrics[ensemble_name] = {
                'accuracy': mean_cv_accuracy,
                'f1_weighted': mean_cv_f1_weighted,
                'f1_macro': mean_cv_f1_macro
            }

        # --- 性能对比柱状图 ---
        print("\n===== Generating Performance Comparison Plot =====")
        model_names = list(model_performance_metrics.keys())
        accuracies = [model_performance_metrics[name]['accuracy'] for name in model_names]
        f1_weighted_scores = [model_performance_metrics[name]['f1_weighted'] for name in model_names]
        f1_macro_scores = [model_performance_metrics[name]['f1_macro'] for name in model_names]


        # 创建一个 DataFrame 方便绘图
        plot_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'F1-Weighted': f1_weighted_scores,
            'F1-Macro': f1_macro_scores
        })

        # 将数据转换为长格式，方便用 seaborn 绘制多指标柱状图
        plot_df_long = plot_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x='Score', y='Model', hue='Metric', data=plot_df_long, palette='viridis')

        plt.xlabel('Score', fontname='Arial', fontsize=20)
        plt.ylabel('Model', fontname='Arial', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Model Performance Comparison (5-Fold Cross-Validation)', fontsize=24, fontweight='bold')
        plt.legend(title='Metric', fontsize=14, title_fontsize=16)

        # 在柱状图上添加数值标签
        for container in barplot.containers:
            barplot.bar_label(container, fmt='%.3f', fontsize=12, padding=3)

        plt.tight_layout()
        plt.show()

        # --- Dimensionality Reduction calculations BEFORE plotting ---
        n_components_pca = 2
        pca = PCA(n_components=n_components_pca)
        X_pca = pca.fit_transform(X_scaled)

        n_components_tsne = 2
        # 注意：这里使用 n_iter 参数，根据之前的验证结果
        tsne = TSNE(n_components=n_components_tsne, random_state=42, perplexity=30, n_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)

        # --- Function to plot PCA/t-SNE consistently using gridspec ---
        def plot_dimensionality_reduction_with_gridspec(X_transformed, y_labels, title, filename_suffix, save_dir,
                                                        unique_labels):
            fig_height = 8
            fig_width = fig_height * 1.3

            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.2)

            ax = fig.add_subplot(gs[:, 0])
            scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_labels, cmap='viridis', s=50)

            x_min, x_max = X_transformed[:, 0].min(), X_transformed[:, 0].max()
            y_min, y_max = X_transformed[:, 1].min(), X_transformed[:, 1].max()

            range_x = x_max - x_min
            range_y = y_max - y_min

            max_range = max(range_x, range_y)
            padding = max_range * 0.1

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            ax.set_xlim(center_x - (max_range / 2) - padding, center_x + (max_range / 2) + padding)
            ax.set_ylim(center_y - (max_range / 2) - padding, center_y + (max_range / 2) + padding)

            ax.set_aspect('equal', adjustable='box')

            ax.set_xlabel(f'{title} Component 1', fontname='Arial', fontsize=22)
            ax.set_ylabel(f'{title} Component 2', fontname='Arial', fontsize=22)
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            plt.title(f'{title}', fontsize=24, fontweight='bold', fontname='Arial')

            legend_ax = fig.add_subplot(gs[:, 1])
            legend_ax.axis("off")

            handles, labels_ = scatter.legend_elements()

            legend_ax.legend(
                handles,
                unique_labels,
                loc="center left",
                bbox_to_anchor=(0.0, 0.5),
                fontsize=16,
                frameon=False,
                ncol=1,
                title="Amino Acid (AA)",
                title_fontsize=16
            )

            plt.tight_layout()
            plt.show()
            save_scatter_data(X_transformed, original_labels_for_viz, f'{filename_suffix}_data.txt', save_dir)

        # Call the plotting function for PCA using gridspec
        plot_dimensionality_reduction_with_gridspec(X_pca, y, 'PCA', 'pca', save_base_dir, unique_labels)

        # Corrected function call: plot_dimensionality_reduction_with_gridspec
        plot_dimensionality_reduction_with_gridspec(X_tsne, y, 't-SNE', 'tsne', save_base_dir, unique_labels)

        # Feature Importance Analysis (for RandomForest and XGBoost)
        feature_importance_data = {}

        # The feature names are now just 'intensity' and 'shift'
        current_feature_names = feature_cols_final

        if "RandomForest" in final_models and hasattr(final_models["RandomForest"], 'feature_importances_'):
            model_rf = final_models["RandomForest"]
            importances = model_rf.feature_importances_
            # Ensure the number of feature importances matches the number of feature names
            if len(importances) == len(current_feature_names):
                feature_importance_data["RandomForest"] = pd.DataFrame(
                    {'feature': current_feature_names, 'importance': importances}) \
                    .sort_values(by='importance', ascending=False)
            else:
                print(f"Warning: Number of feature importances ({len(importances)}) does not match number of feature names ({len(current_feature_names)}) for RandomForest. Skipping importance plot.")


        if "XGBoost" in final_models and hasattr(final_models["XGBoost"], 'feature_importances_'):
            model_xgb = final_models["XGBoost"]
            importances = model_xgb.feature_importances_
            # Ensure the number of feature importances matches the number of feature names
            if len(importances) == len(current_feature_names):
                feature_importance_data["XGBoost"] = pd.DataFrame({'feature': current_feature_names, 'importance': importances}) \
                    .sort_values(by='importance', ascending=False)
            else:
                print(f"Warning: Number of feature importances ({len(importances)}) does not match number of feature names ({len(current_feature_names)}) for XGBoost. Skipping importance plot.")


        # Plot and save feature importance
        for model_name, df_imp in feature_importance_data.items():
            print(f"\n===== {model_name} Feature Importance =====")
            print(df_imp)
            plt.figure(figsize=(10, 8))
            plt.barh(df_imp['feature'], df_imp['importance'])
            plt.xlabel('Feature Importance', fontname='Arial', fontsize=22)
            plt.ylabel('Feature Name', fontname='Arial', fontsize=22)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.title(f'{model_name} Feature Importance', fontsize=24) # 修改了标题，去掉了"Optimized"
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            save_feature_importance(df_imp, f'{model_name}_feature_importance.txt', save_base_dir)

        # Save all model evaluation reports
        if save_base_dir:
            try:
                model_eval_path = os.path.join(save_base_dir, "model_evaluation.txt")
                with open(model_eval_path, 'w') as f:
                    f.write("All Model Evaluation Results (including Cross-validation and Test Set Evaluation):\n\n")
                    for line in all_models_report:
                        f.write(line + "\n")
                messagebox.showinfo("Success", f"All model evaluation results saved to:\n{model_eval_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving model evaluation results: {e}")
        else:
            messagebox.showinfo("Info", "No save path selected, model evaluation results not saved.")

    else:
        messagebox.showinfo("Info", "No training data file selected, program exited.")


if __name__ == "__main__":
    main()