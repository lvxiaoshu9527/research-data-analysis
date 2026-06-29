import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.patches import Ellipse
import os
import sys
import itertools
import random
from scipy.stats import bootstrap
from typing import Tuple, List, Dict, Any, Optional
from warnings import filterwarnings
import json
import joblib

# Suppress all warnings for cleaner output
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=DeprecationWarning)

# --- Tkinter import/fallback logic ---
# Try importing tkinter; if it fails, fall back to console input.
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TKINTER_AVAILABLE = True
except ImportError:
    print("Tkinter is not available. Falling back to console input for file paths and amino acid selection.")
    TKINTER_AVAILABLE = False


    class MockTkinter:  # Simulate tkinter classes to prevent errors when tkinter is unavailable
        def Tk(self): return self

        def withdraw(self): pass

        def filedialog(self): return self

        def askopenfilename(self, *args, **kwargs): return input("Enter path to data CSV file: ")

        def askdirectory(self, *args, **kwargs): return input("Enter directory to save results: ")

        def messagebox(self): return self

        def showinfo(self, *args, **kwargs): print(f"INFO: {args[1]}")

        def showerror(self, *args, **kwargs): print(f"ERROR: {args[1]}")


    # Assign MockTkinter instances if Tkinter is not available
    tk = MockTkinter()
    filedialog = tk.filedialog()
    messagebox = tk.messagebox()

# Set Matplotlib default font to support English, and ensure Arial and other English fonts are available
mpl.rcParams['font.family'] = ['Arial', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


# --- User-provided file selection and save functions ---
def select_file() -> Optional[str]:
    """Opens a file selection dialog to choose the raw data CSV file, or prompts via console."""
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()  # Hide the main root window
        file_path = filedialog.askopenfilename(
            title="选择原始数据 CSV 文件",  # Select Raw Data CSV File
            filetypes=[("CSV files", "*.csv")]  # CSV 文件
        )
        root.destroy()  # Destroy the hidden root after dialog closes
    else:
        file_path = input(
            "请输入原始数据 CSV 文件的完整路径: ")  # Please enter the full path to your raw data CSV file:
    return file_path


def get_save_directory() -> Optional[str]:
    """Opens a folder selection dialog for the user to choose a directory for saving files, or prompts via console."""
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()  # Hide the main root window
        save_dir = filedialog.askdirectory(
            title="选择保存结果的目录",  # Select Directory to Save Results
        )
        root.destroy()  # Destroy the hidden root after dialog closes
    else:
        save_dir = input(
            "请输入保存结果的目录完整路径: ")  # Please enter the full path to the directory where results should be saved:
    return save_dir


def save_json_results(data: Dict[str, Any], filename: str, save_dir: Optional[str]):
    """Saves results to a JSON file."""
    if save_dir:
        file_path = os.path.join(save_dir, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Results saved to JSON: {file_path}")
        except Exception as e:
            print(f"Error saving JSON to {file_path}: {e}")
    else:
        print("No save path selected, JSON results not saved.")


# --- Data Preparation & Preprocessing Functions ---
def preprocess_data(df: pd.DataFrame, feature_cols: List[str], label_col: str = 'AA', missing_threshold: float = 0.2) -> \
        Tuple[pd.DataFrame, pd.Series, List[str], StandardScaler]:
    """
    Preprocesses the data: handles missing values, standardizes features, and encodes labels.
    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str], StandardScaler]: Scaled features, encoded labels, original unique label names, and the fitted StandardScaler.
    """
    required_cols = [label_col] + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in the CSV: {', '.join(missing_cols)}")

    df.dropna(subset=[label_col], inplace=True)
    df_features_numeric = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df_filtered_missing = df[df_features_numeric.isnull().sum(axis=1) / len(feature_cols) <= missing_threshold]
    rows_dropped_missing = df.shape[0] - df_filtered_missing.shape[0]
    if rows_dropped_missing > 0:
        print(f"Dropped {rows_dropped_missing} samples due to more than {missing_threshold * 100}% missing features.")
    df = df_filtered_missing.copy()

    if df.empty:
        raise ValueError("DataFrame is empty after dropping missing values.")

    X = df[feature_cols].copy()
    y_original_labels = df[label_col].copy()

    X.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y_original_labels), index=y_original_labels.index)
    unique_original_labels = list(label_encoder.classes_)

    if len(unique_original_labels) < 2:
        raise ValueError("Processed data contains fewer than 2 unique amino acids. Cannot perform clustering.")

    return X_scaled, y_encoded, unique_original_labels, scaler


def remap_labels_for_subset(y_encoded: pd.Series, unique_original_labels: List[str],
                            amino_acid_subset_names: List[str]) -> Tuple[pd.Series, List[str]]:
    """Remaps labels for a given amino acid subset."""
    subset_encoded_values = [
        unique_original_labels.index(aa_name)
        for aa_name in amino_acid_subset_names
        if aa_name in unique_original_labels
    ]
    filtered_indices = y_encoded[y_encoded.isin(subset_encoded_values)].index
    y_filtered_encoded = y_encoded.loc[filtered_indices]
    actual_subset_original_labels = [unique_original_labels[val] for val in sorted(y_filtered_encoded.unique())]
    new_label_mapping = {old_val: new_val for new_val, old_val in enumerate(sorted(y_filtered_encoded.unique()))}
    y_remapped = y_filtered_encoded.map(new_label_mapping)
    return y_remapped, actual_subset_original_labels


def engineer_features(df_original_features: pd.DataFrame, chiralities: List[str]) -> pd.DataFrame:
    """
    Generates new features based on original intensity and shift data.
    """
    df_engineered = df_original_features.copy()

    for i in range(len(chiralities)):
        for j in range(i + 1, len(chiralities)):
            c1, c2 = chiralities[i], chiralities[j]
            col1 = f'{c1}_intensity'
            col2 = f'{c2}_intensity'
            if col1 in df_engineered.columns and col2 in df_engineered.columns:
                df_engineered[f'{c1}I_div_{c2}I'] = df_engineered[col1] / df_engineered[col2].replace(0, np.nan)
                df_engineered[f'{c2}I_div_{c1}I'] = df_engineered[col2] / df_engineered[col1].replace(0, np.nan)

    for i in range(len(chiralities)):
        for j in range(i + 1, len(chiralities)):
            c1, c2 = chiralities[i], chiralities[j]
            col1 = f'{c1}_shift'
            col2 = f'{c2}_shift'
            if col1 in df_engineered.columns and col2 in df_engineered.columns:
                df_engineered[f'{c1}S_minus_{c2}S'] = df_engineered[col1] - df_engineered[col2]

    for c in chiralities:
        intensity_col = f'{c}_intensity'
        shift_col = f'{c}_shift'
        if intensity_col in df_engineered.columns and shift_col in df_engineered.columns:
            df_engineered[f'{c}I_div_{c}S'] = df_engineered[intensity_col] / df_engineered[shift_col].replace(0, np.nan)

    intensity_cols = [f'{c}_intensity' for c in chiralities if f'{c}_intensity' in df_engineered.columns]
    if intensity_cols:
        max_intensity_per_row = df_engineered[intensity_cols].max(axis=1)
        for col in intensity_cols:
            df_engineered[f'{col}_norm'] = df_engineered[col] / max_intensity_per_row.replace(0, np.nan)

    df_engineered.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_engineered_scaled = pd.DataFrame(scaler.fit_transform(df_engineered), columns=df_engineered.columns,
                                       index=df_engineered.index)

    print(f"Engineered features added. Total features: {X_engineered_scaled.shape[1]}")
    return X_engineered_scaled


def cluster_with_method(
        X: pd.DataFrame,
        y_true: pd.Series,
        method: str,
        n_clusters: int,
        random_state_val: int,
        **kwargs
) -> Tuple[Optional[np.ndarray], Any, Any, Optional[
    pd.DataFrame]]:  # Returns cluster_labels, clustering_model, dim_reduction_model, X_transformed (LDA output)
    """
    Unified clustering interface, returns cluster_labels, clustering_model_instance, dim_reduction_model_instance, and transformed X (if LDA).
    """
    cluster_labels = None
    clustering_model = None
    dim_reduction_model = None
    X_transformed = None

    if X.shape[0] == 0: return None, None, None, None
    if n_clusters <= 0: return None, None, None, None
    if X.shape[0] < n_clusters: return None, None, None, None

    try:
        if method == 'LDA+KMeans':
            unique_true_labels = y_true.unique()
            if len(unique_true_labels) < 2:
                print("LDA+KMeans Error: Fewer than 2 unique true labels, cannot perform LDA dimensionality reduction.")
                return None, None, None, None

            # LDA n_components should be min(n_clusters - 1, num_classes - 1, num_features)
            # Since we expect n_clusters to be the same as num_classes, it simplifies to num_classes - 1
            n_components_lda = min(n_clusters - 1, len(unique_true_labels) - 1, X.shape[1])

            if n_components_lda <= 0:
                print("LDA dimensionality reduction will result in non-positive dimensions, skipping LDA+KMeans.")
                return None, None, None, None

            try:
                lda = LDA(n_components=n_components_lda, **kwargs)
                X_lda = lda.fit_transform(X, y_true)
                dim_reduction_model = lda  # Capture the fitted LDA model
                X_transformed = pd.DataFrame(X_lda, index=X.index)  # Store LDA projected data
            except Exception as e:
                print(f"LDA failed: {e}. Skipping LDA+KMeans.")
                return None, None, None, None

            clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state_val, n_init='auto', **kwargs)
            cluster_labels = clustering_model.fit_predict(X_lda)
        else:
            raise ValueError(f"Unsupported clustering method: {method}. Only 'LDA+KMeans' is supported.")
    except Exception as e:
        print(f"Clustering method {method} failed: {e}. Returning None labels and models.")
        return None, None, None, None

    return cluster_labels, clustering_model, dim_reduction_model, X_transformed


# --- Evaluation Metrics Functions ---
def evaluate_clustering(
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        combo_name: Tuple[str, ...],
        original_subset_labels_names: List[str]
) -> Dict[str, Any]:
    """Evaluates clustering results using ARI and Confusion Matrix."""
    results = {'combination': combo_name}

    if len(y_true.unique()) > 1 and len(np.unique(y_pred[y_pred != -1])) > 1:
        results['ARI'] = adjusted_rand_score(y_true, y_pred)
    else:
        results['ARI'] = 0.0

    results['Confusion_Matrix'] = confusion_matrix(y_true, y_pred)
    return results


def calculate_purity_metrics(y_true: pd.Series, y_pred: np.ndarray, n_bootstraps: int = 1000) -> Tuple[
    float, float, float]:
    """
    Calculates cluster purity and its 95% confidence interval using bootstrapping.

    Purity is defined as the sum of maximum proportions for each cluster, divided by total samples.
    It indicates how "pure" each cluster is in terms of containing mostly samples from a single true class.

    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted cluster labels.
        n_bootstraps (int): Number of bootstrap samples.

    Returns:
        Tuple[float, float, float]: (mean_purity, lower_ci, upper_ci)
    """

    # Filter out noise points if any (-1 labels)
    valid_indices = y_pred != -1
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]

    if y_true_filtered.empty or len(np.unique(y_pred_filtered)) == 0:
        return 0.0, 0.0, 0.0  # Return 0 purity if no valid points or clusters

    def get_purity(y_true_sample, y_pred_sample):
        if y_true_sample.empty or len(np.unique(y_pred_sample)) == 0:
            return 0.0
        contingency_table = pd.crosstab(y_pred_sample, y_true_sample)
        # Sum of maximum counts in each row (cluster)
        max_counts_per_cluster = contingency_table.max(axis=1).sum()
        total_samples = len(y_true_sample)
        return max_counts_per_cluster / total_samples

    # Use bootstrap to estimate confidence interval
    # The 'bootstrap' function from scipy.stats expects a single array-like input
    # and a statistic function that takes that array-like input.
    # Here, we need to resample (y_true, y_pred) pairs.
    # So, we'll create a combined array of indices to resample from.
    data_indices = np.arange(len(y_true_filtered))

    def purity_statistic(indices_for_bootstrap_sample):
        y_true_bs = y_true_filtered.iloc[indices_for_bootstrap_sample]
        y_pred_bs = y_pred_filtered[indices_for_bootstrap_sample]
        return get_purity(y_true_bs, y_pred_bs)

    try:
        # resample data_indices to get bootstrap samples
        res = bootstrap((data_indices,), purity_statistic, n_resamples=n_bootstraps,
                        confidence_level=0.95, random_state=42, method='percentile')

        mean_purity = np.mean(res.bootstrap_distribution)  # Get mean from the distribution
        lower_ci = res.confidence_interval.low
        upper_ci = res.confidence_interval.high
    except Exception as e:
        print(f"Warning: Bootstrap calculation for purity failed: {e}. Falling back to 0.0 CI.")
        # If bootstrap fails (e.g., too few samples or issues with statistic),
        # calculate a single purity score and return 0.0 for CI.
        mean_purity = get_purity(y_true_filtered, y_pred_filtered)
        lower_ci = mean_purity
        upper_ci = mean_purity  # No confidence interval if bootstrap failed

    return mean_purity, lower_ci, upper_ci


# --- Visualization Output Functions ---
def plot_dual_encoded_clusters(
        X_transformed: pd.DataFrame,  # Now directly takes the 2D transformed data
        y_true_encoded: pd.Series,
        y_pred: np.ndarray,
        combo_name: Tuple[str, ...],
        filename: str,
        true_label_names: List[str],
        ari_score: Optional[float] = None,
        purity_score_display: Optional[str] = None,  # For formatted purity string
        method_name: Optional[str] = None,
        k_value: Optional[int] = None
):
    """
    Plots a 2D scatter plot of the clustered data, showing true labels and predicted clusters.
    Assumes X_transformed is already 2D (from LDA).
    Adds confidence ellipses for each predicted cluster.
    """
    if X_transformed.empty or y_true_encoded.empty or y_pred is None or len(np.unique(y_pred[y_pred != -1])) < 2:
        print(f"Warning: Insufficient data for 2D cluster plot for {combo_name}. Skipping.")
        return

    df_plot_filtered = X_transformed.copy()
    df_plot_filtered['True_Label_Encoded'] = y_true_encoded
    df_plot_filtered['Predicted_Cluster'] = y_pred

    # Filter out noise points for plotting and evaluation if method produces them
    df_plot_filtered = df_plot_filtered[df_plot_filtered['Predicted_Cluster'] != -1].copy()
    if df_plot_filtered.empty:
        print(f"Warning: No non-noise points available for 2D cluster plot for {combo_name}. Skipping.")
        return

    # Map encoded true labels back to original names for plotting
    df_plot_filtered['True_Label_Name'] = df_plot_filtered['True_Label_Encoded'].map(
        {i: name for i, name in enumerate(true_label_names)}
    )

    df_plot_filtered.columns = ['Dim1', 'Dim2', 'True_Label_Encoded', 'Predicted_Cluster', 'True_Label_Name']

    plt.figure(figsize=(12, 10))
    # Use True_Label_Name for hue (color) and Predicted_Cluster for style (marker/shape)
    sns.scatterplot(
        data=df_plot_filtered,
        x='Dim1',
        y='Dim2',
        hue='True_Label_Name',
        style='Predicted_Cluster',
        palette='tab20',  # Use a diverse palette for true labels
        markers=True,  # Allow different markers for predicted clusters
        s=100,  # Size of points
        alpha=0.8,
        edgecolor='w',  # White border for points
        linewidth=0.5
    )

    # Add confidence ellipses for each predicted cluster
    ax = plt.gca()
    for cluster_id in sorted(df_plot_filtered['Predicted_Cluster'].unique()):
        cluster_data = df_plot_filtered[df_plot_filtered['Predicted_Cluster'] == cluster_id]
        if len(cluster_data) > 1:  # Need at least 2 points for covariance
            # Calculate covariance matrix
            cov = np.cov(cluster_data[['Dim1', 'Dim2']].values.T)
            # Calculate mean of the cluster
            mean = cluster_data[['Dim1', 'Dim2']].mean().values

            try:
                # Get eigenvalues and eigenvectors
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Sort eigenvalues and eigenvectors
                order = eigvals.argsort()[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]

                # Get angle of the largest eigenvector
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

                # Width and height of ellipse based on 2 standard deviations (95% confidence)
                width, height = 2 * np.sqrt(5.991 * eigvals)  # Chi-squared for 2 DOF, 95% CI is ~5.991

                # Create ellipse patch
                ellipse = Ellipse(xy=mean, width=width[0], height=height[1], angle=angle,
                                  color=sns.color_palette('pastel')[cluster_id % len(sns.color_palette('pastel'))],
                                  alpha=0.2, fill=True, zorder=0, label=f'Cluster {cluster_id} Ellipse')
                ax.add_patch(ellipse)
            except np.linalg.LinAlgError:
                print(f"  Warning: Could not draw ellipse for Cluster {cluster_id} (singular matrix). Skipping.")
            except Exception as e:
                print(f"  Error drawing ellipse for Cluster {cluster_id}: {e}. Skipping.")

    plt.title(f"2D Cluster Visualization for {'_'.join(combo_name)} (Method: {method_name})\n"
              f"k={k_value}, Dim Reduction: LDA" +
              (f", ARI: {ari_score:.2f}" if ari_score is not None else "") +
              (f", Purity: {purity_score_display}" if purity_score_display is not None else ""),
              fontsize=14)
    plt.xlabel(f"LDA Dim1", fontsize=12)
    plt.ylabel(f"LDA Dim2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Custom legend for both hue and style
    handles, labels = ax.get_legend_handles_labels()
    # Filter out duplicate labels for style (e.g., predicted clusters) if seaborn adds them automatically for hue
    unique_labels_map = {}
    for h, l in zip(handles, labels):
        unique_labels_map[l] = h

    final_handles = list(unique_labels_map.values())
    final_labels = list(unique_labels_map.keys())

    plt.legend(final_handles, final_labels, title="True AA / Predicted Cluster", bbox_to_anchor=(1.05, 1),
               loc='upper left', ncol=1, fontsize=9, title_fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()
    print(f"2D cluster plot saved to: {filename}")


def plot_cluster_purity_matrix(
        y_true_encoded: pd.Series,
        y_pred: np.ndarray,
        combo_name: Tuple[str, ...],
        filename: str,
        true_label_names: List[str],
        method_name: Optional[str] = None,
        ari_score: Optional[float] = None,
        purity_score_display: Optional[str] = None,  # For formatted purity string
        k_value: Optional[int] = None
):
    """
    Generates a heatmap showing the purity of each predicted cluster with respect to true amino acid labels.
    """
    if y_true_encoded.empty or y_pred is None or len(y_pred[y_pred != -1]) < 2:
        print(f"Warning: Insufficient data for cluster purity matrix for {combo_name}. Skipping.")
        return

    df_for_purity = pd.DataFrame({'True_AA_Encoded': y_true_encoded, 'Predicted_Cluster': y_pred})
    df_for_purity = df_for_purity[df_for_purity['Predicted_Cluster'] != -1]

    if df_for_purity.empty:
        print(f"Warning: No non-noise samples for cluster purity matrix for {combo_name}. Skipping.")
        return

    contingency_table = pd.crosstab(df_for_purity['Predicted_Cluster'], df_for_purity['True_AA_Encoded'])

    all_true_aa_encoded = sorted(y_true_encoded.unique())
    all_predicted_clusters = sorted(df_for_purity['Predicted_Cluster'].unique())

    purity_matrix_raw = pd.DataFrame(0, index=all_predicted_clusters, columns=all_true_aa_encoded)

    for cluster_id in contingency_table.index:
        for aa_encoded in contingency_table.columns:
            purity_matrix_raw.loc[cluster_id, aa_encoded] = contingency_table.loc[cluster_id, aa_encoded]

    purity_matrix_normalized = purity_matrix_raw.div(purity_matrix_raw.sum(axis=1), axis=0).fillna(0)

    purity_matrix_normalized.columns = [true_label_names[idx] for idx in purity_matrix_normalized.columns]
    purity_matrix_normalized.index = [f"Cluster {int(c)}" for c in purity_matrix_normalized.index]

    plt.figure(figsize=(10, 8))
    sns.heatmap(purity_matrix_normalized, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=.5, linecolor='lightgray',
                cbar_kws={'label': 'Proportion of Amino Acid in Cluster'})

    plot_title = f"Cluster Purity Matrix for {'_'.join(combo_name)}\n" \
                 f"Method: {method_name}, k={k_value}"
    if ari_score is not None: plot_title += f", ARI: {ari_score:.2f}"
    if purity_score_display is not None: plot_title += f", Purity: {purity_score_display}"

    plt.title(plot_title, fontsize=14)
    plt.xlabel("True Amino Acid Label", fontsize=12)
    plt.ylabel("Predicted Cluster Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()


def select_amino_acid_subset(amino_acid_list: List[str]) -> List[str]:
    """
    Pops up a Tkinter multi-selection window to let the user select amino acids.
    If Tkinter is not available, falls back to console input.
    """
    selected_amino_acids = []

    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()  # Hide the main root window

        selection_window = tk.Toplevel(root)
        selection_window.title("选择氨基酸子集进行聚类")  # Select Amino Acid Subset for Clustering
        selection_window.geometry("400x600")

        # --- IMPORTANT: Force window update here ---
        # This helps ensure the window is rendered before grabbing focus or waiting.
        selection_window.update_idletasks()
        selection_window.update()
        # --- End of IMPORTANT section ---

        # Create a frame for the canvas and scrollbar
        frame_canvas = ttk.Frame(selection_window)
        frame_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(frame_canvas)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        check_vars = {}
        for aa in sorted(amino_acid_list):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(inner_frame, text=aa, variable=var)
            cb.pack(anchor="w", padx=5, pady=2)
            check_vars[aa] = var

        def select_all():
            for var in check_vars.values():
                var.set(True)

        def deselect_all():
            for var in check_vars.values():
                var.set(False)

        def on_ok():
            nonlocal selected_amino_acids
            selected_amino_acids = [aa for aa, var in check_vars.items() if var.get()]
            selection_window.destroy()

        # Buttons
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(fill="x", pady=10)

        select_all_btn = ttk.Button(button_frame, text="全选", command=select_all)  # Select All
        select_all_btn.pack(side="left", padx=5)

        deselect_all_btn = ttk.Button(button_frame, text="全不选", command=deselect_all)  # Deselect All
        deselect_all_btn.pack(side="left", padx=5)

        ok_btn = ttk.Button(button_frame, text="确定", command=on_ok)  # OK
        ok_btn.pack(side="right", padx=5)

        # Make the selection window modal and wait for it to close
        selection_window.transient(root)  # Make it a transient window of the root (which is withdrawn)
        selection_window.grab_set()  # Grab focus
        root.wait_window(selection_window)  # Wait for the selection window to close
        root.destroy()  # Destroy the hidden root after the selection window is closed
    else:
        print("\nTkinter is not available. Please enter desired amino acids as a comma-separated list.")
        print(f"Available amino acids: {', '.join(sorted(amino_acid_list))}")
        user_input = input("Enter amino acids (e.g., ALA,GLY,LEU): ")
        selected_amino_acids = [aa.strip().upper() for aa in user_input.split(',') if
                                aa.strip().upper() in amino_acid_list]
        if not selected_amino_acids:
            print("No valid amino acids selected. Proceeding with an empty list.")

    return selected_amino_acids


# --- Main Logic Function ---
def main():
    """
    Controls the overall workflow: data loading, preprocessing, user-selected subset,
    LDA+KMeans clustering, evaluation, and results visualization.
    """
    print("--- DNA-Wrapped SWNT Array Amino Acid Discrimination System (LDA+KMeans Only) ---")

    # --- Setup Directories ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # --- Load Data ---
    input_file_path = select_file()
    if not input_file_path:
        messagebox.showinfo("Info", "No raw data file selected. Exiting program.")
        return

    try:
        df_original = pd.read_csv(input_file_path)
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {input_file_path}")
        return
    except Exception as e:
        messagebox.showerror("Error", f"Error reading file: {e}")
        return

    print("Original Data Info:")
    print(df_original.info())
    print("\nFirst 5 Rows of Original Data:")
    print(df_original.head())

    # Define original feature and label columns based on typical structure
    chiralities = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']
    original_feature_columns = [f'{c}_intensity' for c in chiralities] + [f'{c}_shift' for c in chiralities]
    label_column = 'AA'

    # Preprocess the entire dataset (initial scaling of original features)
    try:
        X_original_scaled, y_full_encoded, all_amino_acids_names, global_scaler_for_X = preprocess_data(
            df_original.copy(), original_feature_columns, label_column, missing_threshold=0.2)
    except ValueError as e:
        messagebox.showerror("Data Preprocessing Error", str(e))
        return

    print(f"\nData loaded and initial preprocessing complete. Found {len(all_amino_acids_names)} unique amino acids.")
    print(f"Amino Acid list: {', '.join(all_amino_acids_names)}.")
    print(f"Original Feature dimensions: {X_original_scaled.shape[1]}")
    print(f"Total samples: {X_original_scaled.shape[0]}")

    # --- Feature Engineering ---
    print("\n--- Performing Feature Engineering ---")
    df_raw_features_for_engineering = df_original[original_feature_columns].copy()
    X_engineered_scaled = engineer_features(df_raw_features_for_engineering, chiralities)
    X_full_scaled = X_engineered_scaled  # This is the final feature set for clustering

    print(f"Final feature set (original + engineered) dimensions: {X_full_scaled.shape[1]}")

    # Get save directory
    save_base_dir = get_save_directory()
    if not save_base_dir:
        messagebox.showinfo("Info", "No save directory selected. Results will not be saved to files.")
        save_base_dir = None
    else:
        os.makedirs(save_base_dir, exist_ok=True)
        print(f"\nResults will be saved to: {save_base_dir}")
        X_full_scaled.to_csv(os.path.join(save_base_dir, "engineered_features_scaled.csv"), index=True)
        y_full_encoded.to_frame(name='Encoded_AA_Label').to_csv(os.path.join(save_base_dir, "preprocessed_labels.csv"),
                                                                index=True)
        label_mapping_df = pd.DataFrame(
            {'Encoded_ID': range(len(all_amino_acids_names)), 'Amino_Acid_Name': all_amino_acids_names})
        label_mapping_df.to_csv(os.path.join(save_base_dir, "amino_acid_label_mapping.csv"), index=False)
        print("Preprocessed and engineered data (features, labels, and label mapping) saved as CSV.")

    # --- User Select Amino Acid Subset ---
    print("\n--- User Amino Acid Subset Selection ---")
    # This will now correctly show a Tkinter window if TKINTER_AVAILABLE is True
    selected_amino_acids_subset = select_amino_acid_subset(all_amino_acids_names)

    if not selected_amino_acids_subset or len(selected_amino_acids_subset) < 2:
        messagebox.showinfo("Info", "No valid amino acid subset selected (or less than 2 selected). Exiting program.")
        return

    print(f"\nSelected amino acid subset for clustering: {', '.join(selected_amino_acids_subset)}")

    # --- Prepare data for selected subset ---
    y_subset_remapped, original_subset_labels_names = remap_labels_for_subset(
        y_full_encoded, all_amino_acids_names, selected_amino_acids_subset
    )
    X_subset = X_full_scaled.loc[y_subset_remapped.index]

    k_clusters = len(original_subset_labels_names)  # k_clusters is the number of selected amino acids
    n_repeats = 10  # Default repeats for robustness (as per original code context, can be user-defined)

    if X_subset.empty or k_clusters < 2 or X_subset.shape[0] < k_clusters:
        messagebox.showerror("Error",
                             f"Insufficient data for clustering with selected subset ({k_clusters} amino acids, {X_subset.shape[0]} samples). Exiting.")
        return

    print(f"\n--- Performing LDA+KMeans Clustering for selected subset ---")
    print(f"Number of clusters (k): {k_clusters}")
    print(f"Number of repeats (n_repeats): {n_repeats}")

    method_name = 'LDA+KMeans'
    ari_scores_repeats = []
    purity_scores_repeats = []
    all_cms_repeats = []

    # Store model instances from one successful run for this method and combination
    final_clustering_model_instance = None
    final_dim_reduction_model_instance = None
    final_X_lda_projection = None
    final_cluster_labels = None

    for repeat_idx in range(n_repeats):
        current_random_state = 42 + repeat_idx  # Use a consistent but varying random state

        cluster_labels, clustering_model_inst, dim_reduction_model_inst, X_lda_transformed = cluster_with_method(
            X_subset, y_subset_remapped, method_name, k_clusters, random_state_val=current_random_state
        )

        if cluster_labels is None:
            print(f"  Repeat {repeat_idx + 1}/{n_repeats}: LDA+KMeans failed. Skipping this repeat.")
            continue

        # Capture model instances and LDA projection from the first successful run for later saving
        if final_clustering_model_instance is None:
            final_clustering_model_instance = clustering_model_inst
            final_dim_reduction_model_instance = dim_reduction_model_inst
            final_X_lda_projection = X_lda_transformed
            final_cluster_labels = cluster_labels

        eval_results_repeat = evaluate_clustering(
            X_subset, y_subset_remapped, cluster_labels, tuple(selected_amino_acids_subset),
            original_subset_labels_names
        )
        ari_scores_repeats.append(eval_results_repeat.get('ARI', 0.0))
        all_cms_repeats.append(eval_results_repeat.get('Confusion_Matrix'))

        # Calculate purity for each repeat (using fewer bootstraps for quick assessment)
        purity, _, _ = calculate_purity_metrics(y_subset_remapped, cluster_labels, n_bootstraps=50)
        purity_scores_repeats.append(purity)

        print(
            f"  Repeat {repeat_idx + 1}/{n_repeats}: ARI={ari_scores_repeats[-1]:.4f}, Purity={purity_scores_repeats[-1]:.4f}")

    if ari_scores_repeats:
        mean_ari = np.mean(ari_scores_repeats)
        std_ari = np.std(ari_scores_repeats)

        # Calculate full purity with 1000 bootstraps for the final report
        mean_purity, lower_purity_ci, upper_purity_ci = calculate_purity_metrics(y_subset_remapped,
                                                                                 final_cluster_labels,
                                                                                 n_bootstraps=1000)
        purity_display_str = f"{mean_purity * 100:.1f}% ± {(upper_purity_ci - lower_purity_ci) * 100 / 2:.1f}% CI [{lower_purity_ci * 100:.1f}%, {upper_purity_ci * 100:.1f}%]"

        print(f"\n--- LDA+KMeans Final Results for {', '.join(selected_amino_acids_subset)} ---")
        print(f"Mean ARI: {mean_ari:.4f} (Std: {std_ari:.4f})")
        print(f"Mean Purity: {purity_display_str}")
    else:
        print("\nLDA+KMeans clustering failed for the selected subset. No results to display or save.")
        return

    # --- Save Results and Visualizations ---
    if save_base_dir:
        combo_folder_name = "_".join(selected_amino_acids_subset)
        output_folder = os.path.join(save_base_dir, f"LDA_KMeans_Results_k{k_clusters}_{combo_folder_name}")
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nResults will be saved to: {output_folder}")

        # Save LDA Projection with Cluster Labels
        if final_X_lda_projection is not None and final_cluster_labels is not None:
            lda_proj_df = final_X_lda_projection.copy()
            lda_proj_df.columns = [f'LDA{i + 1}' for i in range(lda_proj_df.shape[1])]
            lda_proj_df['PredictedCluster'] = final_cluster_labels
            lda_proj_df.index.name = 'SampleID'
            lda_proj_file_path = os.path.join(output_folder, "LDA_projection_with_cluster.txt")
            lda_proj_df.to_csv(lda_proj_file_path, sep='\t', index=True)
            print(f"LDA projection with cluster labels saved to: {lda_proj_file_path}")

        # Generate Visualizations (Purity Matrix and 2D Cluster Plot)
        # We need the ARI score for the plot title, using the mean ARI from repeats
        mean_ari_for_plot = mean_ari

        # Confusion Matrix (CSV)
        first_successful_cm = next((cm for cm in all_cms_repeats if cm is not None), None)
        if first_successful_cm is not None:
            cm_filename_base = os.path.join(output_folder, f"LDA_KMeans_Confusion")
            pd.DataFrame(first_successful_cm, index=original_subset_labels_names,
                         columns=[f"Cluster {i}" for i in range(first_successful_cm.shape[1])]).to_csv(
                f"{cm_filename_base}.csv")
            print(f"Confusion matrix data saved to: {cm_filename_base}.csv")
        else:
            print("No valid confusion matrix available for saving.")

        # Plot 2D Cluster Plot (from LDA output)
        plot_dual_encoded_filename = os.path.join(output_folder, f"LDA_KMeans_2D_Plot.png")
        if final_X_lda_projection is not None and final_cluster_labels is not None:
            plot_dual_encoded_clusters(
                final_X_lda_projection, y_subset_remapped, final_cluster_labels, tuple(selected_amino_acids_subset),
                plot_dual_encoded_filename, original_subset_labels_names,
                ari_score=mean_ari_for_plot, purity_score_display=purity_display_str,
                method_name=method_name, k_value=k_clusters
            )
            print(f"2D Cluster Plot saved to: {plot_dual_encoded_filename}")
        else:
            print("Skipping 2D Cluster Plot due to missing LDA projection or cluster labels.")

        # Plot Cluster Purity Matrix
        purity_matrix_filename = os.path.join(output_folder, f"LDA_KMeans_Purity_Matrix.png")
        if final_cluster_labels is not None:
            plot_cluster_purity_matrix(
                y_subset_remapped, final_cluster_labels, tuple(selected_amino_acids_subset), purity_matrix_filename,
                original_subset_labels_names,
                ari_score=mean_ari_for_plot, purity_score_display=purity_display_str,
                method_name=method_name, k_value=k_clusters
            )
            print(f"Cluster Purity Matrix saved to: {purity_matrix_filename}")
        else:
            print("Skipping Cluster Purity Matrix due to missing cluster labels.")

        # --- Save the LDA+KMeans Model Pipeline ---
        print("\n--- Saving LDA+KMeans Model Pipeline ---")

        # The saved models below assume the input data for prediction has already gone through
        # the same feature engineering steps as 'X_full_scaled' and then been fed to the global scaler.
        # If a full raw-to-cluster pipeline is critical, the 'engineer_features' function needs refactoring
        # into a scikit-learn-compatible transformer.

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_base_filename = os.path.join(output_folder, f"lda_kmeans_model_{timestamp}")

        # Save Global Scaler (applied to all features after engineering)
        if global_scaler_for_X:  # Check if global_scaler_for_X was successfully created
            joblib.dump(global_scaler_for_X, f"{model_base_filename}_global_scaler.pkl")
            print(f"Global StandardScaler saved as {model_base_filename}_global_scaler.pkl")
        else:
            print("Global StandardScaler not available for saving.")

        # Save Dim Reduction Model (LDA)
        dim_red_model_filename = "N/A"
        if final_dim_reduction_model_instance:
            dim_red_model_type = type(final_dim_reduction_model_instance).__name__
            joblib.dump(final_dim_reduction_model_instance,
                        f"{model_base_filename}_{dim_red_model_type.lower()}_reducer.pkl")
            dim_red_model_filename = f"{model_base_filename}_{dim_red_model_type.lower()}_reducer.pkl"
            print(f"Dimensionality Reduction Model ({dim_red_model_type}) saved as {dim_red_model_filename}")
        else:
            print("Dimensionality Reduction Model (LDA) not available for saving.")

        # Save Clustering Model (KMeans)
        if final_clustering_model_instance:
            clustering_model_type = type(final_clustering_model_instance).__name__
            joblib.dump(final_clustering_model_instance,
                        f"{model_base_filename}_{clustering_model_type.lower()}_clusterer.pkl")
            print(
                f"Clustering Model ({clustering_model_type}) saved as {model_base_filename}_{clustering_model_type.lower()}_clusterer.pkl")
        else:
            print("Clustering Model (KMeans) not available for saving.")

        # Save metadata for the saved pipeline
        pipeline_metadata = {
            'amino_acid_list': selected_amino_acids_subset,
            'k_value': k_clusters,
            'clustering_method': method_name,
            'clustering_parameters': {'n_clusters': k_clusters, 'random_state': 42},  # Using fixed 42 for simplicity
            'feature_names_for_subset': X_subset.columns.tolist(),
            'ARI_mean': mean_ari,
            'Purity_mean': mean_purity,
            'Purity_CI_lower': lower_purity_ci,
            'Purity_CI_upper': upper_purity_ci,
            'global_scaler_model_path': f"{model_base_filename}_global_scaler.pkl" if global_scaler_for_X else "N/A",
            'dim_reduction_model_path': dim_red_model_filename,
            'clustering_model_path': f"{model_base_filename}_{clustering_model_type.lower()}_clusterer.pkl" if final_clustering_model_instance else "N/A",
        }
        save_json_results(pipeline_metadata, "lda_kmeans_pipeline_metadata.json", output_folder)
        print(f"Pipeline metadata saved to: {os.path.join(output_folder, 'lda_kmeans_pipeline_metadata.json')}")

    else:
        print("\nNo save directory selected. Models and detailed results not saved.")

    print("\n--- System Execution Complete ---")


if __name__ == '__main__':
    main()
