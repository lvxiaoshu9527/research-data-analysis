import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS  # Added OPTICS
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors # For DBSCAN/OPTICS epsilon estimation
from matplotlib.patches import Ellipse
import os
import sys
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import itertools
import random
from scipy.spatial.distance import pdist, squareform
import math
from typing import Tuple, List, Dict, Any, Generator, Optional
from warnings import filterwarnings
import json
import networkx as nx
import joblib  # For model saving
import datetime  # For timestamping model files

# Suppress all warnings for cleaner output
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=DeprecationWarning)

# --- Constants for Configuration ---
N_REPEATS = 3  # Number of times to repeat clustering for robustness
AAINDEX_FEATURES_TO_SELECT_COUNT = 8  # Target number of AAindex features after PCA/selection
ARI_STOP_THRESHOLD = 0.8  # Criterion for maximally separable and minimum stable k
SILHOUETTE_LOWER_BOUND = 0  # Criterion for ma

# ximally separable and minimum stable k
STABILITY_THRESHOLD_ARI_STD_BASE = 0.05  # Base threshold for ARI Standard Deviation for stable k

# --- Tkinter and UMAP import/fallback logic ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TKINTER_AVAILABLE = True
except ImportError:
    print("Tkinter not available. Falling back to console input for file paths.")
    TKINTER_AVAILABLE = False


    class MockTkinter:
        def Tk(self): return self

        def withdraw(self): pass

        def filedialog(self): return self

        def askopenfilename(self, *args, **kwargs): return input("Enter path to data CSV file: ")

        def askdirectory(self, *args, **kwargs): return input("Enter directory to save results: ")

        def messagebox(self): return self

        def showinfo(self, *args, **kwargs): print(f"INFO: {args[1]}")

        def showerror(self, *args, **kwargs): print(f"ERROR: {args[1]}")


    tk = MockTkinter()
    filedialog = tk.filedialog()
    messagebox = tk.messagebox()

try:
    import umap.umap_ as umap
except ImportError:
    print("UMAP is not installed. Please install it: pip install umap-learn")
    sys.exit(1)

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    print(
        "sklearn-extra not installed. KMedoids will not be available. Please install it: pip install scikit-learn-extra")
    KMedoids = None

try:
    from ccp_kmeans import COPKMeans
except ImportError:
    print(
        "COPKMeans library not installed. COP-KMeans will not be available. Please install it: pip install ccp-kmeans")
    COPKMeans = None

mpl.rcParams['font.family'] = ['Arial', 'Microsoft YaHei', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


# --- User-provided file selection and save functions ---
def select_file() -> Optional[str]:
    """Opens a file selection dialog to choose the raw data CSV file, or prompts via console."""
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Raw Data CSV File",
            filetypes=[("CSV files", "*.csv")]
        )
        root.destroy()
    else:
        file_path = input("Please enter the full path to your raw data CSV file: ")
    return file_path


def get_save_directory() -> Optional[str]:
    """Opens a folder selection dialog for the user to choose a directory for saving files, or prompts via console."""
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        save_dir = filedialog.askdirectory(
            title="Select Directory to Save Results"
        )
        root.destroy()
    else:
        save_dir = input("Please enter the full path to the directory where results should be saved: ")
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


# --- Physicochemical Properties PCA Function ---
def plot_property_pca(properties_df: pd.DataFrame, filename: str):
    """
    Performs PCA on amino acid physicochemical properties and visualizes them.
    Args:
        properties_df (pd.DataFrame): Amino acid physicochemical properties DataFrame.
        filename (str): File path to save the plot.
    """
    if properties_df.empty or properties_df.shape[0] < 2:
        print("Warning: Insufficient physicochemical property data to perform PCA visualization.")
        return

    scaler = StandardScaler()
    properties_scaled = pd.DataFrame(scaler.fit_transform(properties_df),
                                     columns=properties_df.columns,
                                     index=properties_df.index)

    pca = PCA(n_components=min(2, properties_scaled.shape[1]), random_state=42)
    properties_pca = pca.fit_transform(properties_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=properties_pca[:, 0],
        y=properties_pca[:, 1],
        hue=properties_df.index.tolist(),
        palette='tab20',
        s=100,
        alpha=0.8,
        legend='full'
    )
    plt.title(
        f"PCA of Amino Acid Physical-Chemical Properties\n(Explained Variance: {pca.explained_variance_ratio_.sum():.2f})",
        fontsize=14)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Amino Acid", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()
    print(f"Amino acid physicochemical properties PCA plot saved to: {filename}")


# --- 1. Data Preparation & Preprocessing Functions ---
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

    return X_scaled, y_encoded, unique_original_labels, scaler  # Return the scaler instance


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


def _parse_aaindex_text_file(filepath: str) -> pd.DataFrame:
    """
    Parses a AAindex text file (e.g., aaindex1.txt) and extracts amino acid properties.
    It focuses on indices that provide values for individual amino acids (not pairs like A/L).
    """
    aaindex_data_raw = {}
    current_index_id = None
    in_data_section = False
    current_aas_order = []
    all_standard_aas = "ARNDCQEGHILKMFPSTWYV"
    temp_values_for_current_index = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                if line.startswith('H'):
                    if current_index_id and in_data_section:
                        if len(temp_values_for_current_index) == len(current_aas_order):
                            aaindex_data_raw[current_index_id] = {aa: val for aa, val in
                                                                  zip(current_aas_order, temp_values_for_current_index)}
                        else:
                            print(f"Warning: Data mismatch for '{current_index_id}'. Skipping.")
                            aaindex_data_raw.pop(current_index_id, None)
                    current_index_id = line[1:].strip().split()[0]
                    aaindex_data_raw[current_index_id] = {}
                    in_data_section = False
                    current_aas_order = []
                    temp_values_for_current_index = []
                elif line.startswith('I'):
                    parts = line[1:].strip().split()
                    if any('/' in p for p in parts):
                        current_index_id = None
                        continue
                    current_aas_order = [p.upper() for p in parts if len(p) == 1 and p.upper() in all_standard_aas]
                    if not current_aas_order:
                        current_index_id = None
                        continue
                    in_data_section = True
                elif line.startswith('//'):
                    if current_index_id and in_data_section:
                        if len(temp_values_for_current_index) == len(current_aas_order):
                            aaindex_data_raw[current_index_id] = {aa: val for aa, val in
                                                                  zip(current_aas_order, temp_values_for_current_index)}
                        else:
                            print(f"Warning: Data mismatch for '{current_index_id}' before '//'. Skipping.")
                            aaindex_data_raw.pop(current_index_id, None)
                    in_data_section = False
                    current_index_id = None
                    temp_values_for_current_index = []
                elif in_data_section and current_index_id:
                    for val_str in line.split():
                        try:
                            temp_values_for_current_index.append(float(val_str) if val_str != '-' else np.nan)
                        except ValueError:
                            temp_values_for_current_index.append(np.nan)

        if current_index_id and in_data_section:
            if len(temp_values_for_current_index) == len(current_aas_order):
                aaindex_data_raw[current_index_id] = {aa: val for aa, val in
                                                      zip(current_aas_order, temp_values_for_current_index)}
            else:
                print(f"Warning: Data mismatch for last entry '{current_index_id}'. Skipping.")
                aaindex_data_raw.pop(current_index_id, None)

        df_aaindex = pd.DataFrame.from_dict(
            {idx_id: data_dict for idx_id, data_dict in aaindex_data_raw.items() if data_dict}, orient='index').T
        df_aaindex.index.name = 'Amino_Acid'
        df_aaindex = df_aaindex[df_aaindex.index.isin(list(all_standard_aas))]
        for col in df_aaindex.columns:
            df_aaindex[col] = df_aaindex[col].fillna(
                df_aaindex[col].mean() if not df_aaindex[col].isnull().all() else 0.0)
        df_aaindex = df_aaindex.reindex(list(all_standard_aas), fill_value=np.nan)
        return df_aaindex.astype(float)

    except FileNotFoundError:
        print(f"Error: AAindex file not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error parsing AAindex file {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def load_aaindex_properties(filepath: str,
                            aaindex_features_to_select_count: int = AAINDEX_FEATURES_TO_SELECT_COUNT) -> pd.DataFrame:
    """
    Loads AAindex data by parsing the text file, performs PCA for dimensionality reduction,
    and returns selected components.
    """
    print(f"\nAttempting to parse AAindex data from: {filepath}")
    aaindex_df = _parse_aaindex_text_file(filepath)

    if aaindex_df.empty:
        print("No valid AAindex properties parsed. Returning empty DataFrame.")
        return pd.DataFrame()

    numeric_cols = aaindex_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        print("No numeric properties found in parsed AAindex data.")
        return pd.DataFrame()

    aaindex_numeric = aaindex_df[numeric_cols]
    aaindex_numeric = aaindex_numeric.dropna(axis=1, how='all')
    aaindex_numeric = aaindex_numeric.loc[:, (aaindex_numeric != aaindex_numeric.iloc[0]).any()]

    if aaindex_numeric.empty:
        print("AAindex data is empty after cleaning for PCA.")
        return pd.DataFrame()

    n_components_pca = min(aaindex_features_to_select_count, aaindex_numeric.shape[1])
    if n_components_pca < 1:
        print("Not enough unique numeric features in AAindex for PCA after selection.")
        return pd.DataFrame()

    scaler = StandardScaler()
    aaindex_scaled = pd.DataFrame(scaler.fit_transform(aaindex_numeric),
                                  columns=aaindex_numeric.columns,
                                  index=aaindex_numeric.index)

    pca = PCA(n_components=n_components_pca, random_state=42)
    aaindex_pca_features = pca.fit_transform(aaindex_scaled)

    pca_cols = [f'AAindex_PC{i + 1}' for i in range(aaindex_pca_features.shape[1])]
    aaindex_processed_df = pd.DataFrame(aaindex_pca_features, columns=pca_cols, index=aaindex_scaled.index)

    print(
        f"AAindex loaded and PCA-reduced to {aaindex_processed_df.shape[1]} features (explaining {pca.explained_variance_ratio_.sum():.2f} variance).")
    return aaindex_processed_df


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


# --- 2. Combination Search Strategy Functions ---
def get_combination_sampling_count(total_combinations: int, k_value: int) -> int:
    """
    Calculates the number of combinations to sample based on total available combinations and k value.
    """
    # if k_value <= 5:
    #     return min(200, total_combinations) #200
    # elif k_value <= 10:
    #     return min(5000, total_combinations) #5000
    # else:
    #     return min(10000, total_combinations) #10000
    if k_value <= 5:
        return min(100, total_combinations)
    elif k_value <= 8:
        return min(500, total_combinations)
    elif k_value <= 10:
        return min(1000, total_combinations)
    else:
        return min(1500, total_combinations)


def prioritize_combinations_by_properties(
        amino_acids: List[str],
        k: int,
        properties_df: pd.DataFrame,
        sample_count: int,
        random_seed: int
) -> List[Tuple[str, ...]]:
    """
    Prioritizes combinations based on physicochemical property differences using average Euclidean distance.
    Then samples based on the calculated sampling count.
    """
    all_combinations = list(itertools.combinations(amino_acids, k))

    if len(all_combinations) <= sample_count:
        return all_combinations

    relevant_properties_df = properties_df.loc[properties_df.index.intersection(amino_acids)]

    if relevant_properties_df.empty or relevant_properties_df.shape[0] < k:
        print(
            f"Warning: Insufficient physicochemical property data for k={k} prioritization. Falling back to random sampling.")
        random.seed(random_seed)
        return random.sample(all_combinations, sample_count)

    combination_scores = []
    for combo in all_combinations:
        combo_properties = relevant_properties_df.loc[list(combo)]
        if combo_properties.shape[0] < k or combo_properties.shape[0] < 2:
            avg_distance = 0.0
        else:
            distances = pdist(combo_properties, metric='euclidean')
            avg_distance = distances.mean()
        combination_scores.append((combo, avg_distance))

    combination_scores.sort(key=lambda x: x[1], reverse=True)
    prioritized_combinations = [combo for combo, score in combination_scores[:sample_count]]

    return prioritized_combinations


def combination_search_controller(
        all_amino_acids: List[str],
        max_k: int,
        min_k: int = 2,
        processed_aaindex_df: Optional[pd.DataFrame] = None,
        base_seed_for_sampling: int = 42
) -> Generator[Tuple[Tuple[str, ...], int, str], None, None]:
    """
    Controls amino acid combination generation and selection with tiered sampling.
    Yields: (combination_tuple, k_value, combination_type_str)
    """
    for k in range(min_k, max_k + 1):
        total_combinations_for_k = math.comb(len(all_amino_acids), k)
        sample_count_for_k = get_combination_sampling_count(total_combinations_for_k, k)

        num_property_based = int(sample_count_for_k * 0.8)
        num_random_sample = sample_count_for_k - num_property_based

        current_k_combinations = []

        if processed_aaindex_df is not None and not processed_aaindex_df.empty:
            available_for_property_sort = [aa for aa in all_amino_acids if aa in processed_aaindex_df.index]
            if len(available_for_property_sort) >= k:
                prioritized = prioritize_combinations_by_properties(
                    available_for_property_sort, k, processed_aaindex_df, num_property_based, base_seed_for_sampling + k
                )
                current_k_combinations.extend(prioritized)
            else:
                num_random_sample += num_property_based
        else:
            num_random_sample = sample_count_for_k

        all_possible_combos_for_k = list(itertools.combinations(all_amino_acids, k))
        remaining_for_random_sampling = [combo for combo in all_possible_combos_for_k if
                                         combo not in current_k_combinations]

        if num_random_sample > 0 and remaining_for_random_sampling:
            random.seed(base_seed_for_sampling + k + 1000)
            random_sampled = random.sample(remaining_for_random_sampling,
                                           min(num_random_sample, len(remaining_for_random_sampling)))
            current_k_combinations.extend(random_sampled)

        print(f"Total {len(current_k_combinations)} combinations selected for k={k}.")

        yielded_combos = set()
        for combo in current_k_combinations:
            if combo not in yielded_combos:
                yield combo, k, "mixed_sampling"
                yielded_combos.add(combo)


# --- 3. Dual-Mode Clustering Evaluation System Functions ---
def _auto_constraint_generation(X: pd.DataFrame, y_true: pd.Series, num_cannot_links_per_class_pair: int = 5) -> Tuple[
    List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Automatically generates must-link and cannot-link constraints for COP-KMeans.
    """
    must_links = []
    cannot_links = []

    labels_unique = y_true.unique()
    label_to_indices = {label: list(y_true[y_true == label].index) for label in labels_unique}

    for label in labels_unique:
        indices = label_to_indices[label]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                must_links.append((indices[i], indices[j]))

    if len(labels_unique) > 1:
        for i, label1 in enumerate(labels_unique):
            for j, label2 in enumerate(labels_unique):
                if label1 >= label2: continue
                indices1 = label_to_indices[label1]
                indices2 = label_to_indices[label2]
                if not indices1 or not indices2: continue
                num_generated = 0
                max_attempts = num_cannot_links_per_class_pair * 10
                attempts = 0
                while num_generated < num_cannot_links_per_class_pair and attempts < max_attempts:
                    idx_a = random.choice(indices1)
                    idx_b = random.choice(indices2)
                    if (idx_a, idx_b) not in cannot_links and (idx_b, idx_a) not in cannot_links:
                        cannot_links.append((idx_a, idx_b))
                        num_generated += 1
                    attempts += 1
    return must_links, cannot_links


def cluster_with_method(
        X: pd.DataFrame,
        y_true: pd.Series,
        method: str,
        n_clusters: int,
        random_state_val: int,
        **kwargs
) -> Tuple[Optional[np.ndarray], Any, Any]:  # Now returns cluster_labels, clustering_model, dim_reduction_model
    """
    Unified clustering interface, returns cluster_labels, clustering_model_instance, and dim_reduction_model_instance (if any).
    """
    cluster_labels = None
    clustering_model = None
    dim_reduction_model = None  # For LDA

    if X.shape[0] == 0: return None, None, None
    if n_clusters <= 0 and method not in ['DBSCAN', 'OPTICS']: return None, None, None
    if X.shape[0] < n_clusters and method not in ['DBSCAN', 'OPTICS', 'LDA+KMeans']: return None, None, None

    try:
        if method == 'KMeans':
            clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state_val, n_init='auto', **kwargs)
            cluster_labels = clustering_model.fit_predict(X)

        elif method == 'KMedoids':
            if KMedoids is None: print("KMedoids not available."); return None, None, None
            clustering_model = KMedoids(n_clusters=n_clusters, random_state=random_state_val, metric='euclidean',
                                        **kwargs)
            cluster_labels = clustering_model.fit_predict(X)

        elif method == 'LDA+KMeans':
            unique_true_labels = y_true.unique()
            if len(unique_true_labels) < 2:
                print("LDA+KMeans Error: Fewer than 2 unique true labels, cannot perform LDA dimensionality reduction.")
                return None, None, None

            n_components_lda = min(n_clusters - 1, len(unique_true_labels) - 1, X.shape[1])
            if n_components_lda <= 0:
                print("LDA dimensionality reduction will result in non-positive dimensions, skipping LDA+KMeans.")
                return None, None, None

            try:
                lda = LDA(n_components=n_components_lda, **kwargs)
                X_lda = lda.fit_transform(X, y_true)
                dim_reduction_model = lda  # Capture the fitted LDA model
            except Exception as e:
                print(f"LDA failed: {e}. Skipping LDA+KMeans.")
                return None, None, None

            clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state_val, n_init='auto', **kwargs)
            cluster_labels = clustering_model.fit_predict(X_lda)

        elif method == 'COP-KMeans':
            if COPKMeans is None: print("COP-KMeans not available."); return None, None, None
            must_links, cannot_links = _auto_constraint_generation(X, y_true)
            clustering_model = COPKMeans(n_clusters=n_clusters, random_state=random_state_val, **kwargs)
            cluster_labels = clustering_model.fit_predict(X.values, ml=must_links, cl=cannot_links)

        elif method == 'DBSCAN':
            if X.shape[0] < 2: print("DBSCAN: Not enough samples."); return None, None, None
            min_samples = kwargs.get('min_samples', max(3, int(X.shape[1] * 1.5)))
            nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
            distances, _ = nn.kneighbors(X)
            estimated_eps = np.mean(distances[:, min_samples - 1])
            eps = kwargs.get('eps', estimated_eps * 1.0)
            eps = max(0.01, eps)
            clustering_model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
            cluster_labels = clustering_model.fit_predict(X)
            if len(np.unique(cluster_labels[cluster_labels != -1])) <= 1: return None, None, None

        elif method == 'OPTICS':
            if OPTICS is None: print("OPTICS not available."); return None, None, None
            if X.shape[0] < 2: print("OPTICS: Not enough samples."); return None, None, None
            min_samples = kwargs.get('min_samples', max(3, int(X.shape[1] * 1.5)))
            clustering_model = OPTICS(min_samples=min_samples, **kwargs)
            cluster_labels = clustering_model.fit_predict(X)
            if len(np.unique(cluster_labels[cluster_labels != -1])) <= 1: return None, None, None

        elif method == 'AgglomerativeClustering':
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
            cluster_labels = clustering_model.fit_predict(X)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    except Exception as e:
        print(f"Clustering method {method} failed: {e}. Returning None labels and models.")
        return None, None, None

    return cluster_labels, clustering_model, dim_reduction_model


# --- 4. Evaluation Metrics Functions ---
def evaluate_clustering(
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        combo_name: Tuple[str, ...],
        original_subset_labels_names: List[str]
) -> Dict[str, Any]:
    """Evaluates clustering results using ARI, Silhouette Score, and Confusion Matrix."""
    results = {'combination': combo_name}

    if len(y_true.unique()) > 1 and len(np.unique(y_pred[y_pred != -1])) > 1:
        results['ARI'] = adjusted_rand_score(y_true, y_pred)
    else:
        results['ARI'] = 0.0

    if X.shape[0] > 1 and len(np.unique(y_pred[y_pred != -1])) > 1:
        try:
            non_noise_indices = y_pred != -1
            if np.sum(non_noise_indices) > 1 and len(np.unique(y_pred[non_noise_indices])) > 1:
                results['Silhouette_Score'] = silhouette_score(X[non_noise_indices], y_pred[non_noise_indices])
            else:
                results['Silhouette_Score'] = np.nan
        except Exception:
            results['Silhouette_Score'] = np.nan
    else:
        results['Silhouette_Score'] = np.nan

    results['Confusion_Matrix'] = confusion_matrix(y_true, y_pred)
    return results


# --- 5. Visualization Output Functions ---
def plot_dual_encoded_clusters(
        X: pd.DataFrame,
        y_true_encoded: pd.Series,
        y_pred: np.ndarray,
        combo_name: Tuple[str, ...],
        filename: str,
        true_label_names: List[str],
        ari_score: Optional[float] = None,
        silhouette_score: Optional[float] = None,
        method_name: Optional[str] = None,
        k_value: Optional[int] = None
):
    """
    Plots a 2D scatter plot of the clustered data, showing true labels and predicted clusters.
    Performs dimensionality reduction (UMAP > PCA) if data has > 2 dimensions.
    Adds confidence ellipses for each predicted cluster.
    """
    if X.empty or y_true_encoded.empty or y_pred is None or len(np.unique(y_pred[y_pred != -1])) < 2:
        print(f"Warning: Insufficient data for 2D cluster plot for {combo_name}. Skipping.")
        return

    df_plot = X.copy()
    df_plot['True_Label_Encoded'] = y_true_encoded
    df_plot['Predicted_Cluster'] = y_pred

    # Filter out noise points for plotting and evaluation if method produces them
    df_plot_filtered = df_plot[df_plot['Predicted_Cluster'] != -1].copy()
    if df_plot_filtered.empty:
        print(f"Warning: No non-noise points available for 2D cluster plot for {combo_name}. Skipping.")
        return

    # Map encoded true labels back to original names for plotting
    df_plot_filtered['True_Label_Name'] = df_plot_filtered['True_Label_Encoded'].map(
        {i: name for i, name in enumerate(true_label_names)}
    )

    # Dimensionality Reduction if features > 2
    n_features = X.shape[1]
    plot_data = df_plot_filtered.drop(columns=['True_Label_Encoded', 'Predicted_Cluster', 'True_Label_Name'])
    dim_reduction_method = "Original"

    if n_features > 2:
        try:
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(plot_data)
            df_plot_filtered['Dim1'] = reduced_data[:, 0]
            df_plot_filtered['Dim2'] = reduced_data[:, 1]
            dim_reduction_method = "UMAP"
            print(f"  Applied UMAP for 2D visualization (k={k_value}, combo={combo_name}).")
        except Exception as e_umap:
            print(
                f"  UMAP failed ({e_umap}). Falling back to PCA for 2D visualization (k={k_value}, combo={combo_name}).")
            try:
                pca = PCA(n_components=2, random_state=42)
                reduced_data = pca.fit_transform(plot_data)
                df_plot_filtered['Dim1'] = reduced_data[:, 0]
                df_plot_filtered['Dim2'] = reduced_data[:, 1]
                dim_reduction_method = "PCA"
                print(
                    f"  Applied PCA for 2D visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2f}).")
            except Exception as e_pca:
                print(f"  PCA also failed ({e_pca}). Cannot perform 2D visualization for {combo_name}. Skipping.")
                return
    else:
        df_plot_filtered['Dim1'] = plot_data.iloc[:, 0]
        df_plot_filtered['Dim2'] = plot_data.iloc[:, 1]

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

            # Use a sufficiently large value for alpha to avoid singular matrix if cluster is too spread out
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
              f"k={k_value}, Dim Reduction: {dim_reduction_method}" +
              (f", ARI: {ari_score:.2f}" if ari_score is not None else "") +
              (f", Silhouette: {silhouette_score:.2f}" if silhouette_score is not None and not np.isnan(
                  silhouette_score) else ""),
              fontsize=14)
    plt.xlabel(f"{dim_reduction_method} Dim1", fontsize=12)
    plt.ylabel(f"{dim_reduction_method} Dim2", fontsize=12)
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

def plot_performance_trend(
        results_df: pd.DataFrame,
        filename: str,
        ari_threshold: float = ARI_STOP_THRESHOLD,
        y_axis_limit: Tuple[float, float] = (-1.0, 1.0)
):
    """
    Plots ARI and Silhouette trends vs. k, with separate lines for each method.
    Adds ARI threshold line and marks first breakpoint.
    """
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, 0])

    sns.lineplot(data=results_df, x='k_value', y='ARI_mean', marker='o', hue='clustering_method', ax=ax1, linewidth=2)
    ax1.set_xlabel("Number of Amino Acids (k)", fontsize=12)
    ax1.set_ylabel("Adjusted Rand Index (ARI)", color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(y_axis_limit)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    sns.lineplot(data=results_df, x='k_value', y='Silhouette_Score_mean', marker='x', hue='clustering_method', ax=ax2,
                 linestyle='--', alpha=0.7, linewidth=2, legend=False)
    ax2.set_ylabel("Silhouette Score", color='tab:orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(y_axis_limit)

    ax1.axhline(y=ari_threshold, color='red', linestyle=':', label=f'ARI Threshold ({ari_threshold})')

    overall_avg_results_by_k = results_df.groupby('k_value').agg({'ARI_mean': 'mean'}).reset_index()
    breakpoint_k = None
    for k_val in sorted(overall_avg_results_by_k['k_value'].unique()):
        avg_ari_at_k = overall_avg_results_by_k[overall_avg_results_by_k['k_value'] == k_val]['ARI_mean'].mean()
        if avg_ari_at_k >= ari_threshold:
            breakpoint_k = k_val
            break

    if breakpoint_k is not None:
        ax1.axvline(x=breakpoint_k, color='green', linestyle='--', label=f'First k ≥ ARI Threshold ({breakpoint_k})')
        ax1.text(breakpoint_k + 0.1, ax1.get_ylim()[1] * 0.8, f'k={breakpoint_k}', color='green', ha='left', va='top',
                 fontsize=10)

    handles1, labels1 = ax1.get_legend_handles_labels()
    dummy_silhouette_handle = plt.Line2D([0], [0], marker='x', color='gray', linestyle='--', alpha=0.7,
                                         label='Silhouette Score')
    handles1.append(dummy_silhouette_handle)
    labels1.append('Silhouette Score')

    ax1.legend(handles1, labels1, title="Legend", loc='lower right', bbox_to_anchor=(1.0, 0.0),
               ncol=1, fontsize=9, title_fontsize=10, frameon=True)

    plt.title("Clustering Performance Trend: ARI and Silhouette Score vs. k", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()


def mark_distinction_breakpoints(
        results_df: pd.DataFrame,
        metric: str,
        filename: str,
        threshold: float = ARI_STOP_THRESHOLD
):
    """
    Plots the trend of a metric (e.g., ARI mean) and marks the first k where the average crosses a threshold.
    """
    plt.figure(figsize=(10, 6))
    avg_metric_by_k = results_df.groupby('k_value')[metric].mean().reset_index()
    sns.lineplot(data=avg_metric_by_k, x='k_value', y=metric, marker='o', linewidth=2, color='blue',
                 label=f'Avg {metric} Across Methods')

    sns.lineplot(data=results_df, x='k_value', y=metric, hue='clustering_method', marker='x', linestyle='--', alpha=0.6,
                 legend=True)

    breakpoint_k = None
    for k_val in sorted(avg_metric_by_k['k_value'].unique()):
        avg_metric_at_k = avg_metric_by_k[avg_metric_by_k['k_value'] == k_val][metric].mean()
        if avg_metric_at_k >= threshold:
            breakpoint_k = k_val
            break

    if breakpoint_k is not None:
        plt.axvline(x=breakpoint_k, color='red', linestyle='--',
                    label=f'Breakpoint (k={breakpoint_k}, Avg {metric}>={threshold})')
        plt.text(breakpoint_k + 0.1, plt.ylim()[1] * 0.9, f'k={breakpoint_k}', color='red', ha='left', va='top',
                 fontsize=10)

    plt.axhline(y=threshold, color='purple', linestyle=':', label=f'{metric} Threshold ({threshold})')

    plt.title(f"{metric} Trend with Distinction Breakpoint (Overall Average)", fontsize=14)
    plt.xlabel("Number of Amino Acids (k)", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True)
    plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()


def plot_clustering_stability(
        results_df: pd.DataFrame,
        filename: str,
        stability_threshold_ari_std_base: float = STABILITY_THRESHOLD_ARI_STD_BASE
):
    """
    Plots the stability of clustering (ARI Standard Deviation) vs. k,
    with separate lines for each method and a dynamic stability threshold line.
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_df, x='k_value', y='ARI_std', marker='o', hue='clustering_method', linewidth=2)

    k_values = sorted(results_df['k_value'].unique())
    dynamic_thresholds = [stability_threshold_ari_std_base + 0.01 * k for k in k_values]
    plt.plot(k_values, dynamic_thresholds, color='red', linestyle=':',
             label=f'Dynamic Stability Threshold ({stability_threshold_ari_std_base} + 0.01*k)')

    plt.title("Clustering Stability: ARI Standard Deviation vs. k", fontsize=14)
    plt.xlabel("Number of Amino Acids (k)", fontsize=12)
    plt.ylabel("Adjusted Rand Index (ARI) Standard Deviation", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Clustering Method", loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()
    print(f"Clustering stability plot saved to: {filename}")


def plot_cluster_network_graph(
        X: pd.DataFrame,
        y_true_encoded: pd.Series,
        y_pred: np.ndarray,
        combo_name: Tuple[str, ...],
        filename: str,
        true_label_names: List[str],
        co_assignment_matrix_global: pd.DataFrame,
        all_amino_acids_names: List[str],
        method_name: Optional[str] = None,
        ari_score: Optional[float] = None,
        silhouette_score: Optional[float] = None,
        k_value: Optional[int] = None
):
    """
    Generates a network graph visualizing cluster relationships and amino acid confusion.
    """
    if X.empty or y_true_encoded.empty or y_pred is None or len(y_pred[y_pred != -1]) < 2:
        print(f"Warning: Insufficient data for network graph for {combo_name}. Skipping.")
        return

    G = nx.Graph()

    aa_nodes = true_label_names
    G.add_nodes_from(aa_nodes, bipartite=0, color='red')

    unique_clusters = np.unique(y_pred[y_pred != -1])
    if len(unique_clusters) == 0:
        print(f"Warning: No valid clusters found for network graph for {combo_name}. Skipping.")
        return

    cluster_nodes = [f"Cluster {c}" for c in unique_clusters]
    G.add_nodes_from(cluster_nodes, bipartite=1, color='blue')

    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    df_temp = pd.DataFrame({'True_AA_Encoded': y_true_encoded, 'Predicted_Cluster': y_pred})
    df_temp = df_temp[df_temp['Predicted_Cluster'] != -1]

    for true_aa_encoded in df_temp['True_AA_Encoded'].unique():
        true_aa_name = true_label_names[true_aa_encoded]
        subset_df_aa = df_temp[df_temp['True_AA_Encoded'] == true_aa_encoded]

        for predicted_cluster_id in subset_df_aa['Predicted_Cluster'].unique():
            count = len(subset_df_aa[subset_df_aa['Predicted_Cluster'] == predicted_cluster_id])
            if count > 0:
                cluster_node_name = f"Cluster {predicted_cluster_id}"
                if G.has_node(true_aa_name) and G.has_node(cluster_node_name):
                    G.add_edge(true_aa_name, cluster_node_name, weight=count)

                x1, y1 = pos[true_aa_name]
                x2, y2 = pos[cluster_node_name]
                line_width = np.log1p(count) / 2
                line_alpha = 0.6 + (0.4 * (count / X.shape[0]))

                plt.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=line_width, alpha=line_alpha,
                         zorder=1)

    relevant_aa_indices = [all_amino_acids_names.index(aa) for aa in combo_name if aa in all_amino_acids_names]

    for i_idx, j_idx in itertools.combinations(relevant_aa_indices, 2):
        global_aa1_name = all_amino_acids_names[i_idx]
        global_aa2_name = all_amino_acids_names[j_idx]

        if global_aa1_name not in combo_name or global_aa2_name not in combo_name:
            continue

        confusion_strength = co_assignment_matrix_global.iloc[i_idx, j_idx]

        if confusion_strength > 0:
            x1, y1 = pos[global_aa1_name]
            x2, y2 = pos[global_aa2_name]
            line_width = confusion_strength * 5
            line_alpha = 0.5 + (confusion_strength * 0.5)
            plt.plot([x1, x2], [y1, y2], color='darkorange', linestyle='--', linewidth=line_width, alpha=line_alpha,
                     zorder=0)

    nx.draw_networkx_nodes(G, pos, nodelist=aa_nodes, node_color='red', node_size=1000, alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_color='blue', node_size=1000, alpha=0.9, ax=ax)

    aa_labels = {node: node for node in aa_nodes}
    cluster_labels_dict = {node: node for node in cluster_nodes}

    nx.draw_networkx_labels(G, pos, aa_labels, font_size=10, font_color='black', ax=ax)
    nx.draw_networkx_labels(G, pos, cluster_labels_dict, font_size=10, font_color='white', ax=ax)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Amino Acid Node', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Cluster Node', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='AA to Cluster Assignment'),
        plt.Line2D([0], [0], color='darkorange', linestyle='--', linewidth=2, label='Confused AA Pair (Co-clustering)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, title="Legend")

    plot_title = f"Cluster Relation Network for {'_'.join(combo_name)}\n" \
                 f"Method: {method_name}, k={k_value}"
    if ari_score is not None: plot_title += f", ARI: {ari_score:.2f}"
    if silhouette_score is not None and not np.isnan(
        silhouette_score): plot_title += f", Silhouette: {silhouette_score:.2f}"

    ax.set_title(plot_title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()


def plot_cluster_purity_matrix(
        y_true_encoded: pd.Series,
        y_pred: np.ndarray,
        combo_name: Tuple[str, ...],
        filename: str,
        true_label_names: List[str],
        method_name: Optional[str] = None,
        ari_score: Optional[float] = None,
        silhouette_score: Optional[float] = None,
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
    if silhouette_score is not None and not np.isnan(
        silhouette_score): plot_title += f", Silhouette: {silhouette_score:.2f}"

    plt.title(plot_title, fontsize=14)
    plt.xlabel("True Amino Acid Label", fontsize=12)
    plt.ylabel("Predicted Cluster Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()


# --- 6. Property Analysis Functions ---
def load_amino_acid_properties_for_analysis(filepath: str) -> pd.DataFrame:
    """Loads physicochemical properties of amino acids specifically for analysis."""
    df = _parse_aaindex_text_file(filepath)
    if df.empty:
        print(f"Warning: No valid amino acid properties parsed from '{filepath}' for detailed analysis.")
        standard_aas = list("ARNDCQEGHILKMFPSTWYV")
        dummy_data = {
            'Hydrophobicity': [np.random.uniform(-2, 2) for _ in range(len(standard_aas))],
            'Molecular_Weight': [np.random.uniform(70, 250) for _ in range(len(standard_aas))],
            'pI': [np.random.uniform(3, 11) for _ in range(len(standard_aas))]
        }
        return pd.DataFrame(dummy_data).set_index('Amino_Acid')
    return df


def get_amino_acid_property_diffs(aa1_name: str, aa2_name: str, properties_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates property differences between two amino acids for specified properties."""
    properties_to_check = ['Hydrophobicity', 'Molecular_Weight', 'pI']

    if aa1_name not in properties_df.index or aa2_name not in properties_df.index:
        return {f"{prop}_Diff": np.nan for prop in properties_to_check}

    props1 = properties_df.loc[aa1_name]
    props2 = properties_df.loc[aa2_name]

    diffs = {}
    for prop in properties_to_check:
        if prop in properties_df.columns:
            diffs[f'{prop}_Diff'] = abs(props1[prop] - props2[prop])
        else:
            diffs[f'{prop}_Diff'] = np.nan
    return diffs


def analyze_and_plot_confused_pairs_properties(
        cm: np.ndarray,
        true_label_names: List[str],
        properties_df: pd.DataFrame,
        filename: str,
        num_top_confused: int = 5
) -> pd.DataFrame:
    """
    Identifies top N confused pairs from the confusion matrix and analyzes their property differences.
    Generates an annotated confusion matrix plot focusing on these differences.
    Returns a DataFrame of top confused pairs with their properties.
    """
    confused_pairs_data = []

    off_diagonal_counts = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j:
                off_diagonal_counts.append({'true_idx': i, 'pred_idx': j, 'count': cm[i, j]})

    off_diagonal_counts_sorted = sorted(off_diagonal_counts, key=lambda x: x['count'], reverse=True)

    unique_confused_aa_pairs = set()
    top_confused_pairs_summary = []

    for item in off_diagonal_counts_sorted:
        if item['count'] == 0: continue

        true_aa_name = true_label_names[item['true_idx']]
        confused_with_aa_name = true_label_names[item['pred_idx']]

        pair_key = tuple(sorted((true_aa_name, confused_with_aa_name)))

        if pair_key not in unique_confused_aa_pairs:
            prop_diffs = get_amino_acid_property_diffs(true_aa_name, confused_with_aa_name, properties_df)
            top_confused_pairs_summary.append({
                'Pair': f"{true_aa_name} <-> {confused_with_aa_name}",
                'Confusion_Count': item['count'],
                **prop_diffs
            })
            unique_confused_aa_pairs.add(pair_key)

        if len(top_confused_pairs_summary) >= num_top_confused:
            break

    top_confused_df = pd.DataFrame(top_confused_pairs_summary)

    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.3)
    ax_main = plt.subplot(gs[0, 0])
    ax_text = plt.subplot(gs[0, 1])
    ax_text.axis('off')

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Cluster {i}" for i in range(cm.shape[1])],
                yticklabels=true_label_names,
                linewidths=.5, linecolor='lightgray', ax=ax_main)
    ax_main.set_title(f"Confusion Matrix\n({', '.join(true_label_names)})", fontsize=14)
    ax_main.set_xlabel("Predicted Cluster Label", fontsize=12)
    ax_main.set_ylabel("True Amino Acid Label", fontsize=12)

    text_content = "Top Confused Pairs & Property Diffs:\n"
    if not top_confused_df.empty:
        for index, row in top_confused_df.head(num_top_confused).iterrows():
            text_content += (f"\nPair: {row['Pair']} (Count: {row['Confusion_Count']})\n"
                             f"  Hydro: {row.get('Hydrophobicity_Diff', np.nan):.2f}, "
                             f"MW: {row.get('Molecular_Weight_Diff', np.nan):.0f}, "
                             f"pI: {row.get('pI_Diff', np.nan):.2f}")
    else:
        text_content += "No significant confusion observed."

    ax_text.text(0, 1, text_content, transform=ax_text.transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    plt.tight_layout()

    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()
    print(f"Confusion matrix plot saved to: {filename}")

    return pd.DataFrame(top_confused_df)


def analyze_feature_importance(X_scaled_df: pd.DataFrame, y_encoded: pd.Series, filename: str):
    """
    Analyzes feature importance using Mutual Information and plots top features.
    """
    if X_scaled_df.empty or y_encoded.empty or len(y_encoded.unique()) < 2:
        print("Warning: Insufficient data for feature importance analysis.")
        return

    # Using sklearn's mutual_info_classif as a placeholder, requires explicit import
    # This function was not imported in the original code, but used in previous context.
    # Added it to imports at the top.
    from sklearn.feature_selection import mutual_info_classif
    mi_scores = mutual_info_classif(X_scaled_df, y_encoded, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_scaled_df.columns).sort_values(ascending=False)

    plt.figure(figsize=(12, 7))
    mi_series.head(20).plot(kind='barh', color=sns.color_palette("viridis", len(mi_series.head(20))))
    plt.title("Top 20 Feature Importance (Mutual Information with Amino Acid Label)", fontsize=14)
    plt.xlabel("Mutual Information Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()
    print(f"Feature importance plot saved to: {filename}")

    return mi_series


def calculate_amino_acid_distinguishability(all_results_df: pd.DataFrame,
                                            all_amino_acids_names: List[str]) -> pd.DataFrame:
    """
    Calculates a 'distinguishability score' for each amino acid.
    Defined as the average ARI for all combinations in which the amino acid appears.
    """
    distinguishability_scores = []

    for aa in all_amino_acids_names:
        aa_combinations_results = all_results_df[
            all_results_df['combination'].apply(lambda x: aa in x)
        ]

        if not aa_combinations_results.empty:
            mean_ari = aa_combinations_results['ARI'].mean()
            num_combinations = len(aa_combinations_results)
            distinguishability_scores.append({
                'Amino_Acid': aa,
                'Distinguishability_Score': mean_ari,
                'Count_Combinations': num_combinations
            })
        else:
            distinguishability_scores.append({
                'Amino_Acid': aa,
                'Distinguishability_Score': np.nan,
                'Count_Combinations': 0
            })

    df_distinguishability = pd.DataFrame(distinguishability_scores).sort_values(
        by='Distinguishability_Score', ascending=False
    )
    print("\nAmino Acid Distinguishability Scores (Mean ARI):")
    print(df_distinguishability)
    return df_distinguishability


def plot_confusion_frequency_heatmap(
        co_assignment_matrix: np.ndarray,
        total_pair_eval_counts: np.ndarray,
        all_amino_acids_names: List[str],
        filename: str
) -> pd.DataFrame:
    """
    Plots a heatmap of the normalized amino acid confusion frequency matrix.
    """
    normalized_matrix = co_assignment_matrix.copy().astype(float)

    for i in range(normalized_matrix.shape[0]):
        for j in range(normalized_matrix.shape[1]):
            if total_pair_eval_counts[i, j] > 0:
                normalized_matrix[i, j] = co_assignment_matrix[i, j] / total_pair_eval_counts[i, j]
            else:
                normalized_matrix[i, j] = 0.0

    normalized_matrix = np.clip(normalized_matrix, 0.0, 1.0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=all_amino_acids_names, yticklabels=all_amino_acids_names,
                linewidths=.5, linecolor='lightgray')
    plt.title("Amino Acid Co-Clustering Frequency Heatmap", fontsize=16)
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Amino Acid", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()
    print(f"Confusion frequency heatmap saved to: {filename}")

    return pd.DataFrame(normalized_matrix, index=all_amino_acids_names, columns=all_amino_acids_names)


def generate_combo_visualizations(
        combo_tuple: Tuple[str, ...],
        method_name: str,
        ari_score: float,
        silhouette_score: float,
        k_value: int,
        output_base_folder: str,
        X_full_scaled: pd.DataFrame,
        y_full_encoded: pd.Series,
        all_amino_acids_names: List[str],
        confusion_frequency_df: pd.DataFrame,
        aaindex_original_for_analysis: pd.DataFrame,
        all_results: List[Dict[str, Any]],
        random_seed: int = 99
):
    """Helper function to generate all 4 types of visualizations for a given combination."""
    y_subset_remapped, original_subset_labels_names = remap_labels_for_subset(
        y_full_encoded, all_amino_acids_names, list(combo_tuple)
    )
    X_subset = X_full_scaled.loc[y_subset_remapped.index]

    # Retrieve parameters to recreate the model for consistent plotting
    # Assuming all_results contains the necessary parameters for `cluster_with_method`
    relevant_result_entry = next(
        (res for res in all_results if res['k_value'] == k_value and
         res['combination'] == combo_tuple and res['clustering_method'] == method_name),
        None
    )

    if relevant_result_entry is None:
        print(f"Error: Could not find result entry for {combo_tuple} with {method_name} to generate visualizations.")
        return

    # Use the stored random state for consistency if available, otherwise use provided random_seed
    clustering_random_state = relevant_result_entry['clustering_model_params'].get('random_state', random_seed)

    final_plot_cluster_labels, _, _ = cluster_with_method(  # We don't need models here, just labels
        X_subset, y_subset_remapped, method_name, k_value, random_state_val=clustering_random_state
    )

    if final_plot_cluster_labels is None:
        print(
            f"Clustering failed for combination {combo_tuple} with method {method_name} for final plot generation. Skipping visualizations.")
        return

    combo_folder_name = "_".join(list(combo_tuple))
    output_folder = os.path.join(output_base_folder, f"Combo_k{k_value}AA_{combo_folder_name}_{method_name}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created folder: {output_folder}")

    # 2D Clustering Scatter Plot
    dual_encoded_filename = os.path.join(output_folder, f"Combo_{k_value}AA_{method_name}_Plot.png")
    plot_dual_encoded_clusters(
        X_subset, y_subset_remapped, final_plot_cluster_labels, combo_tuple, dual_encoded_filename,
        original_subset_labels_names,
        ari_score=ari_score, silhouette_score=silhouette_score, method_name=method_name, k_value=k_value
    )
    print(f"  Dual-encoded cluster plot saved to：{dual_encoded_filename}")

    # Cluster Relation Network Graph
    network_filename = os.path.join(output_folder, f"Combo_{k_value}AA_{method_name}_Network.png")
    plot_cluster_network_graph(
        X_subset, y_subset_remapped, final_plot_cluster_labels, combo_tuple, network_filename,
        original_subset_labels_names, confusion_frequency_df, all_amino_acids_names,
        method_name=method_name, ari_score=ari_score, silhouette_score=silhouette_score, k_value=k_value
    )
    print(f"  Network graph saved to：{network_filename}")

    # Cluster Purity Matrix
    purity_filename = os.path.join(output_folder, f"Combo_{k_value}AA_{method_name}_Purity.png")
    plot_cluster_purity_matrix(
        y_subset_remapped, final_plot_cluster_labels, combo_tuple, purity_filename,
        original_subset_labels_names,
        method_name=method_name, ari_score=ari_score, silhouette_score=silhouette_score, k_value=k_value
    )
    print(f"  Purity matrix saved to：{purity_filename}")

    # Confusion Matrix for analysis
    cm_filename_base = os.path.join(output_folder, f"Combo_{k_value}AA_{method_name}_Confusion")
    cm_data = relevant_result_entry['Confusion_Matrix']
    if cm_data is not None:
        pd.DataFrame(cm_data, index=original_subset_labels_names,
                     columns=[f"Cluster {i}" for i in range(cm_data.shape[1])]).to_csv(f"{cm_filename_base}.csv")
        print(f"  Confusion matrix data saved to: {cm_filename_base}.csv")
        analyze_and_plot_confused_pairs_properties(
            cm_data, original_subset_labels_names, aaindex_original_for_analysis,
            f"{cm_filename_base}.png", num_top_confused=5
        )
        print(f"  Confusion matrix plot saved to：{cm_filename_base}.png")
    else:
        print(f"  Warning: No valid confusion matrix found for {combo_tuple} with {method_name} for plotting.")


# --- 7. Main Logic Function ---
def main(continue_after_threshold: bool = True):
    """
    Controls the overall workflow: data loading, preprocessing, combination generation,
    clustering, evaluation, results visualization, and physicochemical analysis.
    """
    print("--- DNA-Wrapped SWNT Array Amino Acid Discrimination System ---")

    # --- Setup Directories ---
    data_dir = "data"
    results_dir = "results"
    os.makedirs(data_dir, exist_ok=True)
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
    X_engineered_scaled = engineer_features(df_original[original_feature_columns].copy(), chiralities)
    X_full_scaled = X_engineered_scaled

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

    # --- Feature Importance Analysis ---
    if save_base_dir:
        print("\n--- Performing Feature Importance Analysis ---")
        feature_importance_scores = analyze_feature_importance(X_full_scaled, y_full_encoded,
                                                               os.path.join(save_base_dir, "feature_importance.png"))
        print(f"Top 10 features by Mutual Information:\n{feature_importance_scores.head(10)}")

    # --- Load AAindex Properties and Plot PCA ---
    aaindex_properties_filepath = os.path.join(data_dir, "aaindex1.txt")
    processed_aaindex_df: Optional[pd.DataFrame] = None
    aaindex_original_for_analysis: Optional[pd.DataFrame] = None

    try:
        processed_aaindex_df = load_aaindex_properties(aaindex_properties_filepath, AAINDEX_FEATURES_TO_SELECT_COUNT)
        aaindex_original_for_analysis = _parse_aaindex_text_file(aaindex_properties_filepath)
        if aaindex_original_for_analysis.empty:
            print(
                f"Warning: Could not parse any individual AA properties from '{aaindex_properties_filepath}' for property difference analysis. Using dummy properties.")
            standard_aas = list("ARNDCQEGHILKMFPSTWYV")
            aaindex_original_for_analysis = pd.DataFrame({
                'Hydrophobicity': [np.random.uniform(-2, 2) for _ in range(len(standard_aas))],
                'Molecular_Weight': [np.random.uniform(70, 250) for _ in range(len(standard_aas))],
                'pI': [np.random.uniform(3, 11) for _ in range(len(standard_aas))]
            }, index=standard_aas)
            aaindex_original_for_analysis.index.name = 'Amino_Acid'

        if save_base_dir and not processed_aaindex_df.empty:
            plot_property_pca(processed_aaindex_df, os.path.join(save_base_dir, "amino_acid_properties_pca.png"))
        else:
            print("Skipping AAindex PCA plot due to no valid data or no save directory.")

    except Exception as e:
        print(
            f"\nError processing AAindex properties: {e}. Property-based combination search will use random sampling and dummy properties for analysis.")
        processed_aaindex_df = None
        aaindex_original_for_analysis = None

    # --- Combination Search & Evaluation Parameters ---
    min_k_value_for_search = 8
    max_k_value_for_search = 12  #12
    if max_k_value_for_search > len(all_amino_acids_names):
        max_k_value_for_search = len(all_amino_acids_names)
    if min_k_value_for_search < 2:
        min_k_value_for_search = 2
    if min_k_value_for_search > max_k_value_for_search:
        messagebox.showerror("Error",
                             f"Invalid k range: min_k ({min_k_value_for_search}) is greater than max_k ({max_k_value_for_search}). Adjust the hardcoded range.")
        return

    all_results = []
    aggregated_results_for_plots = []
    best_results_per_k = {}  # This will store the ARI-best for each k for JSON, not actual model objects
    min_distinguishable_k_value = -1
    max_distinguishable_k_value = -1
    maximally_separable_k_value = -1
    maximally_separable_combo_details: Optional[Dict[str, Any]] = None

    amino_acid_idx_map = {name: i for i, name in enumerate(all_amino_acids_names)}
    num_all_aas = len(all_amino_acids_names)
    co_assignment_matrix = np.zeros((num_all_aas, num_all_aas))
    total_pair_evaluation_counts = np.zeros((num_all_aas, num_all_aas))

    clustering_methods_to_test = ['KMedoids', 'LDA+KMeans']
    if 'KMeans' not in clustering_methods_to_test: clustering_methods_to_test.append('KMeans')
    if 'DBSCAN' not in clustering_methods_to_test: clustering_methods_to_test.append('DBSCAN')
    if 'OPTICS' not in clustering_methods_to_test: clustering_methods_to_test.append('OPTICS')
    if 'AgglomerativeClustering' not in clustering_methods_to_test: clustering_methods_to_test.append(
        'AgglomerativeClustering')
    clustering_methods_to_test = [m for m in clustering_methods_to_test if
                                  ('KMedoids' not in m or KMedoids is not None) and
                                  ('COP-KMeans' not in m or COPKMeans is not None) and
                                  ('OPTICS' not in m or OPTICS is not None)]

    print(f"\nClustering methods to be used: {', '.join(clustering_methods_to_test)}")

    # --- Main Loop: Combination Traversal, Clustering & Evaluation (with Repeats) ---
    global_base_seed_for_sampling = 42
    global_clustering_trial_counter = 0

    for combination, k_value, combo_type in combination_search_controller(
            all_amino_acids_names,
            min_k=min_k_value_for_search,
            max_k=max_k_value_for_search,
            processed_aaindex_df=processed_aaindex_df,
            base_seed_for_sampling=global_base_seed_for_sampling
    ):
        print(f"\n--- Evaluating Combination k={k_value}, Combo: {combination} (Source: {combo_type}) ---")

        global_clustering_trial_counter += 1

        y_subset_remapped, original_subset_labels_names = remap_labels_for_subset(
            y_full_encoded, all_amino_acids_names, list(combination)
        )
        X_subset = X_full_scaled.loc[y_subset_remapped.index]

        n_clusters_expected = len(original_subset_labels_names)
        if n_clusters_expected < 2 or X_subset.shape[0] < n_clusters_expected:
            print(
                f"Combination {combination} has only {n_clusters_expected} amino acids or insufficient samples ({X_subset.shape[0]}), skipping clustering.")
            continue

        true_labels_original_names_for_subset = y_subset_remapped.map(
            {idx: name for idx, name in enumerate(original_subset_labels_names)}
        )
        aa_counts_in_current_subset = true_labels_original_names_for_subset.value_counts()

        for method_name in clustering_methods_to_test:
            print(f"  Testing method: {method_name} ({N_REPEATS} repeats)")
            ari_scores_repeats = []
            silhouette_scores_repeats = []
            all_cms_repeats = []
            # Store models from one successful run for this method and combination
            current_clustering_model_instance = None
            current_dim_reduction_model_instance = None

            for repeat_idx in range(N_REPEATS):
                current_random_state = (global_clustering_trial_counter * 100 * len(clustering_methods_to_test)) + \
                                       (clustering_methods_to_test.index(method_name) * N_REPEATS) + \
                                       repeat_idx + 1

                cluster_labels, clustering_model_inst, dim_reduction_model_inst = cluster_with_method(
                    X_subset, y_subset_remapped, method_name, n_clusters_expected, random_state_val=current_random_state
                )

                if cluster_labels is None:
                    continue

                # Capture model instances from the first successful run for this combination-method pair
                if current_clustering_model_instance is None:
                    current_clustering_model_instance = clustering_model_inst
                    current_dim_reduction_model_instance = dim_reduction_model_inst

                for aa_i_name, count_i in aa_counts_in_current_subset.items():
                    idx_i = amino_acid_idx_map[aa_i_name]
                    if count_i >= 2: total_pair_evaluation_counts[idx_i, idx_i] += count_i * (count_i - 1) // 2
                    for aa_j_name, count_j in aa_counts_in_current_subset.items():
                        idx_j = amino_acid_idx_map[aa_j_name]
                        if idx_i < idx_j:
                            total_pair_evaluation_counts[idx_i, idx_j] += count_i * count_j
                            total_pair_evaluation_counts[idx_j, idx_i] += count_i * count_j

                eval_results_repeat = evaluate_clustering(
                    X_subset, y_subset_remapped, cluster_labels, combination, original_subset_labels_names
                )
                ari_scores_repeats.append(eval_results_repeat.get('ARI', 0.0))
                silhouette_scores_repeats.append(eval_results_repeat.get('Silhouette_Score', np.nan))
                all_cms_repeats.append(eval_results_repeat.get('Confusion_Matrix'))

                if cluster_labels is not None and len(np.unique(cluster_labels[cluster_labels != -1])) > 0:
                    for cluster_id in np.unique(cluster_labels[cluster_labels != -1]):
                        samples_in_cluster_indices = X_subset.index[cluster_labels == cluster_id]
                        true_aas_in_this_cluster_series = true_labels_original_names_for_subset.loc[
                            samples_in_cluster_indices]

                        aa_counts_in_cluster = true_aas_in_this_cluster_series.value_counts()
                        unique_aas_in_cluster = aa_counts_in_cluster.index.tolist()

                        for aa, count in aa_counts_in_cluster.items():
                            if count >= 2:
                                idx = amino_acid_idx_map[aa]
                                co_assignment_matrix[idx, idx] += count * (count - 1) // 2

                        for i in range(len(unique_aas_in_cluster)):
                            aa_i = unique_aas_in_cluster[i]
                            count_i = aa_counts_in_cluster[aa_i]
                            idx_i = amino_acid_idx_map[aa_i]

                            for j in range(i + 1, len(unique_aas_in_cluster)):
                                aa_j = unique_aas_in_cluster[j]
                                count_j = aa_counts_in_cluster[aa_j]
                                idx_j = amino_acid_idx_map[aa_j]

                                co_assignment_matrix[idx_i, idx_j] += count_i * count_j
                                co_assignment_matrix[idx_j, idx_i] += count_i * count_j

            if ari_scores_repeats:
                mean_ari = np.mean(ari_scores_repeats)
                std_ari = np.std(ari_scores_repeats)
                mean_silhouette = np.nanmean(silhouette_scores_repeats) if not all(
                    np.isnan(silhouette_scores_repeats)) else np.nan
                std_silhouette = np.nanstd(silhouette_scores_repeats) if not all(
                    np.isnan(silhouette_scores_repeats)) else np.nan

                # Collect clustering parameters for JSON export
                cluster_params_dict = {'n_clusters': n_clusters_expected, 'random_state': current_random_state}
                if current_clustering_model_instance:
                    # Example of capturing specific parameters
                    if method_name == 'DBSCAN':
                        cluster_params_dict['eps'] = getattr(current_clustering_model_instance, 'eps', None)
                        cluster_params_dict['min_samples'] = getattr(current_clustering_model_instance, 'min_samples',
                                                                     None)
                    elif method_name == 'OPTICS':
                        cluster_params_dict['min_samples'] = getattr(current_clustering_model_instance, 'min_samples',
                                                                     None)
                    # Add other algorithm-specific params as needed

                dim_reduction_params_dict = {}
                if current_dim_reduction_model_instance:
                    dim_reduction_params_dict['n_components'] = getattr(current_dim_reduction_model_instance,
                                                                        'n_components', None)
                    dim_reduction_params_dict['type'] = 'LDA'  # Or 'PCA', 't-SNE' if function were to return them
                    dim_reduction_params_dict['explained_variance_ratio'] = getattr(
                        current_dim_reduction_model_instance, 'explained_variance_ratio_', None)

                aggregated_results_for_plots.append({
                    'k_value': k_value,
                    'combination': combination,
                    'clustering_method': method_name,
                    'combination_type': combo_type,
                    'ARI_mean': mean_ari,
                    'ARI_std': std_ari,
                    'Silhouette_Score_mean': mean_silhouette,
                    'Silhouette_Score_std': std_silhouette
                })

                first_successful_cm = next((cm for cm in all_cms_repeats if cm is not None), None)

                all_results.append({
                    'k_value': k_value,
                    'combination': combination,
                    'clustering_method': method_name,
                    'combination_type': combo_type,
                    'ARI': mean_ari,
                    'ARI_std': std_ari,
                    'Silhouette_Score': mean_silhouette,
                    'Silhouette_Score_std': std_silhouette,
                    'Confusion_Matrix': first_successful_cm,
                    'clustering_model_params': cluster_params_dict,
                    'dim_reduction_model_params': dim_reduction_params_dict,
                    'feature_names_for_subset': X_subset.columns.tolist(),
                    'clustering_model_instance': current_clustering_model_instance,  # Store actual model instance
                    'dim_reduction_model_instance': current_dim_reduction_model_instance  # Store actual model instance
                })

                print(
                    f"    Mean ARI: {mean_ari:.4f} (Std: {std_ari:.4f}), Mean Silhouette: {mean_silhouette:.4f} (Std: {std_silhouette:.4f})")
                if not np.isnan(mean_silhouette) and mean_silhouette < SILHOUETTE_LOWER_BOUND:
                    print(
                        f"    ⚠️ Warning: Mean Silhouette Score ({mean_silhouette:.4f}) is below threshold ({SILHOUETTE_LOWER_BOUND:.1f}).")

         # Filter all_results for the current k_value and combination to find the best method for this specific combination
        current_combo_results: List[Dict[str, Any]] = []  # <-- 添加这一行进行防御性初始化
        current_combo_results = [res for res in all_results if
                                         res['k_value'] == k_value and res['combination'] == combination]

        # Update best_results_per_k (for JSON export of best per k)
        best_ari_for_combination_current_k = -1.0
        best_entry_for_k_json = None
        for res_entry in current_combo_results:
            if res_entry['ARI'] > best_ari_for_combination_current_k:
                best_ari_for_combination_current_k = res_entry['ARI']
                # Create a dict for JSON, omitting model instances
                best_entry_for_k_json = {k: v for k, v in res_entry.items() if
                                         k not in ['clustering_model_instance', 'dim_reduction_model_instance']}
                if best_entry_for_k_json['Confusion_Matrix'] is not None:
                    best_entry_for_k_json['Confusion_Matrix'] = best_entry_for_k_json[
                        'Confusion_Matrix'].tolist()  # Convert CM to list for JSON

        if best_entry_for_k_json:
            current_k_best_in_json_existing = best_results_per_k.get(k_value, {'ARI': -1.0})
            if best_ari_for_combination_current_k > current_k_best_in_json_existing['ARI']:
                best_results_per_k[k_value] = best_entry_for_k_json
                print(f"  Updated best result for k={k_value} (Mean ARI: {best_ari_for_combination_current_k:.4f}).")

        # Check for minimum stably distinguishable k
        if min_distinguishable_k_value == -1:
            for res_entry in current_combo_results:
                dynamic_stability_threshold = STABILITY_THRESHOLD_ARI_STD_BASE + 0.01 * res_entry['k_value']
                if (res_entry['ARI'] >= ARI_STOP_THRESHOLD and
                        not np.isnan(res_entry['Silhouette_Score']) and res_entry[
                            'Silhouette_Score'] >= SILHOUETTE_LOWER_BOUND and
                        res_entry['ARI_std'] <= dynamic_stability_threshold):
                    min_distinguishable_k_value = k_value
                    print(f"\nFirst combination meeting minimum stable distinction thresholds found at k={k_value}!")
                    break

        # Check for maximum stable distinguishable k
        for agg_res in aggregated_results_for_plots:
            if (agg_res['k_value'] == k_value and agg_res['combination'] == combination):
                dynamic_stability_threshold = STABILITY_THRESHOLD_ARI_STD_BASE + 0.01 * agg_res['k_value']
                if (agg_res['ARI_mean'] >= ARI_STOP_THRESHOLD and
                        not np.isnan(agg_res['Silhouette_Score_mean']) and agg_res[
                            'Silhouette_Score_mean'] >= SILHOUETTE_LOWER_BOUND and
                        agg_res['ARI_std'] <= dynamic_stability_threshold):
                    max_distinguishable_k_value = max(max_distinguishable_k_value, k_value)

        # Check for maximally separable k
        for res_entry in current_combo_results:
            if (res_entry['ARI'] >= ARI_STOP_THRESHOLD and
                    not np.isnan(res_entry['Silhouette_Score']) and res_entry[
                        'Silhouette_Score'] >= SILHOUETTE_LOWER_BOUND):

                if res_entry['k_value'] > maximally_separable_k_value:
                    maximally_separable_k_value = res_entry['k_value']
                    maximally_separable_combo_details = res_entry.copy()
                elif res_entry['k_value'] == maximally_separable_k_value:
                    if maximally_separable_combo_details is None or res_entry['ARI'] > \
                            maximally_separable_combo_details['ARI']:
                        maximally_separable_combo_details = res_entry.copy()

    print("\n--- All Combinations Evaluated ---")

    # --- 6. Results Summary and Output ---
    if not all_results:
        print("\nNo clustering evaluation results were generated.")
        return

    # Create DataFrame for export, dropping model instances
    results_df = pd.DataFrame(all_results)
    results_df_for_export = results_df.drop(
        columns=['clustering_model_instance', 'dim_reduction_model_instance', 'Confusion_Matrix'], errors='ignore')

    print("\n--- All Clustering Evaluation Results Summary (First 5 Rows) ---")
    print(results_df_for_export.head())

    print(f"\n--- Top High ARI Combinations (ARI >= 0.85) ---")
    top_high_ari_combos = results_df_for_export[
        results_df_for_export['ARI'] >= 0.85
        ].sort_values(by='ARI', ascending=False).head(10)
    print(top_high_ari_combos)

    if save_base_dir:
        results_df_for_export.to_csv(os.path.join(save_base_dir, "all_clustering_results.csv"), index=False)
        print(f"\nAll results saved to：{os.path.join(save_base_dir, 'all_clustering_results.csv')}")
        save_json_results(best_results_per_k, "best_combination_per_k_results.json",
                          save_base_dir)  # Still keeping this name as it's just 'best for each k'
    else:
        print("\nNo save directory selected. Results CSV/JSON not saved.")

    # --- Amino Acid Co-Clustering Frequency Heatmap ---
    if save_base_dir:
        print("\n--- Generating Amino Acid Co-Clustering Frequency Heatmap ---")
        confusion_frequency_df = plot_confusion_frequency_heatmap(
            co_assignment_matrix, total_pair_evaluation_counts, all_amino_acids_names,
            os.path.join(save_base_dir, "amino_acid_confusion_frequency_heatmap.png")
        )
        confusion_frequency_df.to_csv(os.path.join(save_base_dir, "amino_acid_confusion_frequency_matrix.csv"),
                                      index=True)
        print(
            f"Amino Acid Co-Clustering Frequency Matrix saved to: {os.path.join(save_base_dir, 'amino_acid_confusion_frequency_matrix.csv')}")
    else:
        print("\nNo save directory selected. Co-clustering frequency matrix not saved.")

    # --- Amino Acid Distinguishability Score ---
    if save_base_dir:
        distinguishability_df = calculate_amino_acid_distinguishability(results_df, all_amino_acids_names)
        distinguishability_df.to_csv(os.path.join(save_base_dir, "amino_acid_distinguishability_scores.csv"),
                                     index=False)
        print(
            f"Amino Acid Distinguishability Scores saved to: {os.path.join(save_base_dir, 'amino_acid_distinguishability_scores.csv')}")
    else:
        print("\nNo save directory selected. Distinguishability scores not saved.")

    # --- 7. Final Visualizations ---
    if save_base_dir:
        print("\n--- Generating Final Visualizations ---")
        aggregated_results_df_for_plots = pd.DataFrame(aggregated_results_for_plots)

        # Performance Trend Plot (ARI and Silhouette)
        plot_performance_trend(aggregated_results_df_for_plots,
                               os.path.join(save_base_dir, "performance_trend_plot.png"), ARI_STOP_THRESHOLD)
        print(f"Performance trend plot saved to：{os.path.join(save_base_dir, 'performance_trend_plot.png')}")

        # Distinction Breakpoint Plot
        mark_distinction_breakpoints(aggregated_results_df_for_plots, 'ARI_mean',
                                     os.path.join(save_base_dir, "ari_distinction_breakpoints.png"), ARI_STOP_THRESHOLD)
        print(f"Distinction breakpoint plot saved to：{os.path.join(save_base_dir, 'ari_distinction_breakpoints.png')}")

        # Clustering Stability Plot (ARI Standard Deviation)
        plot_clustering_stability(aggregated_results_df_for_plots,
                                  os.path.join(save_base_dir, "clustering_stability_ari_std.png"),
                                  STABILITY_THRESHOLD_ARI_STD_BASE)
        print(f"Clustering stability plot saved to：{os.path.join(save_base_dir, 'clustering_stability_ari_std.png')}")

        # --- Generate Visualizations for Top High ARI Combinations (ARI >= 0.85) ---
        print(f"\n--- Generating Visualizations for Top High ARI Combinations (ARI >= 0.85) ---")
        if not top_high_ari_combos.empty:
            output_base_folder_top_high_ari = os.path.join(save_base_dir, "TopHighARI_Visualizations")
            os.makedirs(output_base_folder_top_high_ari, exist_ok=True)
            print(f"Created folder: {output_base_folder_top_high_ari}")

            for i, result_row_df in top_high_ari_combos.iterrows():
                full_result_entry_for_top_ari = next(
                    (res for res in all_results if res['k_value'] == result_row_df['k_value'] and
                     res['combination'] == result_row_df['combination'] and
                     res['clustering_method'] == result_row_df['clustering_method']),
                    None
                )
                if full_result_entry_for_top_ari:
                    combo_tuple = full_result_entry_for_top_ari['combination']
                    method_name = full_result_entry_for_top_ari['clustering_method']
                    ari_score = full_result_entry_for_top_ari['ARI']
                    silhouette_score = full_result_entry_for_top_ari['Silhouette_Score']
                    k_value = full_result_entry_for_top_ari['k_value']

                    generate_combo_visualizations(
                        combo_tuple, method_name, ari_score, silhouette_score, k_value,
                        output_base_folder_top_high_ari, X_full_scaled, y_full_encoded, all_amino_acids_names,
                        confusion_frequency_df, aaindex_original_for_analysis, all_results, random_seed=100 + i
                    )
                else:
                    print(
                        f"Warning: Could not find full details for top high ARI combo {result_row_df['combination']}. Skipping its visualizations.")
        else:
            print(f"No combinations found with ARI >= 0.85. Skipping specific visualizations.")

        # --- Generate Visualizations for Maximally Separable Combination ---
        print(
            f"\n--- Generating Visualizations for Maximally Separable Combination (k={maximally_separable_k_value}) ---")
        if maximally_separable_k_value != -1 and maximally_separable_combo_details:
            combo_tuple = maximally_separable_combo_details['combination']
            method_name = maximally_separable_combo_details['clustering_method']
            ari_score = maximally_separable_combo_details['ARI']
            silhouette_score = maximally_separable_combo_details['Silhouette_Score']
            k_value = maximally_separable_combo_details['k_value']

            output_base_folder_maximally_separable = os.path.join(save_base_dir,
                                                                  f"MaximallySeparable_k{k_value}AA_Visualizations")
            os.makedirs(output_base_folder_maximally_separable, exist_ok=True)
            print(f"Created folder: {output_base_folder_maximally_separable}")

            # Generate visualizations
            generate_combo_visualizations(
                combo_tuple, method_name, ari_score, silhouette_score, k_value,
                output_base_folder_maximally_separable, X_full_scaled, y_full_encoded, all_amino_acids_names,
                confusion_frequency_df, aaindex_original_for_analysis, all_results, random_seed=200
            )

            # --- Save the Maximally Separable Model Pipeline ---
            print("\n--- Saving Maximally Separable Combination Model Pipeline ---")

            # 1. Global Scaler (from preprocess_data)
            # global_scaler_for_X is already fitted during preprocess_data

            # 2. Recreate feature engineering (This is not a scikit-learn pipeline component)
            # Need to apply the engineer_features function to raw data before scaling for prediction
            # The current pipeline uses X_full_scaled (already engineered and globally scaled) as input.
            # To create a complete pipeline from raw data, would need a custom transformer for feature engineering.
            # For simplicity, we'll assume prediction data is handled similar to X_full_scaled input.
            # So, the pipeline starts with the global scaler.

            # 3. Dimensionality Reduction Model (if LDA)
            dim_red_model_to_save = None
            if maximally_separable_combo_details['dim_reduction_model_instance']:
                dim_red_model_to_save = maximally_separable_combo_details['dim_reduction_model_instance']

            # 4. Clustering Model
            clustering_model_to_save = maximally_separable_combo_details['clustering_model_instance']

            # Construct the pipeline based on whether dim reduction was used
            pipeline_steps = []

            # The X_full_scaled is already the output of global scaling and feature engineering.
            # So, if we want a pipeline from RAW data, we need to add the feature engineering step.
            # If we assume input to the pipeline is already feature-engineered & scaled (like X_full_scaled),
            # then the pipeline starts from dim reduction / clustering.
            # Given user request to "对新数据（不同浓度的同种氨基酸）执行：相同的特征预处理 相同的特征工程转换 聚类预测",
            # this implies the full pipeline from raw data.
            # This requires converting engineer_features into a sklearn-compatible transformer.
            # For this version, let's explicitly note that the saved models assume the input `X_full_scaled` format.
            # If a full raw-to-cluster pipeline is critical, `engineer_features` needs refactoring to `BaseEstimator, TransformerMixin`.

            # For now, let's save the components that directly operate on the X_full_scaled subset.
            # This means the user would need to apply the global_scaler_for_X and engineer_features first.

            # Let's create a *dummy* pipeline for illustration, saving individual components as requested.
            # The most straightforward way to save for prediction is to save `global_scaler_for_X`
            # and then the `dim_red_model_to_save` (if any) and `clustering_model_to_save`.

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_base_filename = os.path.join(save_base_dir, f"maximally_separable_combination_model_{timestamp}")

            # Save Global Scaler
            joblib.dump(global_scaler_for_X, f"{model_base_filename}_global_scaler.pkl")
            print(f"Global StandardScaler saved as {model_base_filename}_global_scaler.pkl")

            # Save Dim Reduction Model if applicable
            dim_red_model_filename = "N/A"
            if dim_red_model_to_save:
                dim_red_model_type = type(dim_red_model_to_save).__name__
                joblib.dump(dim_red_model_to_save, f"{model_base_filename}_{dim_red_model_type.lower()}_reducer.pkl")
                dim_red_model_filename = f"{model_base_filename}_{dim_red_model_type.lower()}_reducer.pkl"
                print(f"Dimensionality Reduction Model ({dim_red_model_type}) saved as {dim_red_model_filename}")

            # Save Clustering Model
            clustering_model_type = type(clustering_model_to_save).__name__
            joblib.dump(clustering_model_to_save,
                        f"{model_base_filename}_{clustering_model_type.lower()}_clusterer.pkl")
            print(
                f"Clustering Model ({clustering_model_type}) saved as {model_base_filename}_{clustering_model_type.lower()}_clusterer.pkl")

            # Update JSON with model paths
            maximally_separable_json_data = {
                'amino_acid_list': list(combo_tuple),
                'k_value': k_value,
                'clustering_method': method_name,
                'clustering_parameters': maximally_separable_combo_details['clustering_model_params'],
                'feature_names_for_subset': maximally_separable_combo_details['feature_names_for_subset'],
                'ARI_mean': ari_score,
                'ARI_std': maximally_separable_combo_details['ARI_std'],
                'Silhouette_Score': silhouette_score,
                'global_scaler_model_path': f"{model_base_filename}_global_scaler.pkl",
                'dim_reduction_model_path': dim_red_model_filename,
                'clustering_model_path': f"{model_base_filename}_{clustering_model_type.lower()}_clusterer.pkl",
                'random_state_for_plot_generation': 200  # The seed used in generate_combo_visualizations call
            }
            save_json_results(maximally_separable_json_data, "maximally_separable_combination.json", save_base_dir)
            print(
                f"Saved maximally separable combination model metadata to: {os.path.join(save_base_dir, 'maximally_separable_combination.json')}")

        else:
            print(
                "No combination met the Maximally Separable criteria. Skipping visualizations and model saving for it.")

        # --- Generate Visualizations for Minimum Stable K Top 3 Combinations ---
        print(
            f"\n--- Generating Visualizations for Top 3 Minimum Stable K Combinations (k={max_distinguishable_k_value}) ---")
        if max_distinguishable_k_value != -1:
            stable_k_results = aggregated_results_df_for_plots[
                (aggregated_results_df_for_plots['k_value'] == max_distinguishable_k_value) &
                (aggregated_results_df_for_plots['ARI_mean'] >= ARI_STOP_THRESHOLD) &
                (aggregated_results_df_for_plots['Silhouette_Score_mean'].notna()) &
                (aggregated_results_df_for_plots['Silhouette_Score_mean'] >= SILHOUETTE_LOWER_BOUND) &
                (aggregated_results_df_for_plots.apply(
                    lambda row: row['ARI_std'] <= (STABILITY_THRESHOLD_ARI_STD_BASE + 0.01 * row['k_value']), axis=1))
                ].sort_values(by='ARI_mean', ascending=False)

            unique_stable_combinations_to_plot = []
            seen_combos_for_stable_k = set()
            for idx, row_df in stable_k_results.iterrows():
                combo_tuple = row_df['combination']
                method_name = row_df['clustering_method']
                full_result_for_plot = next(
                    (res for res in all_results if res['k_value'] == row_df['k_value'] and
                     res['combination'] == combo_tuple and res['clustering_method'] == method_name),
                    None
                )
                if full_result_for_plot:
                    if combo_tuple not in seen_combos_for_stable_k:
                        unique_stable_combinations_to_plot.append(full_result_for_plot)
                        seen_combos_for_stable_k.add(combo_tuple)
                if len(unique_stable_combinations_to_plot) >= 3:
                    break

            if unique_stable_combinations_to_plot:
                min_stable_k_folder = os.path.join(save_base_dir,
                                                   f"MinStable_k{max_distinguishable_k_value}AA_Visualizations")
                os.makedirs(min_stable_k_folder, exist_ok=True)
                print(f"Created folder: {min_stable_k_folder}")

                for i, result_dict in enumerate(unique_stable_combinations_to_plot):
                    combo_tuple = result_dict['combination']
                    method_name = result_dict['clustering_method']
                    ari_score = result_dict['ARI']
                    silhouette_score = result_dict['Silhouette_Score']
                    k_value = result_dict['k_value']

                    generate_combo_visualizations(
                        combo_tuple, method_name, ari_score, silhouette_score, k_value,
                        min_stable_k_folder, X_full_scaled, y_full_encoded, all_amino_acids_names,
                        confusion_frequency_df, aaindex_original_for_analysis, all_results, random_seed=300 + i
                    )
            else:
                print(
                    f"No unique combinations found that satisfy Minimum Stable k={max_distinguishable_k_value} criteria.")
        else:
            print("Maximum stable k value was not found, skipping visualizations for it.")

        # --- Generate Visualizations for Full Set of 18 Amino Acids ---
        print("\n--- Generating Visualizations for Full Set of 18 Amino Acids ---")
        full_set_folder = os.path.join(save_base_dir, "FullSet_18AA_Visualizations")
        os.makedirs(full_set_folder, exist_ok=True)
        print(f"Created folder: {full_set_folder}")

        full_k_value = len(all_amino_acids_names)
        full_set_combo_name = tuple(all_amino_acids_names)
        full_set_method = 'KMeans'
        full_set_random_state = 123

        print(f"Performing clustering for all {full_k_value} amino acids using {full_set_method}.")

        full_set_cluster_labels, _, _ = cluster_with_method(  # We don't need models here, just labels
            X_full_scaled, y_full_encoded, full_set_method, full_k_value, random_state_val=full_set_random_state
        )

        if full_set_cluster_labels is not None:
            full_set_eval_results = evaluate_clustering(
                X_full_scaled, y_full_encoded, full_set_cluster_labels, full_set_combo_name, all_amino_acids_names
            )
            full_set_ari = full_set_eval_results.get('ARI')
            full_set_silhouette = full_set_eval_results.get('Silhouette_Score')
            full_set_cm = full_set_eval_results.get('Confusion_Matrix')

            print(f"Full Set (18AA) Clustering Results: ARI={full_set_ari:.4f}, Silhouette={full_set_silhouette:.4f}")

            # Retrieve dummy all_results entry for full set to pass to generate_combo_visualizations
            # This is a bit of a hack since full set is not part of the main `all_results` list
            dummy_full_set_entry = {
                'k_value': full_k_value,
                'combination': full_set_combo_name,
                'clustering_method': full_set_method,
                'ARI': full_set_ari,
                'Silhouette_Score': full_set_silhouette,
                'Confusion_Matrix': full_set_cm,
                'clustering_model_params': {'n_clusters': full_k_value, 'random_state': full_set_random_state},
                'dim_reduction_model_params': {},
                'feature_names_for_subset': X_full_scaled.columns.tolist(),
                'clustering_model_instance': KMeans(n_clusters=full_k_value, random_state=full_set_random_state,
                                                    n_init='auto').fit(X_full_scaled),  # Dummy fitted model
                'dim_reduction_model_instance': None
            }
            # Temporarily add to a copy of all_results for generate_combo_visualizations
            temp_all_results_with_full_set = all_results + [dummy_full_set_entry]

            generate_combo_visualizations(
                full_set_combo_name, full_set_method, full_set_ari, full_set_silhouette, full_k_value,
                full_set_folder, X_full_scaled, y_full_encoded, all_amino_acids_names,
                confusion_frequency_df, aaindex_original_for_analysis, temp_all_results_with_full_set, random_seed=123
            )
        else:
            print(f"Clustering failed for the full set of 18 amino acids. Skipping visualizations.")

        # --- Generate PDF Report (combining various analyses) ---
        if save_base_dir:
            pdf_report_path = os.path.join(save_base_dir, "amino_acid_discrimination_report.pdf")
            print(
                f"\nGenerating comprehensive PDF report to：{pdf_report_path} (Placeholder - actual PDF generation will need a library like FPDF/ReportLab)")

            report_full_set_ari = f"{full_set_ari:.4f}" if 'full_set_ari' in locals() and full_set_ari is not None else 'N/A'
            report_full_set_silhouette = f"{full_set_silhouette:.4f}" if 'full_set_silhouette' in locals() and full_set_silhouette is not None and not np.isnan(
                full_set_silhouette) else 'N/A'
            report_full_set_method = full_set_method if 'full_set_method' in locals() else 'N/A'

            report_maximally_separable_k = maximally_separable_k_value if maximally_separable_k_value != -1 else 'Not found'
            report_maximally_separable_combo_name = maximally_separable_combo_details[
                'combination'] if maximally_separable_combo_details else 'N/A'
            report_maximally_separable_method = maximally_separable_combo_details[
                'clustering_method'] if maximally_separable_combo_details else 'N/A'
            report_maximally_separable_ari = f"{maximally_separable_combo_details['ARI']:.4f}" if maximally_separable_combo_details and not np.isnan(
                maximally_separable_combo_details['ARI']) else 'N/A'
            report_maximally_separable_silhouette = f"{maximally_separable_combo_details['Silhouette_Score']:.4f}" if maximally_separable_combo_details and not np.isnan(
                maximally_separable_combo_details['Silhouette_Score']) else 'N/A'
            report_top_confused_df_str = "No significant confused pairs or properties data for the best combination."  # <-- 这一行应该存在并赋初值

            report_top_high_ari_summary = top_high_ari_combos.to_string(
                index=False) if not top_high_ari_combos.empty else "No combinations met the ARI >= 0.85 criteria."

            report_content = f"""
# Amino Acid Discrimination Analysis Report

## 1. Executive Summary
- **目标:** 识别基于DNA包裹SWNT阵列的最大可区分氨基酸组合。
- **最小稳定可区分 k (基于 ARI >= {ARI_STOP_THRESHOLD} & Silhouette >= {SILHOUETTE_LOWER_BOUND} & 动态 ARI 标准差 <= ({STABILITY_THRESHOLD_ARI_STD_BASE} + 0.01*k)):** {max_distinguishable_k_value if max_distinguishable_k_value != -1 else '在测试范围内未找到'}
- **最大可区分 k (基于 ARI >= {ARI_STOP_THRESHOLD} & Silhouette >= {SILHOUETTE_LOWER_BOUND}):** {report_maximally_separable_k}
    - **对应组合:** {report_maximally_separable_combo_name}
    - **方法:** {report_maximally_separable_method}
    - **ARI:** {report_maximally_separable_ari}
    - **Silhouette:** {report_maximally_separable_silhouette}
- **注意:** 分析在指定 k 值范围 ({min_k_value_for_search} 到 {max_k_value_for_search}) 内执行。

## 2. 数据概览与预处理
- **输入数据:** {input_file_path}
- **缺失值处理:** 丢弃了超过 {0.2 * 100}% 特征缺失的样本。
- **归一化:** 应用了Z-score标准化。
- **特征工程:** 应用了强度比率、位移差、强度/位移比率、响应归一化。
- **物理化学性质添加:** 真实的AAindex性质 (PCA降维至 {AAINDEX_FEATURES_TO_SELECT_COUNT} 个特征用于组合搜索)。
- **预处理后总样本数:** {X_full_scaled.shape[0]}
- **总唯一氨基酸数:** {len(all_amino_acids_names)}

## 3. 组合搜索策略
- **探索的 k 范围:** {min_k_value_for_search} 到 {max_k_value_for_search}
- **每个 k 的组合采样数量:** 分层策略 (k<=5: min(200, 总数); k<=10: min(5000, 总数); 否则: min(10000, 总数))。
- **采样方法:** 80% 基于物理化学性质差异 (AAindex PCA空间中的欧氏距离)，20% 随机采样。
- 所有采样均使用可复现的随机种子。

## 4. 聚类性能概览
- **评估的聚类方法:** {', '.join(clustering_methods_to_test)} (每种方法重复 {N_REPEATS} 次，记录均值和标准差)。
- **质量标准:** ARI >= {ARI_STOP_THRESHOLD} 并且 Silhouette >= {SILHOUETTE_LOWER_BOUND}。
- **综合排名前 {len(top_high_ari_combos)} 的高 ARI 组合 (ARI >= 0.85):**
{report_top_high_ari_summary}

## 5. 性能趋势分析
- **ARI 趋势:** 请参阅 `performance_trend_plot.png`。显示了随着 k 增加，各方法平均 ARI 的趋势。
- **Silhouette 趋势:** 请参阅 `performance_trend_plot.png`。显示了随着 k 增加，各方法平均 Silhouette 分数的趋势。
- **区分断点:** 请参阅 `ari_distinction_breakpoints.png`。标记了平均 ARI 首次超过 {ARI_STOP_THRESHOLD} 的 k 值。
- **聚类稳定性:** 请参阅 `clustering_stability_ari_std.png`。显示了各方法 ARI 在重复运行中的标准差，指示不稳定性。ARI 标准差大于 ({STABILITY_THRESHOLD_ARI_STD_BASE} + 0.01*k) 的组合/方法被认为不稳定。

## 6. 最大可区分组合详细分析
- **组合:** {report_maximally_separable_combo_name}
- **聚类方法:** {report_maximally_separable_method}
- **平均 ARI:** {report_maximally_separable_ari}
- **平均 Silhouette 分数:** {report_maximally_separable_silhouette}
- **保存的模型文件:**
    - 全局标准化器: `results/maximally_separable_combination_model_<timestamp>_global_scaler.pkl`
    - 维度降低模型 (如适用): `results/maximally_separable_combination_model_<timestamp>_<dim_red_type>_reducer.pkl`
    - 聚类模型: `results/maximally_separable_combination_model_<timestamp>_<clusterer_type>_clusterer.pkl`
- **模型元数据:** 详细信息已保存至 `results/maximally_separable_combination.json`。

### 6.1 聚类可视化
- 请参阅 `MaximallySeparable_k{report_maximally_separable_k}AA_Visualizations/...` 文件夹中的散点图，显示了真实标签 (颜色) 与预测聚类 (样式)，并带有置信椭圆。

### 6.2 混淆分析与物理化学性质
- 请参阅 `MaximallySeparable_k{report_maximally_separable_k}AA_Visualizations/...` 文件夹中的混淆矩阵。
- **最混淆的 {5} 对及性质差异:**
    {report_top_confused_df_str}

## 7. 物理化学性质分析
- **氨基酸性质 PCA:** 请参阅 `amino_acid_properties_pca.png`。通过AAindex特征的PCA可视化了氨基酸在性质空间中的分布。
- **特征重要性分析:** 请参阅 `feature_importance.png`。显示了特征的互信息分数，指示它们对氨基酸区分的贡献。
- **氨基酸共聚类频率矩阵:** 请参阅 `amino_acid_confusion_frequency_heatmap.png`。该热力图可视化了在所有评估的组合和方法中，任意两个氨基酸被分配到同一聚类中的频率。值越高表示共聚类 (混淆) 越频繁。
- **氨基酸可区分性分数:** 请参阅 `amino_acid_distinguishability_scores.csv`。为每个氨基酸提供了一个量化分数，计算为其参与的所有组合中的平均 ARI。分数越高表示可区分性越好。

## 8. 全集 (18 氨基酸) 基线分析
- **方法:** {report_full_set_method}
- **ARI:** {report_full_set_ari}
- **Silhouette:** {report_full_set_silhouette}
- 全套 18 氨基酸的可视化可在 `FullSet_18AA_Visualizations/` 中找到。

## 9. 特殊关注氨基酸
- **带电氨基酸:** 天冬氨酸 (Asp), 谷氨酸 (Glu), 赖氨酸 (Lys), 精氨酸 (Arg)
- **芳香族氨基酸:** 苯丙氨酸 (Phe), 酪氨酸 (Tyr), 色氨酸 (Trp)
- *通过检查混淆频率矩阵和单个混淆矩阵，可以对这些组之间的系统性混淆模式进行进一步分析。*

## 10. 结论与下一步
- 分析结果总结。
- 对未来工作的建议 (例如，探索更高级的聚类算法、特征工程、实时预测、整合领域知识进行性质选择)。

本报告旨在整合所有关键发现和可视化结果。
"""
            with open(pdf_report_path.replace(".pdf", ".md"), 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(
                f"Comprehensive report (Markdown format for content) saved to：{pdf_report_path.replace('.pdf', '.md')}")
            print(
                "Note: Direct PDF generation with plots requires additional libraries (e.g., FPDF, ReportLab) and explicit code to embed images. This markdown file outlines the report structure.")

        else:
            print("\nNo save directory selected. Final visualizations and reports not saved.")

    print("\n--- System Execution Complete ---")
    if min_distinguishable_k_value != -1:
        print(f"最小稳定可区分氨基酸组合大小 (k)：{min_distinguishable_k_value}")
    else:
        print("在测试范围内未找到满足 ARI 和 Silhouette 阈值的组合。")

    if max_distinguishable_k_value != -1:
        print(f"最大稳定可区分氨基酸组合大小 (k)：{max_distinguishable_k_value}")
    else:
        print("在测试范围内未找到满足 ARI、Silhouette 和稳定性阈值的组合。")

    if maximally_separable_k_value != -1:
        print(
            f"最大可区分氨基酸组合大小 (k, ARI >= {ARI_STOP_THRESHOLD}, Silhouette >= {SILHOUETTE_LOWER_BOUND})：{maximally_separable_k_value}")
        print(
            f"  对应组合: {maximally_separable_combo_details['combination']} 使用方法: {maximally_separable_combo_details['clustering_method']}")
    else:
        print(
            f"在测试范围内未找到满足最大可区分性标准 (ARI >= {ARI_STOP_THRESHOLD}, Silhouette >= {SILHOUETTE_LOWER_BOUND}) 的组合。")


if __name__ == '__main__':
    main(continue_after_threshold=True)
