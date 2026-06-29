import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, calinski_harabasz_score, \
    davies_bouldin_score, normalized_mutual_info_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # Explicitly import LDA
from sklearn.feature_selection import mutual_info_classif  # For feature importance
import os
import sys
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import itertools
import random
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from typing import Tuple, List, Dict, Any, Generator, Optional
from warnings import filterwarnings
from matplotlib.patches import Ellipse  # For confidence ellipses
import json  # For JSON output

# Suppress all warnings for cleaner output
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=DeprecationWarning)

# --- Constants for Configuration ---
N_REPEATS = 5  # Number of times to repeat clustering for robustness
AAINDEX_FEATURES_TO_SELECT_COUNT = 8  # Target number of AAindex features after PCA/selection
ARI_STOP_THRESHOLD = 0.85
SILHOUETTE_LOWER_BOUND = 0.5
MAX_K_VALUE_EXPLORE = 10  # Max k to evaluate for trend plots (as per request)

# --- User-provided Tkinter and UMAP import/fallback logic ---
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

# Attempt to import UMAP; exit if it fails.
try:
    import umap.umap_ as umap
except ImportError:
    print("UMAP is not installed. Please install it: pip install umap-learn")
    sys.exit(1)

# Attempt to import sklearn_extra for KMedoids
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    print(
        "sklearn-extra not installed. KMedoids will not be available. Please install it: pip install scikit-learn-extra")
    KMedoids = None

# Attempt to import COP-KMeans (needs ccp-kmeans)
try:
    from ccp_kmeans import COPKMeans
except ImportError:
    print(
        "COPKMeans library not installed. COP-KMeans will not be available. Please install it: pip install ccp-kmeans")
    COPKMeans = None

# Attempt to import OPTICS (part of sklearn.cluster)
try:
    from sklearn.cluster import OPTICS
except ImportError:
    print("OPTICS (sklearn.cluster.OPTICS) not available. Please ensure you have a recent scikit-learn version.")
    OPTICS = None

# Set Matplotlib default font to support English and ensure Arial and other English fonts are available
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


# --- Physicochemical Properties PCA Function (defined early for visibility) ---
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

    pca = PCA(n_components=2, random_state=42)
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
Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Preprocesses the data: handles missing values, standardizes features, and encodes labels.
    Args:
        df (pd.DataFrame): Raw input data.
        feature_cols (List[str]): List of column names to be used as features.
        label_col (str): Column name containing amino acid labels.
        missing_threshold (float): Percentage of missing features allowed (e.g., 0.2 for 20%).
    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str]]: Scaled features, encoded labels, and original unique label names.
    Raises:
        ValueError: If essential columns are missing.
    """
    # Check for missing essential columns
    required_cols = [label_col] + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in the CSV: {', '.join(missing_cols)}")

    # Drop rows where 'AA' label is missing
    df.dropna(subset=[label_col], inplace=True)

    # Calculate missing percentage per row for features and drop if exceeding threshold
    initial_rows = df.shape[0]
    df_features_numeric = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df_filtered_missing = df[df_features_numeric.isnull().sum(axis=1) / len(feature_cols) <= missing_threshold]
    rows_dropped_missing = initial_rows - df_filtered_missing.shape[0]
    if rows_dropped_missing > 0:
        print(f"Dropped {rows_dropped_missing} samples due to more than {missing_threshold * 100}% missing features.")
    df = df_filtered_missing.copy()

    if df.empty:
        raise ValueError("DataFrame is empty after dropping missing values.")

    X = df[feature_cols].copy()
    y_original_labels = df[label_col].copy()

    X.fillna(0, inplace=True)  # Replace NaNs with 0, or consider other imputation strategies

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y_original_labels), index=y_original_labels.index)
    unique_original_labels = list(label_encoder.classes_)

    if len(unique_original_labels) < 2:
        raise ValueError("Processed data contains fewer than 2 unique amino acids. Cannot perform clustering.")

    return X_scaled, y_encoded, unique_original_labels


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


def load_aaindex_properties(filepath: str, n_components_pca: int = AAINDEX_FEATURES_TO_SELECT_COUNT) -> pd.DataFrame:
    """
    Loads AAindex data, performs PCA for dimensionality reduction, and returns selected components.
    Assumes AAindex CSV has 'Amino_Acid' as index and remaining columns are properties.
    Args:
        filepath (str): Path to the AAindex CSV file.
        n_components_pca (int): Number of principal components to select.
    Returns:
        pd.DataFrame: DataFrame with selected principal components as features for each amino acid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"AAindex file not found: {filepath}. Please download and place it.")

    aaindex_df = pd.read_csv(filepath, index_col='Amino_Acid')

    # Drop any non-numeric columns if present (e.g., descriptions)
    numeric_cols = aaindex_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric properties found in AAindex file.")

    aaindex_numeric = aaindex_df[numeric_cols]

    # Drop columns with all NaNs or constant values (zero variance)
    aaindex_numeric = aaindex_numeric.dropna(axis=1, how='all')
    aaindex_numeric = aaindex_numeric.loc[:, (aaindex_numeric != aaindex_numeric.iloc[0]).any()]

    if aaindex_numeric.empty or aaindex_numeric.shape[1] < n_components_pca:
        print(
            "Warning: AAindex data has too few features or is empty after cleaning for PCA. Using all available features or fewer components.")
        n_components_pca = min(n_components_pca, aaindex_numeric.shape[1])
        if n_components_pca < 1:
            raise ValueError("No valid numeric features remaining in AAindex for PCA.")

    scaler = StandardScaler()
    aaindex_scaled = pd.DataFrame(scaler.fit_transform(aaindex_numeric),
                                  columns=aaindex_numeric.columns,
                                  index=aaindex_numeric.index)

    pca = PCA(n_components=n_components_pca, random_state=42)
    aaindex_pca_features = pca.fit_transform(aaindex_scaled)

    # Create new DataFrame with PCA components
    pca_cols = [f'AAindex_PC{i + 1}' for i in range(aaindex_pca_features.shape[1])]
    aaindex_processed_df = pd.DataFrame(aaindex_pca_features, columns=pca_cols, index=aaindex_scaled.index)

    print(
        f"AAindex loaded. Reduced to {aaindex_processed_df.shape[1]} features (explaining {pca.explained_variance_ratio_.sum():.2f} variance).")
    return aaindex_processed_df


def engineer_features(df_original_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generates new features based on original intensity and shift data.
    Assumes original features have columns like '(X,Y)_intensity' and '(X,Y)_shift'.
    Args:
        df_original_features (pd.DataFrame): DataFrame containing original (unscaled) features.
    Returns:
        pd.DataFrame: DataFrame with original and engineered features, scaled.
    """
    df_engineered = df_original_features.copy()
    chiralities = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']

    # Intensity ratios: intensity_i / intensity_j
    for i in range(len(chiralities)):
        for j in range(i + 1, len(chiralities)):
            c1, c2 = chiralities[i], chiralities[j]
            col1 = f'{c1}_intensity'
            col2 = f'{c2}_intensity'
            if col1 in df_engineered.columns and col2 in df_engineered.columns:
                # Handle division by zero
                df_engineered[f'{c1}I_div_{c2}I'] = df_engineered[col1] / df_engineered[col2].replace(0, np.nan)
                df_engineered[f'{c2}I_div_{c1}I'] = df_engineered[col2] / df_engineered[col1].replace(0, np.nan)

    # Shift differences: shift_i - shift_j
    for i in range(len(chiralities)):
        for j in range(i + 1, len(chiralities)):
            c1, c2 = chiralities[i], chiralities[j]
            col1 = f'{c1}_shift'
            col2 = f'{c2}_shift'
            if col1 in df_engineered.columns and col2 in df_engineered.columns:
                df_engineered[f'{c1}S_minus_{c2}S'] = df_engineered[col1] - df_engineered[col2]

    # Intensity/shift ratio for specific nanotubes
    for c in chiralities:
        intensity_col = f'{c}_intensity'
        shift_col = f'{c}_shift'
        if intensity_col in df_engineered.columns and shift_col in df_engineered.columns:
            # Handle division by zero for shift (if shift can be zero)
            df_engineered[f'{c}I_div_{c}S'] = df_engineered[intensity_col] / df_engineered[shift_col].replace(0, np.nan)

    # Response normalization (per sample, across all intensity features)
    intensity_cols = [f'{c}_intensity' for c in chiralities if f'{c}_intensity' in df_engineered.columns]
    if intensity_cols:
        max_intensity_per_row = df_engineered[intensity_cols].max(axis=1)
        for col in intensity_cols:
            # Handle division by zero for max_intensity_per_row if it's zero
            df_engineered[f'{col}_norm'] = df_engineered[col] / max_intensity_per_row.replace(0, np.nan)

    # Fill NaNs created by division (e.g., 0/0 or X/0) with 0 or a sensible value.
    # For ratios, NaN is usually appropriate if the denominator was 0. Replacing with 0 means no signal.
    df_engineered.fillna(0, inplace=True)

    # Standardize all engineered features along with original features
    # This scaler should be fitted on the combined original+engineered features.
    scaler = StandardScaler()
    X_engineered_scaled = pd.DataFrame(scaler.fit_transform(df_engineered), columns=df_engineered.columns,
                                       index=df_engineered.index)

    print(f"Engineered features added. Total features: {X_engineered_scaled.shape[1]}")
    return X_engineered_scaled


# --- 2. Combination Search Strategy Functions ---
def generate_combinations(items: List[str], k: int) -> Generator[Tuple[str, ...], None, None]:
    """Generates C(n, k) combinations from a list of items."""
    return itertools.combinations(items, k)


def sample_combinations(combinations_generator: Generator[Tuple[str, ...], None, None], max_combinations: int = 50) -> \
List[Tuple[str, ...]]:
    """Samples combinations if the total count exceeds max_combinations."""
    all_combinations = list(combinations_generator)
    if len(all_combinations) > max_combinations:
        return random.sample(all_combinations, max_combinations)
    return all_combinations


def prioritize_combinations_by_properties(
        amino_acids: List[str],
        k: int,
        properties_df: pd.DataFrame,  # Now this is the processed AAindex df
        max_combinations: int = 50
) -> List[Tuple[str, ...]]:
    """
    Prioritizes combinations based on physicochemical property differences.
    Calculates average Euclidean distance in the selected AAindex feature space.
    """
    all_combinations = list(itertools.combinations(amino_acids, k))

    if len(all_combinations) <= max_combinations:
        return all_combinations

    # Ensure properties_df contains only amino acids relevant to this subset
    relevant_properties_df = properties_df.loc[properties_df.index.isin(amino_acids)]

    if relevant_properties_df.empty or relevant_properties_df.shape[0] < k:
        print(
            "Warning: Insufficient physicochemical property data for prioritization. Falling back to random sampling.")
        return random.sample(all_combinations, max_combinations)

    combination_scores = []
    for combo in all_combinations:
        combo_properties = relevant_properties_df.loc[list(combo)]

        if combo_properties.shape[0] < 2:  # Cannot calculate distance for single amino acid
            avg_distance = 0.0
        else:
            distances = pdist(combo_properties,
                              metric='euclidean')  # Already using PCA features, no need to scale again
            avg_distance = distances.mean()
        combination_scores.append((combo, avg_distance))

    # Sort by average distance (descending) to get combinations with highest property differences
    combination_scores.sort(key=lambda x: x[1], reverse=True)

    return [combo for combo, score in combination_scores[:max_combinations]]


def combination_search_controller(
        all_amino_acids: List[str],
        max_k: int,
        min_k: int = 2,
        max_combinations_per_k: int = 50,
        properties_df: Optional[pd.DataFrame] = None  # Pass the processed properties DataFrame directly
) -> Generator[Tuple[Tuple[str, ...], int, str], None, None]:
    """
    Controls amino acid combination generation and selection.
    Yields: (combination_tuple, k_value, combination_type_str)
    """
    use_property_prioritization = False
    if properties_df is not None and not properties_df.empty:
        # Check if properties_df contains enough unique amino acids to align with all_amino_acids
        if all(aa in properties_df.index for aa in all_amino_acids):
            use_property_prioritization = True
            print("Using property-based prioritization for combination search.")
        else:
            print("Warning: Properties data does not cover all amino acids. Falling back to random sampling.")
    else:
        print("No amino acid properties data provided or data is empty. Will use random sampling.")

    for k in range(min_k, max_k + 1):
        print(f"Generating k={k} combinations...")

        if use_property_prioritization:
            combinations = prioritize_combinations_by_properties(all_amino_acids, k, properties_df,
                                                                 max_combinations=max_combinations_per_k)
            combo_type = "property_based"
        else:
            combinations_generator = generate_combinations(all_amino_acids, k)
            combinations = sample_combinations(combinations_generator, max_combinations=max_combinations_per_k)
            combo_type = "random"

        for combo in combinations:
            yield combo, k, combo_type


# --- 3. Dual-Mode Clustering Evaluation System Functions ---
def _auto_constraint_generation(X: pd.DataFrame, y_true: pd.Series, num_cannot_links_per_class_pair: int = 5) -> Tuple[
    List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Automatically generates must-link and cannot-link constraints for COP-KMeans.
    must-link: All pairs within the same true class.
    cannot-link: Random pairs from different true classes that are relatively close in feature space.
    Args:
        X (pd.DataFrame): Feature data.
        y_true (pd.Series): True labels (numeric encoded).
        num_cannot_links_per_class_pair (int): Number of cannot-link constraints to generate for each pair of different classes.
    Returns:
        Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: Lists of must-link and cannot-link pairs (indices).
    """
    must_links = []
    cannot_links = []

    labels_unique = y_true.unique()
    label_to_indices = {label: list(y_true[y_true == label].index) for label in labels_unique}

    # Generate must-links: all pairs within the same true class
    for label in labels_unique:
        indices = label_to_indices[label]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                must_links.append((indices[i], indices[j]))

    # Generate cannot-links: random pairs from different true classes
    # Focus on pairs that might be "confused" by selecting closest points from different classes.
    if len(labels_unique) > 1:
        for i, label1 in enumerate(labels_unique):
            for j, label2 in enumerate(labels_unique):
                if label1 >= label2:  # Avoid duplicate pairs and self-pairs
                    continue

                indices1 = label_to_indices[label1]
                indices2 = label_to_indices[label2]

                if not indices1 or not indices2:
                    continue

                # Find a few close pairs between these two classes for cannot-links
                num_generated = 0
                max_attempts = num_cannot_links_per_class_pair * 10  # Avoid infinite loop
                attempts = 0
                while num_generated < num_cannot_links_per_class_pair and attempts < max_attempts:
                    idx_a = random.choice(indices1)
                    idx_b = random.choice(indices2)

                    if (idx_a, idx_b) not in cannot_links and (idx_b, idx_a) not in cannot_links:
                        cannot_links.append((idx_a, idx_b))
                        num_generated += 1
                    attempts += 1

    print(f"Generated {len(must_links)} must-links and {len(cannot_links)} cannot-links for COP-KMeans.")
    return must_links, cannot_links


def cluster_with_method(
        X: pd.DataFrame,
        y_true: pd.Series,  # y_true here is encoded numeric labels
        method: str,
        n_clusters: int,
        random_state_val: int,  # Explicit random state for reproducibility
        **kwargs
) -> Tuple[Optional[np.ndarray], Any]:
    """
    Unified clustering interface, handles various algorithms and their specific requirements.
    Returns (cluster_labels, model_object) or (None, None) if clustering fails.
    """
    cluster_labels = None
    model = None

    if X.shape[0] == 0:
        print(f"Error: No samples in data for {method} clustering.")
        return None, None

    if n_clusters <= 0 and method not in ['DBSCAN', 'OPTICS']:  # DBSCAN/OPTICS don't need n_clusters upfront
        print(f"Error: Number of clusters n_clusters must be positive for {method}, current value: {n_clusters}.")
        return None, None
    if X.shape[0] < n_clusters and method not in ['DBSCAN', 'OPTICS', 'LDA+KMeans']:
        print(
            f"Warning: Number of samples ({X.shape[0]}) less than n_clusters ({n_clusters}) for {method}. Cannot perform effective clustering.")
        return None, None

    try:
        if method == 'KMeans':
            model = KMeans(n_clusters=n_clusters, random_state=random_state_val, n_init='auto', **kwargs)
            cluster_labels = model.fit_predict(X)
            # Check for empty clusters in KMeans
            if len(np.unique(cluster_labels)) < n_clusters:
                print(
                    f"Warning: KMeans found fewer clusters ({len(np.unique(cluster_labels))}) than expected ({n_clusters}).")

        elif method == 'KMedoids':
            if KMedoids is None: print("KMedoids not available."); return None, None
            model = KMedoids(n_clusters=n_clusters, random_state=random_state_val, metric='euclidean', **kwargs)
            cluster_labels = model.fit_predict(X)
            if len(np.unique(cluster_labels)) < n_clusters:
                print(
                    f"Warning: KMedoids found fewer clusters ({len(np.unique(cluster_labels))}) than expected ({n_clusters}).")

        elif method == 'LDA+KMeans':
            unique_true_labels = y_true.unique()
            if len(unique_true_labels) < 2:
                print("LDA+KMeans Error: Fewer than 2 unique true labels, cannot perform LDA dimensionality reduction.")
                return None, None

            n_components_lda = min(n_clusters - 1, len(unique_true_labels) - 1, X.shape[1])
            if n_components_lda <= 0:
                print("LDA dimensionality reduction will result in non-positive dimensions, skipping LDA+KMeans.")
                return None, None
            if X.shape[0] < len(unique_true_labels):
                print(
                    f"LDA+KMeans Warning: Number of samples ({X.shape[0]}) less than number of original classes ({len(unique_true_labels)}). LDA might be unstable.")

            try:
                lda = LDA(n_components=n_components_lda, **kwargs)
                X_lda = lda.fit_transform(X, y_true)
            except Exception as e:
                print(
                    f"LDA failed (e.g., singular matrix or too few samples/features for classes): {e}. Skipping LDA+KMeans.")
                return None, None

            model = KMeans(n_clusters=n_clusters, random_state=random_state_val, n_init='auto', **kwargs)
            cluster_labels = model.fit_predict(X_lda)
            if len(np.unique(cluster_labels)) < n_clusters:
                print(
                    f"Warning: LDA+KMeans found fewer clusters ({len(np.unique(cluster_labels))}) than expected ({n_clusters}).")

        elif method == 'COP-KMeans':
            if COPKMeans is None: print("COP-KMeans not available."); return None, None
            must_links, cannot_links = _auto_constraint_generation(X, y_true)
            model = COPKMeans(n_clusters=n_clusters, random_state=random_state_val, **kwargs)
            cluster_labels = model.fit_predict(X.values, ml=must_links, cl=cannot_links)
            if len(np.unique(cluster_labels)) < n_clusters:
                print(
                    f"Warning: COP-KMeans found fewer clusters ({len(np.unique(cluster_labels))}) than expected ({n_clusters}).")

        elif method == 'DBSCAN':
            if X.shape[0] < 2: print("DBSCAN: Not enough samples."); return None, None
            min_samples = kwargs.get('min_samples', max(3, X.shape[1] * 2))  # Default min_samples

            # Estimate eps using NearestNeighbors, find average distance to k-th neighbor
            nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
            distances, _ = nn.kneighbors(X)
            estimated_eps = np.mean(distances[:, min_samples - 1])
            eps = kwargs.get('eps', estimated_eps * 1.5)  # A heuristic multiplier
            eps = max(0.01, eps)  # Ensure eps is not too small

            print(f"DBSCAN: Using estimated eps={eps:.4f}, min_samples={min_samples}")
            model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
            cluster_labels = model.fit_predict(X)

            if len(np.unique(cluster_labels[
                                 cluster_labels != -1])) <= 1:  # Only noise or only one cluster found (excluding noise)
                print("DBSCAN result: Only noise points or a single cluster found. Skipping evaluation.")
                return None, None

        elif method == 'OPTICS':
            if OPTICS is None: print("OPTICS not available."); return None, None
            if X.shape[0] < 2: print("OPTICS: Not enough samples."); return None, None
            min_samples = kwargs.get('min_samples', max(3, X.shape[1] * 2))  # Default min_samples (3-5 for user)

            print(f"OPTICS: Using min_samples={min_samples}.")
            model = OPTICS(min_samples=min_samples, **kwargs)
            cluster_labels = model.fit_predict(X)

            if len(np.unique(cluster_labels[
                                 cluster_labels != -1])) <= 1:  # Only noise or only one cluster found (excluding noise)
                print("OPTICS result: Only noise points or a single cluster found. Skipping evaluation.")
                return None, None

        elif method == 'AgglomerativeClustering':
            model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
            cluster_labels = model.fit_predict(X)
            if len(np.unique(cluster_labels)) < n_clusters:
                print(
                    f"Warning: AgglomerativeClustering found fewer clusters ({len(np.unique(cluster_labels))}) than expected ({n_clusters}).")
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    except Exception as e:
        print(f"Clustering method {method} failed: {e}. Returning None labels.")
        return None, None

    return cluster_labels, model


# --- 4. Evaluation Metrics Functions ---
def evaluate_clustering(
        X: pd.DataFrame,
        y_true: pd.Series,  # Encoded numeric true labels
        y_pred: np.ndarray,  # Predicted cluster labels
        combo_name: Tuple[str, ...],
        original_subset_labels_names: List[str]  # Original names for confusion matrix
) -> Dict[str, Any]:
    """
    Evaluates clustering results using ARI, Silhouette Score, and Confusion Matrix.
    """
    results = {
        'combination': combo_name
    }

    # Adjusted Rand Index (ARI)
    if len(y_true.unique()) > 1 and len(np.unique(y_pred)) > 1:
        ari = adjusted_rand_score(y_true, y_pred)
        results['ARI'] = ari
    else:
        results['ARI'] = 0.0

    # Silhouette Score
    if X.shape[0] > 1 and len(np.unique(y_pred)) > 1:
        try:
            silhouette = silhouette_score(X, y_pred)
            results['Silhouette_Score'] = silhouette
        except Exception as e:
            results['Silhouette_Score'] = np.nan
            print(f"Error calculating Silhouette Score: {e}")
    else:
        results['Silhouette_Score'] = np.nan

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    results['Confusion_Matrix'] = cm

    return results


# --- 5. Visualization Output Functions ---
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

    # Plot ARI trend
    sns.lineplot(data=results_df, x='k_value', y='ARI_mean', marker='o', hue='clustering_method', ax=ax1, linewidth=2)
    ax1.set_xlabel("Number of Amino Acids (k)", fontsize=12)
    ax1.set_ylabel("Adjusted Rand Index (ARI)", color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(y_axis_limit)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Silhouette on a secondary Y-axis
    ax2 = ax1.twinx()
    sns.lineplot(data=results_df, x='k_value', y='Silhouette_Score_mean', marker='x', hue='clustering_method', ax=ax2,
                 linestyle='--', alpha=0.7, linewidth=2, legend=False)
    ax2.set_ylabel("Silhouette Score", color='tab:orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(y_axis_limit)

    # Add ARI threshold line
    ax1.axhline(y=ari_threshold, color='red', linestyle=':', label=f'ARI Threshold ({ari_threshold})')

    # Mark distinction breakpoint (overall mean ARI)
    # Average across methods for each k to find breakpoint
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

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    # Create dummy handle for Silhouette legend
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


def plot_dual_encoded_clusters(
        X: pd.DataFrame,
        y_true_encoded: pd.Series,
        y_pred: np.ndarray,
        combo_name: Tuple[str, ...],
        filename: str,
        true_label_names: List[str]
):
    """
    Plots 2D clustering visualization with true labels as color and predicted clusters as style.
    Prioritizes LDA for dimensionality reduction, falls back to PCA, then t-SNE. Adds confidence ellipses.
    """
    if X.shape[0] == 0:
        print("Warning: No sample data, cannot plot dual-encoded clusters.")
        return

    X_reduced = None
    dim_reduction_method = "Original Features"

    if X.shape[1] > 2:
        # Priority 1: LDA for dimensionality reduction
        unique_true_labels_count = len(y_true_encoded.unique())
        if unique_true_labels_count > 1 and X.shape[0] >= unique_true_labels_count and X.shape[
            1] >= unique_true_labels_count - 1:
            try:
                lda = LDA(n_components=min(2, unique_true_labels_count - 1, X.shape[1]), random_state=42)
                X_reduced_lda = lda.fit_transform(X, y_true_encoded)
                if X_reduced_lda.shape[1] == 2:
                    X_reduced = X_reduced_lda
                    dim_reduction_method = f"LDA (Variance Ratio: {lda.explained_variance_ratio_.sum():.2f})"
                    print(f"Using LDA for visualization.")
            except Exception as e:
                print(f"LDA failed for visualization ({e}). Falling back to PCA/t-SNE.")
                X_reduced = None

        # Priority 2: PCA if LDA failed or not applicable, and features > 2
        if X_reduced is None:
            pca = PCA(n_components=2, random_state=42)
            X_reduced_pca = pca.fit_transform(X)
            if pca.explained_variance_ratio_.sum() >= 0.7:
                X_reduced = X_reduced_pca
                dim_reduction_method = f"PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})"
                print(f"Using PCA for visualization.")
            else:
                # Priority 3: t-SNE if PCA explanation is low
                print("PCA explained variance is low. Falling back to t-SNE for visualization.")
                try:
                    if X.shape[0] > 1:
                        perplexity_val = min(5, X.shape[0] - 1) if X.shape[0] > 1 else 1
                        if perplexity_val < 1:
                            print("Insufficient samples for t-SNE. Skipping dual-encoded plot.")
                            return
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
                        X_reduced = tsne.fit_transform(X)
                        dim_reduction_method = "t-SNE"
                        print("Using t-SNE for visualization.")
                    else:
                        print("Insufficient samples for t-SNE. Skipping dual-encoded plot.")
                        return
                except Exception as e:
                    print(f"t-SNE failed for visualization ({e}). Skipping dual-encoded plot.")
                    return
    else:  # If features <= 2, use original features
        X_reduced = X.values
        dim_reduction_method = "Original Features"

    if X_reduced is None or X_reduced.shape[0] == 0:
        print("Failed to perform dimensionality reduction for visualization. Skipping dual-encoded plot.")
        return

    plt.figure(figsize=(10, 8))

    plot_df = pd.DataFrame(X_reduced, columns=['Dim1', 'Dim2'])
    plot_df['True_Label_Encoded'] = y_true_encoded.values
    plot_df['Cluster_Label'] = y_pred

    unique_true_encoded_labels = sorted(y_true_encoded.unique())
    true_label_name_map = {encoded_val: true_label_names[idx] for idx, encoded_val in
                           enumerate(unique_true_encoded_labels)}
    plot_df['True_Label_Name'] = plot_df['True_Label_Encoded'].map(true_label_name_map)

    sns.scatterplot(
        data=plot_df,
        x='Dim1',
        y='Dim2',
        hue='True_Label_Name',
        style='Cluster_Label',
        palette='tab10',
        size=80,
        sizes=(80, 80),
        alpha=0.7,
        linewidth=0,
        legend='full'
    )

    for true_label_name in plot_df['True_Label_Name'].unique():
        subset_df = plot_df[plot_df['True_Label_Name'] == true_label_name]
        if len(subset_df) > 1:
            cov = np.cov(subset_df[['Dim1', 'Dim2']].values.T)
            if np.linalg.det(cov) == 0:
                print(f"Warning: Singular covariance matrix for {true_label_name}, cannot draw ellipse.")
                continue

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues) * 2  # For ~95% confidence

            color_idx = true_label_names.index(true_label_name) % len(sns.color_palette("tab10"))
            ellipse_color = sns.color_palette("tab10")[color_idx]

            ellipse = Ellipse(xy=(subset_df['Dim1'].mean(), subset_df['Dim2'].mean()),
                              width=width, height=height,
                              angle=angle, color=ellipse_color, alpha=0.2, fill=True, ec='black', lw=2)
            plt.gca().add_patch(ellipse)

    plt.title(
        f"Dual-Encoded Cluster Visualization for {'_'.join(combo_name)}\n(True Label Color, Predicted Cluster Style) using {dim_reduction_method}",
        fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(title="True Label / Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
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
    Marks distinction breakpoints (e.g., significant ARI improvement) on the trend plot.
    The breakpoint is marked based on the overall mean of the metric across all methods for each k.
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


# --- 6. Property Analysis Functions ---
def load_amino_acid_properties_for_analysis(filepath: str) -> pd.DataFrame:
    """Loads physicochemical properties of amino acids specifically for analysis (not PCA selection)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Amino acid properties file for analysis not found: {filepath}")
    df = pd.read_csv(filepath, index_col='Amino_Acid')
    return df


def get_amino_acid_property_diffs(aa1_name: str, aa2_name: str, properties_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates property differences between two amino acids for specified properties."""
    properties_to_check = ['Hydrophobicity', 'Molecular_Weight', 'pI']  # Define which properties to report diffs for

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
        # Assuming for 'confusion' analysis, the 'predicted' cluster label 'j' implies confusion with the amino acid
        # that *dominates* cluster 'j', or simply the amino acid at index 'j' if there's a 1-1 mapping.
        # For simplicity in this context, we take the amino acid at the `pred_idx` as the "confused with" amino acid.
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

    return top_confused_df


def analyze_feature_importance(X_scaled_df: pd.DataFrame, y_encoded: pd.Series, filename: str):
    """
    Analyzes feature importance using Mutual Information and plots top features.
    Args:
        X_scaled_df (pd.DataFrame): Scaled feature DataFrame.
        y_encoded (pd.Series): Encoded true labels.
        filename (str): File path to save the plot.
    """
    if X_scaled_df.empty or y_encoded.empty or len(y_encoded.unique()) < 2:
        print("Warning: Insufficient data for feature importance analysis.")
        return

    # Calculate Mutual Information for classification
    mi_scores = mutual_info_classif(X_scaled_df, y_encoded, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_scaled_df.columns).sort_values(ascending=False)

    plt.figure(figsize=(12, 7))
    mi_series.head(20).plot(kind='barh', color=sns.color_palette("viridis", len(mi_series.head(20))))
    plt.title("Top 20 Feature Importance (Mutual Information with Amino Acid Label)", fontsize=14)
    plt.xlabel("Mutual Information Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.gca().invert_yaxis()  # Highest score at the top
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.savefig(filename.replace(".png", ".svg"), format='svg')
    plt.close()
    print(f"Feature importance plot saved to: {filename}")

    return mi_series


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
        X_original_scaled, y_full_encoded, all_amino_acids_names = preprocess_data(df_original.copy(),
                                                                                   original_feature_columns,
                                                                                   label_column, missing_threshold=0.2)
    except ValueError as e:
        messagebox.showerror("Data Preprocessing Error", str(e))
        return

    print(f"\nData loaded and initial preprocessing complete. Found {len(all_amino_acids_names)} unique amino acids.")
    print(f"Amino Acid list: {', '.join(all_amino_acids_names)}")
    print(f"Original Feature dimensions: {X_original_scaled.shape[1]}")
    print(f"Total samples: {X_original_scaled.shape[0]}")

    # --- Feature Engineering ---
    # Apply feature engineering to the *original unscaled features* from df_original.
    # Then standardize the *combined* original and new features.
    print("\n--- Performing Feature Engineering ---")
    X_engineered_scaled = engineer_features(
        df_original[original_feature_columns].copy())  # Pass unscaled original features for engineering

    # After engineering, X_engineered_scaled contains all original and new features, already scaled.
    X_full_scaled = X_engineered_scaled  # Use this as the main feature set for clustering

    print(f"Final feature set (original + engineered) dimensions: {X_full_scaled.shape[1]}")

    # Get save directory
    save_base_dir = get_save_directory()
    if not save_base_dir:
        messagebox.showinfo("Info", "No save directory selected. Results will not be saved to files.")
        save_base_dir = None
    else:
        os.makedirs(save_base_dir, exist_ok=True)
        print(f"\nResults will be saved to: {save_base_dir}")
        # Save the preprocessed dataset as CSV
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
    aaindex_properties_filepath = os.path.join(data_dir,
                                               "AAindex_selected_properties.csv")  # User should provide this file
    processed_aaindex_df: Optional[pd.DataFrame] = None
    try:
        # User needs to prepare AAindex_selected_properties.csv with 'Amino_Acid' as index and numeric properties.
        # Example dummy if file not found, but real data is preferred.
        if not os.path.exists(aaindex_properties_filepath):
            print(f"\nWARNING: AAindex file '{aaindex_properties_filepath}' not found.")
            print(
                "Please create this file. It should be a CSV with 'Amino_Acid' as the first column and various physicochemical properties as subsequent columns.")
            print("Example row: Alanine, 0.62, 89.09, 6.00 ...")
            print("For now, generating a placeholder properties file (Molecular_Weight, Hydrophobicity, pI).")

            placeholder_props_data = {
                'Amino_Acid': all_amino_acids_names,
                'Hydrophobicity': [np.random.uniform(-2, 2) for _ in range(len(all_amino_acids_names))],
                'Molecular_Weight': [np.random.uniform(70, 250) for _ in range(len(all_amino_acids_names))],
                'pI': [np.random.uniform(3, 11) for _ in range(len(all_amino_acids_names))],
                # Add more dummy properties if needed to make PCA useful
                'Polarity': [np.random.uniform(0, 1) for _ in range(len(all_amino_acids_names))]
            }
            placeholder_props_df = pd.DataFrame(placeholder_props_data).set_index('Amino_Acid')
            placeholder_props_df.to_csv(aaindex_properties_filepath, index=True)
            print(f"Placeholder AAindex file generated at: {aaindex_properties_filepath}")

        processed_aaindex_df = load_aaindex_properties(aaindex_properties_filepath, AAINDEX_FEATURES_TO_SELECT_COUNT)
        if save_base_dir and processed_aaindex_df is not None:
            plot_property_pca(processed_aaindex_df, os.path.join(save_base_dir, "amino_acid_properties_pca.png"))
            # For property analysis, we will use the *original* properties, not PCA-reduced ones,
            # to show the direct differences in Hydrophobicity, MW, pI.
            # So, load original properties specifically for analysis.
            aaindex_original_for_analysis = load_amino_acid_properties_for_analysis(aaindex_properties_filepath)
        else:
            aaindex_original_for_analysis = None

    except Exception as e:
        print(
            f"\nError processing AAindex properties: {e}. Property-based combination search will use random sampling.")
        processed_aaindex_df = None
        aaindex_original_for_analysis = None

    # --- Combination Search & Evaluation Parameters ---
    # Max k to evaluate for trend plots (up to 10 as per request, or total AAs)
    max_k_value = min(len(all_amino_acids_names), MAX_K_VALUE_EXPLORE)
    if max_k_value < 2:
        messagebox.showerror("Error", "Minimum k_value is 2. Adjust max_k_value or number of amino acids.")
        return

    max_combinations_per_k = 50  # Max combinations to evaluate for each k

    all_results = []  # Stores all evaluation results from all repeats
    # This will store aggregated results (mean, std dev) for plotting trend lines
    aggregated_results_for_plots = []

    best_results_per_k = {}  # Stores the single best result (by mean ARI) for each k
    min_distinguishable_k_value = -1  # Initialized to -1, indicates not found

    # Define clustering methods to test
    clustering_methods_to_test = ['KMeans', 'KMedoids', 'LDA+KMeans', 'DBSCAN', 'OPTICS', 'AgglomerativeClustering']
    # Filter out methods if their libraries are not available
    clustering_methods_to_test = [m for m in clustering_methods_to_test if eval(m.split('+')[0]) is not None]
    if 'KMedoids' not in clustering_methods_to_test and KMedoids is None: pass
    if 'COP-KMeans' not in clustering_methods_to_test and COPKMeans is None: pass
    if 'OPTICS' not in clustering_methods_to_test and OPTICS is None: pass

    # --- Main Loop: Combination Traversal, Clustering & Evaluation (with Repeats) ---
    trial_counter = 0  # To provide a unique random_state for each clustering trial

    for combination, k_value, combo_type in combination_search_controller(
            all_amino_acids_names,
            max_k=max_k_value,
            max_combinations_per_k=max_combinations_per_k,
            properties_df=processed_aaindex_df  # Use PCA-reduced AAindex for combo prioritization
    ):
        print(f"\n--- Evaluating Combination k={k_value}, Combo: {combination} (Source: {combo_type}) ---")

        y_subset_remapped, original_subset_labels_names = remap_labels_for_subset(
            y_full_encoded, all_amino_acids_names, list(combination)
        )
        X_subset = X_full_scaled.loc[y_subset_remapped.index]

        n_clusters_expected = len(original_subset_labels_names)
        if n_clusters_expected < 2 or X_subset.shape[0] < n_clusters_expected:
            print(
                f"Combination {combination} has only {n_clusters_expected} amino acids or insufficient samples ({X_subset.shape[0]}), skipping clustering.")
            continue

        # Store raw repeat results for this specific combination-method pair
        results_for_this_combo_methods = []

        for method_name in clustering_methods_to_test:
            print(f"  Testing method: {method_name} ({N_REPEATS} repeats)")
            ari_scores_repeats = []
            silhouette_scores_repeats = []
            all_cms_repeats = []  # Store all confusion matrices for this method

            for repeat_idx in range(N_REPEATS):
                trial_counter += 1  # Global unique random_state for each run

                current_random_state = trial_counter  # Use global counter for random state

                cluster_labels, _ = cluster_with_method(
                    X_subset, y_subset_remapped, method_name, n_clusters_expected, random_state_val=current_random_state
                )

                if cluster_labels is None:
                    print(f"    Repeat {repeat_idx + 1}: {method_name} failed. Skipping this repeat.")
                    continue  # Skip this repeat, but continue with others

                eval_results_repeat = evaluate_clustering(
                    X_subset, y_subset_remapped, cluster_labels, combination, original_subset_labels_names
                )
                ari_scores_repeats.append(eval_results_repeat.get('ARI', 0.0))
                silhouette_scores_repeats.append(eval_results_repeat.get('Silhouette_Score', np.nan))
                all_cms_repeats.append(eval_results_repeat.get('Confusion_Matrix'))

            # Aggregate results for this method across repeats
            if ari_scores_repeats:
                mean_ari = np.mean(ari_scores_repeats)
                std_ari = np.std(ari_scores_repeats)
                mean_silhouette = np.nanmean(silhouette_scores_repeats) if not all(
                    np.isnan(silhouette_scores_repeats)) else np.nan
                std_silhouette = np.nanstd(silhouette_scores_repeats) if not all(
                    np.isnan(silhouette_scores_repeats)) else np.nan

                # Store aggregated results for plotting
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

                # Store full detailed results for JSON output (taking the first successful CM or an average one)
                # For simplicity, if multiple CMs are generated, we can store the one from the first repeat
                # or calculate a "median" CM if meaningful, but that's complex.
                # Here, we will just take the CM from the first successful run.
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
                    'Confusion_Matrix': first_successful_cm  # Store one CM for later plotting
                })

                print(
                    f"    Mean ARI: {mean_ari:.4f} (Std: {std_ari:.4f}), Mean Silhouette: {mean_silhouette:.4f} (Std: {std_silhouette:.4f})")
                if not np.isnan(mean_silhouette) and mean_silhouette < SILHOUETTE_LOWER_BOUND:
                    print(
                        f"    ⚠️ Warning: Mean Silhouette Score ({mean_silhouette:.4f}) is below threshold ({SILHOUETTE_LOWER_BOUND:.1f}).")

                # Update best result for current k for JSON (based on mean ARI)
                current_k_best_in_json = best_results_per_k.get(k_value, {'ARI': -1.0})
                if mean_ari > current_k_best_in_json['ARI']:
                    best_results_per_k[k_value] = {
                        'combination': combination,
                        'clustering_method': method_name,
                        'ARI': mean_ari,
                        'Silhouette_Score': mean_silhouette,
                        'Confusion_Matrix_JSON': first_successful_cm.tolist() if first_successful_cm is not None else None,
                        'original_subset_labels_names': original_subset_labels_names  # Needed for CM plotting from JSON
                    }
                    print(f"  Updated best result for k={k_value} (Mean ARI: {mean_ari:.4f}).")

        # --- Check Termination Condition (based on best ARI for the combination across methods) ---
        # Note: This checks the best ARI for the current *combination* and moves to next k if not met.
        # But if continue_after_threshold is True, it will *not* break the k_value loop early.
        if min_distinguishable_k_value == -1:  # Only set if not already found
            # Check if *any* method for this combination meets criteria
            for res_entry in results_for_this_combo_methods:  # Iterate over the temporary list for current combo's methods
                if res_entry['ARI'] >= ARI_STOP_THRESHOLD and not np.isnan(res_entry['Silhouette_Score']) and res_entry[
                    'Silhouette_Score'] >= SILHOUETTE_LOWER_BOUND:
                    min_distinguishable_k_value = k_value
                    print(f"\nFirst combination meeting thresholds found at k={k_value}!")
                    break  # Break from inner loop (methods for this combo)

    print("\n--- All Combinations Evaluated (or stopped as per `continue_after_threshold` setting) ---")

    # --- 6. Results Summary and Output ---
    if not all_results:
        print("\nNo clustering evaluation results were generated.")
        return

    # Create final DataFrame for export and plotting from raw results
    results_df = pd.DataFrame(all_results)
    # This dataframe already has ARI, Silhouette_Score, ARI_std, Silhouette_Score_std

    # Drop Confusion_Matrix column for CSV export as it's not scalar
    results_df_for_export = results_df.drop(columns=['Confusion_Matrix'])

    print("\n--- All Clustering Evaluation Results Summary (First 5 Rows) ---")
    print(results_df_for_export.head())

    # Output top 5 combinations by ARI (using mean ARI if available)
    top_n_overall = 5
    best_ari_overall_results = results_df_for_export.sort_values(by='ARI', ascending=False).head(top_n_overall)
    print(f"\n--- Overall Top {top_n_overall} Combinations (Sorted by Mean ARI) ---")
    print(best_ari_overall_results)

    # Save all results to CSV
    if save_base_dir:
        results_df_for_export.to_csv(os.path.join(save_base_dir, "all_clustering_results.csv"), index=False)
        print(f"\nAll results saved to：{os.path.join(save_base_dir, 'all_clustering_results.csv')}")
    else:
        print("\nNo save directory selected. All results CSV not saved.")

    # Save best results per k to JSON
    if save_base_dir:
        json_output_filepath = os.path.join(save_base_dir, "best_combination_per_k_results.json")
        save_json_results(best_results_per_k, "best_combination_per_k_results.json", save_base_dir)
    else:
        print("\nNo save directory selected. Best combination per k JSON not saved.")

    # --- 7. Final Visualizations ---
    if save_base_dir:
        print("\n--- Generating Final Visualizations ---")
        # Performance Trend Plot (ARI and Silhouette)
        # Use aggregated_results_for_plots which already has mean/std
        aggregated_results_df_for_plots = pd.DataFrame(aggregated_results_for_plots)
        plot_performance_trend(aggregated_results_df_for_plots,
                               os.path.join(save_base_dir, "performance_trend_plot.png"), ARI_STOP_THRESHOLD)
        print(f"Performance trend plot saved to：{os.path.join(save_base_dir, 'performance_trend_plot.png')}")

        # Distinction Breakpoint Plot
        mark_distinction_breakpoints(aggregated_results_df_for_plots, 'ARI_mean',
                                     os.path.join(save_base_dir, "ari_distinction_breakpoints.png"), ARI_STOP_THRESHOLD)
        print(f"Distinction breakpoint plot saved to：{os.path.join(save_base_dir, 'ari_distinction_breakpoints.png')}")

        # Plot for the "best overall" combination (highest ARI from best_ari_overall_results)
        if not best_ari_overall_results.empty:
            # Re-fetch the full row (including Confusion_Matrix) from the original all_results list
            best_overall_combo_tuple = best_ari_overall_results.iloc[0]['combination']
            best_overall_method_name = best_ari_overall_results.iloc[0]['clustering_method']

            # Find the original result dictionary that contains the Confusion_Matrix numpy array
            overall_best_full_result_dict = next(
                (res for res in all_results if res['combination'] == best_overall_combo_tuple and res[
                    'clustering_method'] == best_overall_method_name),
                None
            )

            if overall_best_full_result_dict:
                best_combo_names = overall_best_full_result_dict['combination']
                best_k = overall_best_full_result_dict['k_value']
                best_method = overall_best_full_result_dict['clustering_method']

                y_best_subset_remapped, original_best_subset_labels_names = remap_labels_for_subset(
                    y_full_encoded, all_amino_acids_names, list(best_combo_names)
                )
                X_best_subset = X_full_scaled.loc[y_best_subset_remapped.index]

                # Re-run clustering to get cluster_labels for plotting with fixed random_state=42 for consistency
                final_plot_cluster_labels, _ = cluster_with_method(
                    X_best_subset, y_best_subset_remapped, best_method, best_k, random_state_val=42
                )
                if final_plot_cluster_labels is not None:
                    # 2D Clustering Scatter Plot for the best overall combination
                    dual_encoded_filename = os.path.join(save_base_dir,
                                                         f"BestOverall_DualEncoded_{'_'.join(best_combo_names)}_{best_method}.png")
                    plot_dual_encoded_clusters(X_best_subset, y_best_subset_remapped, final_plot_cluster_labels,
                                               best_combo_names, dual_encoded_filename,
                                               original_best_subset_labels_names)
                    print(f"Best overall dual-encoded cluster plot saved to：{dual_encoded_filename}")

                    # Confusion Matrix for the best overall combination
                    cm_filename = os.path.join(save_base_dir,
                                               f"BestOverall_CM_{'_'.join(best_combo_names)}_{best_method}.png")
                    best_cm = overall_best_full_result_dict[
                        'Confusion_Matrix']  # Directly use the CM from the stored result

                    top_confused_df_best_combo = analyze_and_plot_confused_pairs_properties(
                        best_cm,
                        original_best_subset_labels_names,
                        aaindex_original_for_analysis,  # Use original AAindex properties for diffs
                        cm_filename,
                        num_top_confused=5
                    )
                    print(f"Best overall confusion matrix plot saved to：{cm_filename}")

        # --- Generate PDF Report (combining various analyses) ---
        if save_base_dir:
            pdf_report_path = os.path.join(save_base_dir, "amino_acid_discrimination_report.pdf")
            print(
                f"\nGenerating comprehensive PDF report to：{pdf_report_path} (Placeholder - actual PDF generation will need a library like FPDF/ReportLab)")

            # Prepare data for the report content string
            report_best_combo_names = "N/A"
            report_best_method = "N/A"
            report_best_ari = "N/A"
            report_best_silhouette = "N/A"
            report_top_confused_df_str = "No significant confused pairs or properties data for the best combination."

            if 'overall_best_full_result_dict' in locals() and overall_best_full_result_dict:
                report_best_combo_names = overall_best_full_result_dict['combination']
                report_best_method = overall_best_full_result_dict['clustering_method']
                report_best_ari = f"{overall_best_full_result_dict['ARI']:.4f}"
                report_best_silhouette = f"{overall_best_full_result_dict['Silhouette_Score']:.4f}"
                if 'top_confused_df_best_combo' in locals() and not top_confused_df_best_combo.empty:
                    report_top_confused_df_str = top_confused_df_best_combo.to_string(index=False)

            report_content = f"""
# Amino Acid Discrimination Analysis Report

## 1. Executive Summary
- **Target:** Identify minimum distinguishable amino acid combination using DNA-wrapped SWNT arrays.
- **Minimum Distinguishable k (based on ARI >= {ARI_STOP_THRESHOLD} & Silhouette >= {SILHOUETTE_LOWER_BOUND}):** {min_distinguishable_k_value if min_distinguishable_k_value != -1 else 'Not found within tested range'}

## 2. Data Overview and Preprocessing
- **Input Data:** {input_file_path}
- **Missing Value Handling:** Dropped samples with > {0.2 * 100}% missing features.
- **Normalization:** Z-score standardization applied.
- **Feature Engineering:** Intensity Ratios, Shift Differences, Intensity/Shift Ratios, Response Normalization applied.
- **Physicochemical Properties Added:** AAindex properties (PCA-reduced to {AAINDEX_FEATURES_TO_SELECT_COUNT} features for combination search).
- **Total Samples after Preprocessing:** {X_full_scaled.shape[0]}
- **Total Unique Amino Acids:** {len(all_amino_acids_names)}

## 3. Combination Search Strategy
- **k Range Explored:** 2 to {max_k_value}
- **Combinations per k:** Max {max_combinations_per_k}
- **Prioritization Method:** Property-based (Average Euclidean distance in AAindex PCA space) was used where applicable, otherwise random sampling.

## 4. Clustering Performance Overview
- **Clustering Methods Evaluated:** {', '.join(clustering_methods_to_test)} (Each repeated {N_REPEATS} times)
- **Termination Criteria:** ARI >= {ARI_STOP_THRESHOLD} AND Silhouette >= {SILHOUETTE_LOWER_BOUND}
- **Overall Top {top_n_overall} Combinations (by Mean ARI):**
{best_ari_overall_results.to_string(index=False)}

## 5. Performance Trend Analysis
- **ARI Trend:** See `performance_trend_plot.png` in results directory. Shows average ARI across methods as k increases.
- **Silhouette Trend:** See `performance_trend_plot.png` in results directory. Shows average Silhouette Score across methods as k increases.
- **Distinction Breakpoint:** See `ari_distinction_breakpoints.png`. Marks the first k where overall average ARI crossed {ARI_STOP_THRESHOLD}.

## 6. Detailed Analysis of Best Overall Combination
- **Combination:** {report_best_combo_names}
- **Clustering Method:** {report_best_method}
- **Mean ARI:** {report_best_ari}
- **Mean Silhouette Score:** {report_best_silhouette}

### 6.1 Clustering Visualization
- See `BestOverall_DualEncoded_{'_'.join(report_best_combo_names) if isinstance(report_best_combo_names, tuple) else 'N_A'}_{report_best_method}.png` for the scatter plot showing true labels (color) vs. predicted clusters (style), with confidence ellipses.

### 6.2 Confusion Analysis and Physicochemical Properties
- See `BestOverall_CM_{'_'.join(report_best_combo_names) if isinstance(report_best_combo_names, tuple) else 'N_A'}_{report_best_method}.png` for the confusion matrix.
- **Top {5} Confused Pairs & Property Differences:**
    {report_top_confused_df_str}

## 7. Physicochemical Property Analysis
- **Amino Acid Property PCA:** See `amino_acid_properties_pca.png`. Visualizes amino acids in their property space.
- **Feature Importance Analysis:** See `feature_importance.png`. Displays Mutual Information scores for features.
- **Confusion Frequency vs. Property Differences:**
    *This analysis would correlate the `Confusion_Count` from the confusion matrices with the property differences (e.g., Hydrophobicity_Diff, Molecular_Weight_Diff, pI_Diff) of the respective amino acid pairs across all evaluated combinations.*
    *This requires collecting all `top_confused_df` results and performing a meta-analysis.*

    *Placeholder for Pearson Correlation Result:*
    *(To implement this, one would collect `top_confused_df` for all evaluated combinations, then flatten and correlate the 'Confusion_Count' column with the 'Property_Diff' columns)*
    *Pearson R (Confusion Count vs Hydrophobicity Diff): [Value]*
    *Pearson R (Confusion Count vs MW Diff): [Value]*
    *Pearson R (Confusion Count vs pI Diff): [Value]*

## 8. Special Attention Amino Acids
- **Charged Amino Acids:** Asp, Glu, Lys, Arg
- **Aromatic Amino Acids:** Phe, Tyr, Trp
- *Observation on systemic confusion (e.g., if all charged AAs are clustered together) would be documented here based on manual inspection of confusion matrices or automated pattern detection.*
    *This section requires explicit logic to identify specific amino acid groups in confusion patterns.*

## 9. Conclusion and Next Steps
- Summary of findings.
- Recommendations for future work (e.g., exploring more advanced clustering, feature engineering, real-time prediction).

This report aims to consolidate all key findings and visualizations.
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
        print(f"Minimum stably distinguishable amino acid combination size (k)：{min_distinguishable_k_value}")
    else:
        print("No combination met the ARI and Silhouette thresholds within the tested range.")


if __name__ == '__main__':
    main(continue_after_threshold=True)  # Changed to call with the new parameter
