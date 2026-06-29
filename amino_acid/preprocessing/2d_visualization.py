import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
import json

# Try importing tkinter; if it fails, fall back to console input.
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    TKINTER_AVAILABLE = True
except ImportError:
    print("Tkinter not available. Falling back to console input for file paths.")
    TKINTER_AVAILABLE = False


    class MockTkinter:  # Simulate tkinter classes to prevent errors when tkinter is unavailable
        def Tk(self): return self

        def withdraw(self): pass

        def filedialog(self): return self

        def askopenfilename(self, *args, **kwargs): return input("Enter path to training data CSV file: ")

        def askdirectory(self, *args, **kwargs): return input("Enter directory to save results: ")

        def messagebox(self): return self

        def showinfo(self, *args, **kwargs): print(f"INFO: {args[1]}")

        def showerror(self, *args, **kwargs): print(f"ERROR: {args[1]}")


    tk = MockTkinter()
    filedialog = tk.filedialog()
    messagebox = tk.messagebox()

# Try importing UMAP; if it fails, exit.
try:
    import umap.umap_ as umap
except ImportError:
    print("UMAP is not installed. Please install it: pip install umap-learn")
    sys.exit(1)

# Set Matplotlib default font to support English, and ensure Arial and other English fonts are available
mpl.rcParams['font.family'] = ['Arial', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


def select_file():
    """Open file selection dialog to choose a training data CSV file, or prompt via console."""
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Training Data CSV File",
            filetypes=[("CSV files", "*.csv")]
        )
    else:
        file_path = input("Please enter the full path to your training data CSV file: ")
    return file_path


def get_save_directory():
    """Open folder selection dialog for the user to choose a directory to save files, or prompt via console."""
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        save_dir = filedialog.askdirectory(
            title="Select Directory to Save Results"
        )
    else:
        save_dir = input("Please enter the full path to the directory where results should be saved: ")
    return save_dir


def save_scatter_data(data, labels, filename, output_dir):
    """Save scatter plot data to a TXT file in the specified directory."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w') as f:
            f.write("Component 1\tComponent 2\tLabel\n")
            for i in range(data.shape[0]):
                f.write(f"{data[i, 0]:.6f}\t{data[i, 1]:.6f}\t{labels[i]}\n")
        print(f"Scatter plot data saved to: {file_path}")
    else:
        print("No output directory selected, scatter plot data not saved.")


def save_class_silhouette_scores(scores_df, filename, output_dir):
    """Save amino acid class silhouette scores to a CSV file."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        file_path = os.path.join(output_dir, filename)
        scores_df.to_csv(file_path, index=False)
        print(f"Class silhouette scores saved to: {file_path}")
    else:
        print("No output directory selected, class silhouette scores not saved.")


# --- Functions for calculating and filtering silhouette scores ---
def calculate_and_filter_silhouette(X, labels, amino_acid_names_map, n_top_amino_acids):
    """
    Calculates silhouette scores for each sample and filters based on the average silhouette score per class.

    Parameters:
        X (np.array): Reduced dimension data.
        labels (np.array): Original class labels (numeric encoding).
        amino_acid_names_map (dict): Mapping dictionary for amino acid names (numeric label -> amino acid name).
        n_top_amino_acids (int): Number of amino acid classes to keep.

    Returns:
        tuple: (selected_X, selected_labels_remapped, selected_amino_acid_names, selected_scores_df)
               selected_labels_remapped are re-encoded labels, starting from 0.
    """
    unique_labels_in_data = np.unique(labels)
    if len(unique_labels_in_data) < 2 or len(unique_labels_in_data) > len(X) - 1:
        print("Warning: Insufficient or too many categories to calculate Silhouette score. Skipping filtering.")
        return X, labels, [amino_acid_names_map[l] for l in unique_labels_in_data], pd.DataFrame()

    # Ensure at least 2 unique labels to calculate silhouette score
    if len(np.unique(labels)) < 2:
        print("Warning: Only one unique label found. Cannot calculate Silhouette score.")
        return X, labels, [amino_acid_names_map[l] for l in unique_labels_in_data], pd.DataFrame()

    sample_silhouette_values = silhouette_samples(X, labels)
    silhouette_df = pd.DataFrame({'label': labels, 'silhouette_score': sample_silhouette_values})

    # Calculate average silhouette score for each class
    cluster_silhouette_scores = silhouette_df.groupby('label')['silhouette_score'].mean().reset_index()
    cluster_silhouette_scores.columns = ['label_id', 'avg_silhouette_score']

    # Add amino acid names
    cluster_silhouette_scores['amino_acid_name'] = cluster_silhouette_scores['label_id'].map(amino_acid_names_map)

    # Sort by silhouette score from high to low, and select the top N
    selected_clusters_df = cluster_silhouette_scores.sort_values(
        by='avg_silhouette_score', ascending=False
    ).head(n_top_amino_acids)

    selected_amino_acid_ids = selected_clusters_df['label_id'].tolist()
    selected_amino_acid_names_list = selected_clusters_df['amino_acid_name'].tolist()

    # Filter original data points
    selected_indices = [i for i, label in enumerate(labels) if label in selected_amino_acid_ids]
    selected_X = X[selected_indices]
    selected_labels = labels[selected_indices]  # Original numerical labels

    # Re-map labels so colors are correct and continuous for visualization
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_amino_acid_ids)}
    selected_labels_remapped = np.array([label_mapping[label] for label in selected_labels])

    print(f"Based on Silhouette score, the top {len(selected_amino_acid_names_list)} amino acid classes have been filtered:")
    print(selected_clusters_df[['amino_acid_name', 'avg_silhouette_score']].to_string(index=False))

    return selected_X, selected_labels_remapped, selected_amino_acid_names_list, selected_clusters_df


# --- Improved visualization function ---
def plot_clusters(X, labels_remapped, amino_acid_names_list, title_prefix, filename_suffix, output_dir,
                  silhouette_scores_df=None, params_text="", explained_variance_ratio=None):
    """
    Plots the clustered data after dimensionality reduction.

    Parameters:
        X (np.array): Reduced dimension data.
        labels_remapped (np.array): Re-mapped class labels (continuous integers starting from 0).
        amino_acid_names_list (list): List of all amino acid names, ordered corresponding to original label encoding.
        title_prefix (str): Prefix for the chart title (e.g., "PCA", "t-SNE", "UMAP").
        filename_suffix (str): Suffix for the saved filename.
        output_dir (str): Directory to save the chart.
        silhouette_scores_df (pd.DataFrame, optional): DataFrame containing silhouette scores, for legend annotation.
        params_text (str, optional): Parameter text to add to the title.
        explained_variance_ratio (np.array, optional): Explained variance ratio from PCA, used for axis labels.
    """
    fig_height = 8
    fig_width = fig_height * 1.3

    fig = plt.figure(figsize=(fig_width, fig_height))
    # Use a single subplot, as the legend will be placed outside
    ax = fig.add_subplot(111)

    unique_labels_in_plot = np.unique(labels_remapped)

    # Improved color selection: if unique labels <= 20, use tab20, otherwise use husl for better distinction
    if len(unique_labels_in_plot) <= 20:
        palette = sns.color_palette("tab20", len(unique_labels_in_plot))
    else:
        palette = sns.color_palette("husl", len(unique_labels_in_plot))

    # Build a mapping from re-mapped label IDs to amino acid names
    label_id_to_name_map = {i: name for i, name in enumerate(amino_acid_names_list)}

    for i, label_id in enumerate(sorted(unique_labels_in_plot)):  # Ensure iteration in label ID order for consistent colors
        indices = labels_remapped == label_id
        ax.scatter(X[indices, 0], X[indices, 1],
                   color=palette[i],  # Use the i-th color from the palette
                   label=label_id_to_name_map[label_id],  # Use actual amino acid names as legend
                   alpha=0.7, s=50)

        # Add 95% confidence ellipse for each category
        if len(X[indices, :]) > 1:  # Covariance matrix requires at least 2 points
            # Calculate covariance matrix and mean
            cov = np.cov(X[indices, :], rowvar=False)
            mean = np.mean(X[indices, :], axis=0)

            # Eigenvalues and eigenvectors
            eig_vals, eig_vecs = np.linalg.eigh(cov)

            # Get the angle of the largest eigenvector
            angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

            # Width and height of the ellipse (assuming 95% confidence)
            # For 2D, sqrt(5.991) is approximately 2.4477, corresponding to 95% confidence interval of chi-squared distribution with 2 degrees of freedom
            width, height = 2 * np.sqrt(5.991 * eig_vals)

            ellipse = mpl.patches.Ellipse(xy=mean, width=width, height=height,
                                          angle=angle, color=palette[i], alpha=0.2, fill=True)
            ax.add_patch(ellipse)

    # Adjust plot limits
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    range_x = x_max - x_min
    range_y = y_max - y_min
    max_range = max(range_x, range_y)
    padding = max_range * 0.1
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    ax.set_xlim(center_x - (max_range / 2) - padding, center_x + (max_range / 2) + padding)
    ax.set_ylim(center_y - (max_range / 2) - padding, center_y + (max_range / 2) + padding)
    ax.set_aspect('equal', adjustable='box')

    # Apply PCA specific axis labels with explained variance ratio
    if title_prefix == 'PCA' and explained_variance_ratio is not None:
        ax.set_xlabel(f"PCA 1 ({explained_variance_ratio[0] * 100:.1f}%)", fontname='Arial', fontsize=18)
        ax.set_ylabel(f"PCA 2 ({explained_variance_ratio[1] * 100:.1f}%)", fontname='Arial', fontsize=18)
    else:
        ax.set_xlabel(f'{title_prefix} Component 1', fontname='Arial', fontsize=18)
        ax.set_ylabel(f'{title_prefix} Component 2', fontname='Arial', fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    full_title = f'{title_prefix} Dimensionality Reduction (All Categories)\n{params_text}' if params_text else f'{title_prefix} Dimensionality Reduction (All Categories)'
    plt.title(full_title, fontsize=20, fontweight='bold', fontname='Arial')

    # Legend - Integrate SS scores into labels and optimize placement
    legend_elements = []
    if silhouette_scores_df is not None and not silhouette_scores_df.empty:
        score_dict = silhouette_scores_df.set_index('amino_acid_name')['avg_silhouette_score'].to_dict()
    else:
        score_dict = {}

    for i, label_id in enumerate(sorted(unique_labels_in_plot)):
        name = label_id_to_name_map[label_id]
        score = score_dict.get(name, 'N/A')
        label_text = f"{name}"
        if score != 'N/A':
            label_text += f" (SS: {score:.3f})"

        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       label=label_text,
                       markerfacecolor=palette[i], markersize=10)
        )

    # Optimize legend placement - place outside the upper right of the plotting area
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=16,
        frameon=True,
        ncol=1,
        title="Amino Acids",
        title_fontsize=18
    )

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        plot_filepath_png = os.path.join(output_dir, f'{filename_suffix}_plot.png')
        plt.savefig(plot_filepath_png, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_filepath_png}")
    plt.close(fig)

    # Save scatter data, using original amino acid names, not re-encoded labels
    scatter_labels_named = np.array([label_id_to_name_map[l] for l in labels_remapped])
    save_scatter_data(X, scatter_labels_named, f'{filename_suffix}_data.txt', output_dir)


def analyze_clustering_effect(X_transformed, y_encoded, unique_labels, method_name, config_name, output_dir):
    """
    Analyzes clustering effect using silhouette scores and prints the best-performing classes.
    Returns a sorted list of (amino_acid_name, silhouette_score).
    """
    print(f"\n--- Clustering Effect Analysis for {method_name} ({config_name}) ---")

    # Check if there are enough unique classes and samples to calculate all metrics
    has_enough_clusters_for_metrics = len(np.unique(y_encoded)) >= 2 and X_transformed.shape[0] > len(
        np.unique(y_encoded))

    overall_silhouette = 'N/A'
    calinski_harabasz = 'N/A'
    davies_bouldin = 'N/A'
    sample_silhouette_values = np.array([])  # Initialize as empty array

    if has_enough_clusters_for_metrics:
        try:
            overall_silhouette = silhouette_score(X_transformed, y_encoded)
            print(f"Overall Silhouette Score: {overall_silhouette:.4f}")
            sample_silhouette_values = silhouette_samples(X_transformed, y_encoded)
        except ValueError as e:
            print(f"Could not calculate overall silhouette score: {e}")

        try:
            calinski_harabasz = calinski_harabasz_score(X_transformed, y_encoded)
            print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        except ValueError as e:
            print(f"Could not calculate Calinski-Harabasz score: {e}")

        try:
            davies_bouldin = davies_bouldin_score(X_transformed, y_encoded)
            print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        except ValueError as e:
            print(f"Could not calculate Davies-Bouldin score: {e}")
    else:
        print("Cannot calculate full clustering metrics: Not enough unique classes or samples.")
        return [], {}, 'N/A', 'N/A'  # If conditions not met, return N/A for all scores

    class_silhouette_scores = {}
    if len(sample_silhouette_values) > 0:  # Only proceed if silhouette scores were successfully calculated
        for i, label_idx in enumerate(y_encoded):
            label_name = unique_labels[label_idx]
            if label_name not in class_silhouette_scores:
                class_silhouette_scores[label_name] = []
            class_silhouette_scores[label_name].append(sample_silhouette_values[i])

    avg_class_silhouette = {
        aa: np.mean(scores) for aa, scores in class_silhouette_scores.items() if scores
    }

    sorted_classes = sorted(avg_class_silhouette.items(), key=lambda item: item[1], reverse=True)

    print("\nAverage Silhouette Score per Amino Acid (Higher is Better):")
    for aa, score in sorted_classes:
        print(f"  {aa}: {score:.4f} {'(Negative score: Poor clustering)' if score < 0 else ''}")

    print("\nTop 5 Amino Acids with Best Clustering Effect (based on this configuration):")
    for i, (aa, score) in enumerate(sorted_classes[:5]):
        print(f"{i + 1}. {aa} (Score: {score:.4f})")

    scores_df = pd.DataFrame(sorted_classes, columns=['Amino_Acid', 'Avg_Silhouette_Score'])
    save_class_silhouette_scores(scores_df,
                                 f'{method_name}_{config_name.replace(" ", "_")}_class_silhouette_scores.csv',
                                 output_dir)

    return sorted_classes, avg_class_silhouette, calinski_harabasz, davies_bouldin


def save_pca_components(pca_components, feature_names, explained_variance_ratio, filename, output_dir):
    """Save PCA components (loadings) and explained variance ratio to a CSV file."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        file_path = os.path.join(output_dir, filename)
        # Transpose pca_components so rows are features and columns are principal components
        # pca_components.shape is (n_components, n_features)
        # After transposing, shape is (n_features, n_components)
        pca_loadings_df = pd.DataFrame(pca_components.T,
                                       index=feature_names,
                                       columns=[f'PC{i + 1}' for i in range(
                                           pca_components.shape[0])])  # pca_components.shape[0] is n_components

        # Add explained variance ratio as a separate row
        explained_variance_row = pd.Series(explained_variance_ratio,
                                           index=[f'PC{i + 1}' for i in range(pca_components.shape[0])],
                                           name='Explained Variance Ratio')
        pca_loadings_df = pd.concat([pca_loadings_df, pd.DataFrame(explained_variance_row).T])

        pca_loadings_df.to_csv(file_path, index=True)  # Keep index as it contains feature names and explained variance row label
        print(f"PCA components saved to: {file_path}")
    else:
        print("No output directory selected, PCA components not saved.")


def plot_silhouette_ranking(avg_class_silhouettes_overall, output_dir, filename):
    """
    Generate a horizontal bar chart for amino acid clusterability ranking based on silhouette scores.

    Parameters:
        avg_class_silhouettes_overall (dict): Dictionary of amino acid names to their aggregated silhouette scores.
        output_dir (str): Directory to save the chart.
        filename (str): Filename to save (e.g., 'silhouette_ranking.png').
    """
    if not avg_class_silhouettes_overall:
        print("No silhouette scores available for ranking plot.")
        return

    # Convert to DataFrame and sort
    ranking_df = pd.DataFrame(avg_class_silhouettes_overall.items(), columns=['Amino_Acid', 'Silhouette_Score'])
    ranking_df = ranking_df.sort_values(by='Silhouette_Score', ascending=True)  # Ascending for barh so highest score is at the top

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='Silhouette_Score', y='Amino_Acid', data=ranking_df, palette='viridis')  # Use viridis palette
    plt.xlabel("Silhouette Score", fontsize=18)
    plt.ylabel("Amino Acid", fontsize=18)
    plt.title("Amino Acid Clusterability Ranking", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate values on the bars
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.005, p.get_y() + p.get_height() / 2,
                 f'{width:.3f}', ha='left', va='center', fontsize=12)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Amino Acid Clusterability Ranking plot saved to: {plot_path}")
    plt.close()


def save_feature_importance_plot(feature_importances, feature_names, output_dir, filename):
    """
    Generate and save a vertical bar chart of feature importance.

    Parameters:
        feature_importances (np.array): Array of feature importances.
        feature_names (list): List of feature names corresponding to importances.
        output_dir (str): Directory to save the chart.
        filename (str): Filename for the CSV (e.g., 'feature_importance.png').
    """
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 7))
    # For vertical bar chart: x-axis is Feature, y-axis is Importance
    sns.barplot(x='Feature', y='Importance', data=importance_df, palette='magma')
    plt.xlabel("Feature", fontsize=18)
    plt.ylabel("Importance", fontsize=18)
    plt.title("Feature Importance Ranking (Random Forest Classifier)", fontsize=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=16)  # Rotate x-axis labels for readability
    plt.yticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines on y-axis
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Feature Importance plot saved to: {plot_path}")
    plt.close()


def save_feature_importance_data(feature_importances, feature_names, output_dir, filename):
    """
    Save feature importance data to a CSV file.

    Parameters:
        feature_importances (np.array): Array of feature importances.
        feature_names (list): List of feature names corresponding to importances.
        output_dir (str): Directory to save the CSV.
        filename (str): CSV filename (e.g., 'feature_importance.csv').
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        file_path = os.path.join(output_dir, filename)
        importance_df.to_csv(file_path, index=False)
        print(f"Feature Importance data saved to: {file_path}")
    else:
        print("No output directory selected, feature importance data not saved.")


def progressive_class_curve(X_embedded, y_encoded, label_to_name_map, max_classes, method_name, output_dir,
                            silhouette_threshold=0.5):
    """
    Progressively increases top-N classes, evaluates silhouette scores, and plots the curve showing
    distinguishable classes as N increases. Also identifies the best N based on a silhouette threshold and plots.

    Parameters:
        X_embedded (np.array): Embedded data after dimensionality reduction (PCA/t-SNE/UMAP)
        y_encoded (np.array): Original encoded labels
        label_to_name_map (dict): Numeric label -> amino acid name mapping
        max_classes (int): Maximum number of classes (typically 18)
        method_name (str): Method name, used for saving filenames and chart titles
        output_dir (str): Path to save images
        silhouette_threshold (float): Silhouette score threshold for determining "best N".
    """
    n_list, score_list = [], []
    best_n = 0
    best_score = -1.0  # Initialize with an impossibly low score

    # Variables to store data for the best N plot
    best_n_X_sel = None
    best_n_y_sel = None
    best_n_aa_names = None
    best_n_df_scores = None

    print(f"\n--- Generating Progressive Class Separability Curve for {method_name} ---")
    for n in range(2, max_classes + 1):
        try:
            # Recompute silhouette filtering for top N amino acids each time
            # Note: this means recalculating average silhouette scores for all classes each time
            # X_embedded is the full reduced dataset, we filter from it based on the top N classes
            X_sel, y_sel, aa_names, df_scores = calculate_and_filter_silhouette(
                X_embedded, y_encoded, label_to_name_map, n_top_amino_acids=n
            )
            # Ensure enough data points and clusters to calculate silhouette score
            if len(np.unique(y_sel)) < 2 or len(X_sel) <= len(np.unique(y_sel)):
                print(f"  Skipping N={n}: Not enough unique classes or samples after filtering for SS calculation.")
                continue

            score = silhouette_score(X_sel, y_sel)
            n_list.append(n)
            score_list.append(score)
            print(f"  N = {n}, Silhouette Score = {score:.4f}")

            # Instruction 5: Check for best_n
            # Find the largest N for which the Silhouette score is >= threshold
            if score >= silhouette_threshold and n > best_n:
                best_n = n
                best_score = score
                best_n_X_sel = X_sel
                best_n_y_sel = y_sel
                best_n_aa_names = aa_names
                best_n_df_scores = df_scores

        except Exception as e:
            print(f"Error at N={n} for {method_name}: {e}")
            continue

    # Plot progressive curve
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, score_list, marker='o', linestyle='-', color='royalblue')
    plt.xlabel("Number of Amino Acid Classes", fontsize=16)
    plt.ylabel("Average Silhouette Score", fontsize=16)
    plt.title(f"Progressive Class Separability ({method_name})", fontsize=20, fontweight='bold')
    plt.xticks(n_list, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        plot_path = os.path.join(output_dir, f"{method_name.lower()}_progressive_curve.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Progressive curve saved to: {plot_path}")
    plt.close()

    # Instruction 5: If best_n found, plot the chart
    if best_n > 0 and best_n_X_sel is not None:
        print(f"\n--- Best N found for {method_name}: {best_n} classes with Silhouette Score {best_score:.4f} ---")
        if output_dir:
            plot_clusters(best_n_X_sel, best_n_y_sel, best_n_aa_names,
                          f'{method_name} (Best {best_n} Classes)',
                          f'{method_name.lower()}_best{best_n}_categories',
                          output_dir, best_n_df_scores, f"SS: {best_score:.3f}")


def main():
    """Main function, controls the script execution flow."""

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

        # --- Data preparation for dimensionality reduction ---
        # Goal: Each row is a unique sample (amino acid + concentration + replicate)
        # Features: 8 dimensions (intensity and shift for each of 4 chiralities)

        # Define chiralities and their corresponding intensity/shift columns
        chiralities = ['(6,5)', '(7,5)',
                       '(8,3)', 'S7-(6,5)']
        feature_columns_flat = []
        for chirality in chiralities:
            feature_columns_flat.append(f'{chirality}_intensity')
            feature_columns_flat.append(f'{chirality}_shift')

        # Select features and target labels
        # Ensure all feature columns exist
        missing_cols = [col for col in feature_columns_flat if col not in df_original.columns]
        if missing_cols:
            messagebox.showerror("Error", f"Missing expected feature columns in the CSV: {', '.join(missing_cols)}")
            return

        # X will contain 8-dimensional feature vectors directly from df_original
        X = df_original[feature_columns_flat].copy()
        y_labels_original = df_original['AA'].copy()

        print("\nPrepared Data for Dimensionality Reduction Info:")
        print(X.info())
        print("\nFirst 5 Rows of Prepared Data:")
        print(X.head())
        print(f"\nTotal samples for dimensionality reduction: {len(X)}")

        # Verify number of points per amino acid
        # Assumes df_original implicitly has 3 replicates per concentration.
        # This check confirms each AA has 18 points (6 concentrations * 3 replicates), as expected.
        print("\nNumber of samples per amino acid:")
        sample_counts_per_aa = df_original.groupby('AA').size()
        print(sample_counts_per_aa)
        if not (sample_counts_per_aa == 18).all():
            print("\nWARNING: Not all amino acids have exactly 18 points. Please check your input data structure.")
            print("Expected: 18 points (6 concentrations * 3 replicates) per amino acid.")

        # Handle potential NaNs in feature data before scaling
        X.fillna(0, inplace=True)  # Fill NaNs with 0, or consider other imputation strategies

        # Map AAs to numerical labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_labels_original)
        unique_labels = label_encoder.classes_
        label_to_name_map = {i: name for i, name in enumerate(unique_labels)}

        # Create remapped_y_encoded for plotting to ensure labels are continuous from 0
        # This is useful for consistent color mapping in seaborn/matplotlib palettes
        labels_for_plotting_remapped = np.array(
            [label_encoder.transform([label_name])[0] for label_name in y_labels_original])

        save_base_dir = get_save_directory()
        if not save_base_dir:
            messagebox.showinfo("Info", "No save directory selected, results will not be saved to files.")
            save_base_dir = None
        else:
            # Create a general plots directory for plots not specific to scaler/method
            general_plots_dir = os.path.join(save_base_dir, "general_plots")
            os.makedirs(general_plots_dir, exist_ok=True)

        print("\n===== Performing Dimensionality Reduction and Visualization =====")

        # Dictionary to store all results for summary file
        all_results_summary = {}

        # --- Set N values (can be adjusted as needed) ---
        N_TOP_AMINO_ACIDS_SUMMARY = 5  # Only for reporting best-performing classes in text summary
        N_TOP_AMINO_ACIDS_FOR_PLOTTING = 5  # For Top-N plotting

        # --- Instruction 3: Feature Importance Ranking Plot ---
        print("\n--- Generating Feature Importance Ranking ---")
        try:
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X, y_encoded)  # Use original X features
            feature_importances = rf_classifier.feature_importances_

            if save_base_dir:
                save_feature_importance_plot(feature_importances, feature_columns_flat, general_plots_dir,
                                             'feature_importance.png')
                save_feature_importance_data(feature_importances, feature_columns_flat, general_plots_dir,
                                             'feature_importance.csv')
        except Exception as e:
            print(f"Error generating Feature Importance: {e}")

        # Define scalers to use
        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler()
        }

        # Initialize dictionary to aggregate silhouette scores for overall ranking plot (Instruction 1)
        all_individual_class_silhouettes_for_ranking_plot = {}

        for scaler_name, scaler_obj in scalers.items():
            # Create specific directory for each scaler and method combination
            current_scaler_base_dir = os.path.join(save_base_dir, f"{scaler_name}")
            os.makedirs(current_scaler_base_dir, exist_ok=True)

            print(f"\n--- Running analysis with {scaler_name} ---")
            X_scaled = scaler_obj.fit_transform(X)

            # --- PCA ---
            method_key = f'PCA_{scaler_name}'
            all_results_summary[method_key] = {}
            pca_output_dir = os.path.join(current_scaler_base_dir, "PCA")
            os.makedirs(pca_output_dir, exist_ok=True)

            print(f"\n--- PCA Analysis ({scaler_name}) ---")
            n_components_pca = 2
            pca = PCA(n_components=n_components_pca, random_state=42)
            X_pca = pca.fit_transform(X_scaled)

            pca_params_text = f"Explained Variance: {pca.explained_variance_ratio_.sum():.4f}"

            # Calculate and analyze silhouette scores for the full dataset (for reporting)
            sorted_classes_pca, avg_class_silhouette_pca, pca_ch_score, pca_db_score = analyze_clustering_effect(X_pca,
                                                                                                                 y_encoded,
                                                                                                                 unique_labels,
                                                                                                                 'PCA',
                                                                                                                 'Full_Dataset',
                                                                                                                 pca_output_dir)
            overall_pca_silhouette = silhouette_score(X_pca, y_encoded) if len(np.unique(y_encoded)) > 1 and \
                                                                           X_pca.shape[
                                                                               0] > len(np.unique(y_encoded)) else 'N/A'

            all_results_summary[method_key]['Full_Dataset'] = {
                'params': {'Explained Variance': pca.explained_variance_ratio_.sum()},
                'overall_silhouette': overall_pca_silhouette,
                'calinski_harabasz_score': pca_ch_score,
                'davies_bouldin_score': pca_db_score,
                'class_silhouettes': sorted_classes_pca,
                'top_N_classes': sorted_classes_pca[:N_TOP_AMINO_ACIDS_SUMMARY]
            }

            # Plot PCA for all amino acids
            all_pca_silhouette_df = pd.DataFrame(sorted_classes_pca,
                                                 columns=['amino_acid_name', 'avg_silhouette_score'])
            if save_base_dir:
                plot_clusters(X_pca, labels_for_plotting_remapped, unique_labels,
                              'PCA', f'pca_{scaler_name.lower()}_all_categories', pca_output_dir, all_pca_silhouette_df,
                              pca_params_text, explained_variance_ratio=pca.explained_variance_ratio_)

            # Update overall class silhouette scores for ranking plot (Instruction 1)
            for aa, score in avg_class_silhouette_pca.items():
                all_individual_class_silhouettes_for_ranking_plot[aa] = max(
                    all_individual_class_silhouettes_for_ranking_plot.get(aa, -float('inf')), score)

            # Save PCA components
            save_pca_components(pca.components_, feature_columns_flat, pca.explained_variance_ratio_,
                                f'pca_components_{scaler_name.lower()}.csv', pca_output_dir)

            # --- Instruction 2: Top-N PCA Plot ---
            print(f"\n--- Generating Top-{N_TOP_AMINO_ACIDS_FOR_PLOTTING} PCA Plot ({scaler_name}) ---")
            selected_X_pca, selected_labels_remapped_pca, selected_amino_acid_names_pca, selected_clusters_df_pca = \
                calculate_and_filter_silhouette(X_pca, y_encoded, label_to_name_map, N_TOP_AMINO_ACIDS_FOR_PLOTTING)

            # When plotting, we need the scores of the selected amino acids
            top_n_pca_silhouette_df = pd.DataFrame(selected_clusters_df_pca)  # This already contains scores for selected AAs
            if save_base_dir:
                plot_clusters(selected_X_pca, selected_labels_remapped_pca, selected_amino_acid_names_pca,
                              f'PCA (Top {N_TOP_AMINO_ACIDS_FOR_PLOTTING} Categories)',
                              f'pca_{scaler_name.lower()}_top{N_TOP_AMINO_ACIDS_FOR_PLOTTING}_categories',
                              pca_output_dir, top_n_pca_silhouette_df, pca_params_text, explained_variance_ratio=pca.explained_variance_ratio_)

            # --- Instruction 4 and 5: Progressive Class Separability Curve for PCA ---
            if save_base_dir:
                progressive_class_curve(X_pca, y_encoded, label_to_name_map,
                                        max_classes=len(unique_labels), method_name=f'PCA_{scaler_name}',
                                        output_dir=pca_output_dir)

            # --- t-SNE ---
            method_key = f't-SNE_{scaler_name}'
            all_results_summary[method_key] = {}
            tsne_output_dir = os.path.join(current_scaler_base_dir, "t-SNE")
            os.makedirs(tsne_output_dir, exist_ok=True)

            print(f"\n--- t-SNE Analysis ({scaler_name}) ---")
            tsne_param_combinations = [
                {'perplexity': 30, 'max_iter': 1000, 'learning_rate': 'auto'},
                {'perplexity': 50, 'max_iter': 2000, 'learning_rate': 'auto'},
                {"perplexity": 20, "max_iter": 3000, "learning_rate": 100},
                {"perplexity": 40, "max_iter": 3000, "learning_rate": 200},
                {"perplexity": 60, "max_iter": 3000, "learning_rate": 300},
            ]

            for i, params in enumerate(tsne_param_combinations):
                config_name = f"Config_{i + 1}_p{params['perplexity']}_i{params['max_iter']}_lr{params['learning_rate']}"

                tsne_plot_params = {k: v for k, v in params.items() if k != 'learning_rate'}
                tsne_plot_params_text = ', '.join([f"{k}: {v}" for k, v in tsne_plot_params.items()])
                if params['learning_rate'] != 'auto':
                    tsne_plot_params_text += f", learning_rate: {params['learning_rate']}"

                print(f"\nEvaluating t-SNE with parameters: {params} ({scaler_name})")
                tsne = TSNE(n_components=2, random_state=42, **params)
                X_tsne = tsne.fit_transform(X_scaled)

                # Calculate and analyze silhouette scores for the full dataset (for reporting)
                sorted_classes_tsne, avg_class_silhouette_tsne, tsne_ch_score, tsne_db_score = analyze_clustering_effect(
                    X_tsne, y_encoded, unique_labels,
                    't-SNE', config_name,
                    tsne_output_dir)
                overall_tsne_silhouette = silhouette_score(X_tsne, y_encoded) if len(np.unique(y_encoded)) > 1 and \
                                                                                 X_tsne.shape[0] > len(
                    np.unique(y_encoded)) else 'N/A'

                all_results_summary[method_key][config_name] = {
                    'params': params,
                    'overall_silhouette': overall_tsne_silhouette,
                    'calinski_harabasz_score': tsne_ch_score,
                    'davies_bouldin_score': tsne_db_score,
                    'class_silhouettes': sorted_classes_tsne,
                    'top_N_classes': sorted_classes_tsne[:N_TOP_AMINO_ACIDS_SUMMARY]
                }

                # Plot t-SNE for all amino acids
                all_tsne_silhouette_df = pd.DataFrame(sorted_classes_tsne,
                                                      columns=['amino_acid_name', 'avg_silhouette_score'])
                if save_base_dir:
                    plot_clusters(X_tsne, labels_for_plotting_remapped, unique_labels,
                                  f't-SNE (Config {i + 1})', f'tsne_{scaler_name.lower()}_{config_name}_all_categories',
                                  tsne_output_dir,
                                  all_tsne_silhouette_df, tsne_plot_params_text)

                # Update overall class silhouette scores for ranking plot (Instruction 1)
                for aa, score in avg_class_silhouette_tsne.items():
                    all_individual_class_silhouettes_for_ranking_plot[aa] = max(
                        all_individual_class_silhouettes_for_ranking_plot.get(aa, -float('inf')), score)

                # --- Instruction 2: Top-N t-SNE Plot ---
                print(
                    f"\n--- Generating Top-{N_TOP_AMINO_ACIDS_FOR_PLOTTING} t-SNE Plot (Config {i + 1}, {scaler_name}) ---")
                selected_X_tsne, selected_labels_remapped_tsne, selected_amino_acid_names_tsne, selected_clusters_df_tsne = \
                    calculate_and_filter_silhouette(X_tsne, y_encoded, label_to_name_map,
                                                    N_TOP_AMINO_ACIDS_FOR_PLOTTING)

                top_n_tsne_silhouette_df = pd.DataFrame(selected_clusters_df_tsne)
                if save_base_dir:
                    plot_clusters(selected_X_tsne, selected_labels_remapped_tsne, selected_amino_acid_names_tsne,
                                  f't-SNE (Top {N_TOP_AMINO_ACIDS_FOR_PLOTTING} Categories - Config {i + 1})',
                                  f'tsne_{scaler_name.lower()}_{config_name}_top{N_TOP_AMINO_ACIDS_FOR_PLOTTING}_categories',
                                  tsne_output_dir, top_n_tsne_silhouette_df, tsne_plot_params_text)

                # --- Instruction 4 and 5: Progressive Class Separability Curve for t-SNE (for each configuration) ---
                if save_base_dir:
                    progressive_class_curve(X_tsne, y_encoded, label_to_name_map,
                                            max_classes=len(unique_labels),
                                            method_name=f't-SNE_{scaler_name}_{config_name}',
                                            output_dir=tsne_output_dir)

            # --- UMAP ---
            method_key = f'UMAP_{scaler_name}'
            all_results_summary[method_key] = {}
            umap_output_dir = os.path.join(current_scaler_base_dir, "UMAP")
            os.makedirs(umap_output_dir, exist_ok=True)

            print(f"\n--- UMAP Analysis ({scaler_name}) ---")
            umap_param_combinations = [
                {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'},
                {'n_neighbors': 30, 'min_dist': 0.3, 'metric': 'cosine'},
                {"n_neighbors": 5, "min_dist": 0.01, "metric": "cosine"},
                {"n_neighbors": 10, "min_dist": 0.01, "metric": "correlation"},
                {"n_neighbors": 15, "min_dist": 0.05, "metric": "euclidean"},
            ]

            for i, params in enumerate(umap_param_combinations):
                config_name = f"Config_{i + 1}_nn{params['n_neighbors']}_md{params['min_dist']}_m{params['metric']}"
                all_results_summary[method_key][config_name] = {}

                print(f"\nEvaluating UMAP with parameters: {params} ({scaler_name})")
                reducer = umap.UMAP(n_components=2, random_state=42, **params)
                X_umap = reducer.fit_transform(X_scaled)

                params_text = f"N_Neighbors: {params['n_neighbors']}, Min_Dist: {params['min_dist']}, Metric: {params['metric']}"

                # Calculate and analyze silhouette scores for the full dataset (for reporting)
                sorted_classes_umap, avg_class_silhouette_umap, umap_ch_score, umap_db_score = analyze_clustering_effect(
                    X_umap, y_encoded, unique_labels,
                    'UMAP', config_name,
                    umap_output_dir)
                overall_umap_silhouette = silhouette_score(X_umap, y_encoded) if len(np.unique(y_encoded)) > 1 and \
                                                                                 X_umap.shape[0] > len(
                    np.unique(y_encoded)) else 'N/A'

                all_results_summary[method_key][config_name] = {
                    'params': params,
                    'overall_silhouette': overall_umap_silhouette,
                    'calinski_harabasz_score': umap_ch_score,
                    'davies_bouldin_score': umap_db_score,
                    'class_silhouettes': sorted_classes_umap,
                    'top_N_classes': sorted_classes_umap[:N_TOP_AMINO_ACIDS_SUMMARY]
                }

                # Plot UMAP for all amino acids
                all_umap_silhouette_df = pd.DataFrame(sorted_classes_umap,
                                                      columns=['amino_acid_name', 'avg_silhouette_score'])
                if save_base_dir:
                    plot_clusters(X_umap, labels_for_plotting_remapped, unique_labels,
                                  f'UMAP (Config {i + 1})', f'umap_{scaler_name.lower()}_{config_name}_all_categories',
                                  umap_output_dir,
                                  all_umap_silhouette_df, params_text)

                # Update overall class silhouette scores for ranking plot (Instruction 1)
                for aa, score in avg_class_silhouette_umap.items():
                    all_individual_class_silhouettes_for_ranking_plot[aa] = max(
                        all_individual_class_silhouettes_for_ranking_plot.get(aa, -float('inf')), score)

                # --- Instruction 4 and 5: Progressive Class Separability Curve for UMAP ---
                if save_base_dir:
                    progressive_class_curve(X_umap, y_encoded, label_to_name_map,
                                            max_classes=len(unique_labels),
                                            method_name=f'UMAP_{scaler_name}_{config_name}', output_dir=umap_output_dir)

        # --- Instruction 1: Generate final amino acid clusterability ranking plot ---
        # This plot uses aggregated scores across all methods and scalers
        if save_base_dir and all_individual_class_silhouettes_for_ranking_plot:
            plot_silhouette_ranking(all_individual_class_silhouettes_for_ranking_plot, general_plots_dir,
                                    'silhouette_ranking.png')
        elif save_base_dir:
            print("\nNo aggregated silhouette scores to generate overall ranking plot.")

        # --- Generate final summary text file ---
        if save_base_dir:
            summary_file_path = os.path.join(save_base_dir, "dimensionality_reduction_summary.txt")
            with open(summary_file_path, 'w') as f:
                f.write("Dimensionality Reduction and Clustering Analysis Summary\n\n")
                f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for method_key, method_configs in all_results_summary.items():
                    f.write(f"--- {method_key} Analysis ---\n")
                    for config_name, results in method_configs.items():
                        f.write(f"\n  Configuration: {config_name}\n")
                        f.write(f"    Parameters: {results['params']}\n")
                        f.write(f"    Overall Silhouette Score: {results['overall_silhouette']:.4f}\n" if isinstance(
                            results['overall_silhouette'],
                            float) else f"    Overall Silhouette Score: {results['overall_silhouette']}\n")
                        f.write(
                            f"    Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}\n" if isinstance(
                                results['calinski_harabasz_score'],
                                float) else f"    Calinski-Harabasz Score: {results['calinski_harabasz_score']}\n")
                        f.write(f"    Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}\n" if isinstance(
                            results['davies_bouldin_score'],
                            float) else f"    Davies-Bouldin Score: {results['davies_bouldin_score']}\n")

                        f.write("    Average Silhouette Score per Amino Acid (Higher is Better):\n")
                        if results['class_silhouettes']:
                            for aa, score in results['class_silhouettes']:
                                f.write(
                                    f"      {aa}: {score:.4f} {'(Negative score: Poor clustering)' if score < 0 else ''}\n")
                        else:
                            f.write("      No class silhouette scores available.\n")

                        f.write(
                            f"    Top {N_TOP_AMINO_ACIDS_SUMMARY} Amino Acids with Best Clustering Effect (Full Dataset Analysis):\n")
                        if results['top_N_classes']:
                            for j, (aa, score) in enumerate(results['top_N_classes']):
                                f.write(f"      {j + 1}. {aa} (Score: {score:.4f})\n")
                        else:
                            f.write(f"      No top {N_TOP_AMINO_ACIDS_SUMMARY} available.\n")
                    f.write("\n")  # Add empty line between methods for readability

                # Aggregate all individual class silhouette scores to find overall best (this duplicates all_individual_class_silhouettes_for_ranking_plot but kept for summary text)
                all_individual_class_silhouettes_summary = {}
                for method_key, method_configs in all_results_summary.items():
                    for config_name, results in method_configs.items():
                        for aa, score in results.get('class_silhouettes', []):
                            # Take the highest score if amino acid appears in multiple methods/configs
                            all_individual_class_silhouettes_summary[aa] = max(
                                all_individual_class_silhouettes_summary.get(aa, -float('inf')),
                                score)

                if all_individual_class_silhouettes_summary:
                    sorted_aggregated_classes = sorted(all_individual_class_silhouettes_summary.items(),
                                                       key=lambda item: item[1],
                                                       reverse=True)
                    num_best_classes_overall = min(N_TOP_AMINO_ACIDS_SUMMARY, len(sorted_aggregated_classes))

                    f.write(
                        f"\n--- Overall Top {num_best_classes_overall} Best-Clustering Amino Acids (based on highest individual silhouette score across all methods/configs) ---\n")
                    for i, (aa, score) in enumerate(sorted_aggregated_classes[:num_best_classes_overall]):
                        f.write(f"{i + 1}. {aa} (Score: {score:.4f})\n")
                else:
                    f.write("\n--- Overall Best-Clustering Amino Acids ---\n")
                    f.write("No overall best-clustering amino acids found.\n")

                f.write("\n===== Dimensionality Reduction and Visualization Complete =====")
            print(f"\nSummary text file saved to: {summary_file_path}")
        else:
            print("\nNo save directory selected, summary text file not saved.")

        print("\n===== Dimensionality Reduction and Visualization Complete =====")

    else:
        messagebox.showinfo("Info", "No training data file selected, program exited.")


if __name__ == "__main__":
    main()
