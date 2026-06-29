import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
import itertools
import random
import threading
import time

# 设置Matplotlib字体为英文常用字体，确保不出现方框
mpl.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is displayed correctly

# Global variables for data and results
df_original = None
all_amino_acids = []
current_analysis_results = []  # To store all final optimized results
best_results_for_display = []  # Top 5 results for final display


# --- Helper Functions (from previous versions, adapted) ---
def select_file():
    """Opens a file dialog to select the raw data CSV file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Raw Data CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    return file_path


def get_save_directory():
    """Opens a directory dialog for the user to choose where to save results."""
    root = tk.Tk()
    root.withdraw()
    save_dir = filedialog.askdirectory(
        title="Select Directory to Save Results"
    )
    root.destroy()
    return save_dir


def calculate_metrics(X_for_clustering, true_labels, predicted_labels):
    """
    Calculates clustering evaluation metrics including ARI and NMI.
    X_for_clustering: The data actually used for clustering (e.g., X_scaled).
    Returns NaN if not computable.
    """
    silhouette, calinski_harabasz, davies_bouldin = np.nan, np.nan, np.nan
    ari, nmi = np.nan, np.nan

    n_clusters_pred = len(set(predicted_labels))
    n_samples = len(X_for_clustering)

    # Internal metrics (Silhouette, CH, DB)
    if n_clusters_pred > 1 and n_clusters_pred < n_samples:
        try:
            silhouette = silhouette_score(X_for_clustering, predicted_labels)
        except Exception:
            pass
        try:
            calinski_harabasz = calinski_harabasz_score(X_for_clustering, predicted_labels)
        except Exception:
            pass
        if len(set(predicted_labels)) > 1:  # Davies-Bouldin also requires > 1 clusters
            try:
                davies_bouldin = davies_bouldin_score(X_for_clustering, predicted_labels)
            except Exception:
                pass

    # External metrics (ARI, NMI) - compare with true_labels
    if len(set(true_labels)) > 1 and len(set(predicted_labels)) > 1:
        try:
            ari = adjusted_rand_score(true_labels, predicted_labels)
        except Exception:
            pass
        try:
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        except Exception:
            pass

    return silhouette, calinski_harabasz, davies_bouldin, ari, nmi


# --- Core Analysis Logic ---
def run_tsne_kmeans_analysis(amino_acids_subset, perplexity, learning_rate, max_iter):
    """
    Runs KMeans (on original scaled data first), then t-SNE (for visualization) for a given subset of amino acids.
    Returns calculated metrics (including ARI, NMI) and t-SNE coordinates.
    """
    global df_original

    df_filtered = df_original[df_original['AA'].isin(amino_acids_subset)].copy()

    actual_unique_aas_in_subset = df_filtered['AA'].unique()

    k_clusters = len(actual_unique_aas_in_subset)

    if k_clusters < 2 or len(df_filtered) < k_clusters:
        return {
            'amino_acids': amino_acids_subset,
            'tsne_params': {'perplexity': perplexity, 'learning_rate': learning_rate, 'max_iter': max_iter},
            'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan,
            'ari': np.nan, 'nmi': np.nan,
            'tsne_coords': np.array([]),
            'kmeans_labels': np.array([]),
            'true_labels': np.array([]),
            'filtered_df_empty': True  # Indicate this result is invalid due to empty/insufficient data
        }

    chiralities = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']
    intensity_cols = [f'{c}_intensity' for c in chiralities]
    shift_cols = [f'{c}_shift' for c in chiralities]
    feature_cols = intensity_cols + shift_cols

    X_features = df_filtered[feature_cols].values
    y_true_labels = df_filtered['AA'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)  # Standardized original features

    # 1. KMeans Clustering on X_scaled (original standardized features)
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X_scaled)  # KMeans is done first

    # 2. t-SNE Dimensionality Reduction on X_scaled (for visualization)
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=perplexity,
                max_iter=max_iter,
                learning_rate=learning_rate,
                init='random')
    X_tsne = tsne.fit_transform(X_scaled)  # t-SNE is done second, for plotting

    # Calculate all metrics. Internal metrics use X_scaled.
    silhouette, calinski_harabasz, davies_bouldin, ari, nmi = calculate_metrics(X_scaled, y_true_labels, kmeans_labels)

    return {
        'amino_acids': amino_acids_subset,
        'tsne_params': {'perplexity': perplexity, 'learning_rate': learning_rate, 'max_iter': max_iter},
        'silhouette': silhouette,
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin,
        'ari': ari,
        'nmi': nmi,
        'tsne_coords': X_tsne,  # t-SNE coords are still returned for plotting
        'kmeans_labels': kmeans_labels,
        'true_labels': y_true_labels,
        'filtered_df_empty': False  # Indicate valid result
    }


# --- Tkinter GUI Setup ---
class AminoAcidClusterApp:
    def __init__(self, master):
        self.master = master
        master.title("Amino Acid Cluster Analysis")
        master.geometry("1400x900")  # Increased window size for side-by-side plots

        self.df_original_loaded = False
        self.output_dir = None

        # --- File Load Frame ---
        self.file_frame = tk.Frame(master, bd=2, relief="groove", padx=10, pady=10)
        self.file_frame.pack(pady=10, fill="x")

        tk.Label(self.file_frame, text="1. Load Data:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.load_button = tk.Button(self.file_frame, text="Select Data CSV", command=self.load_data)
        self.load_button.pack(side="left", padx=5)
        self.file_path_label = tk.Label(self.file_frame, text="No file loaded.")
        self.file_path_label.pack(side="left", padx=5)

        # --- Quantity Selection Frame ---
        self.qty_frame = tk.Frame(master, bd=2, relief="groove", padx=10, pady=10)
        self.qty_frame.pack(pady=10, fill="x")

        tk.Label(self.qty_frame, text="2. Select Quantity for Clustering:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.qty_label = tk.Label(self.qty_frame, text="Number of amino acids to cluster (1-18):")
        self.qty_label.pack(side="left", padx=5)

        self.qty_spinner = ttk.Spinbox(self.qty_frame, from_=1, to=18, command=self.update_qty_value)
        self.qty_spinner.set(8)  # Default value
        self.qty_spinner.pack(side="left", padx=5)
        self.selected_qty = int(self.qty_spinner.get())

        self.start_button = tk.Button(self.qty_frame, text="Start Analysis", command=self.start_analysis_thread,
                                      state="disabled")
        self.start_button.pack(side="left", padx=20)

        # --- Progress Bar and Status ---
        self.progress_frame = tk.Frame(master, bd=2, relief="groove", padx=10, pady=10)
        self.progress_frame.pack(pady=10, fill="x")

        tk.Label(self.progress_frame, text="3. Analysis Progress:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=5)
        self.status_label = tk.Label(self.progress_frame, text="Ready.")
        self.status_label.pack(pady=5)

        # --- Results Display Frame (Scrollable) ---
        self.results_frame = tk.Frame(master, bd=2, relief="groove", padx=10, pady=10)
        self.results_frame.pack(pady=10, fill="both", expand=True)
        tk.Label(self.results_frame, text="4. Top 5 Optimized Results (Different Amino Acid Combinations):",
                 font=("Arial", 12, "bold")).pack(anchor="w")

        self.results_canvas = tk.Canvas(self.results_frame)
        self.results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_canvas.yview)
        self.results_scrollable_frame = ttk.Frame(self.results_canvas)

        self.results_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )
        self.results_canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)

        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")

    def load_data(self):
        global df_original, all_amino_acids
        file_path = select_file()
        if file_path:
            try:
                df_original = pd.read_csv(file_path)
                # Basic validation for required columns
                chiralities = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']
                intensity_cols = [f'{c}_intensity' for c in chiralities]
                shift_cols = [f'{c}_shift' for c in chiralities]
                feature_cols = intensity_cols + shift_cols
                required_cols = ['AA', '浓度/uM'] + feature_cols

                if not all(col in df_original.columns for col in required_cols):
                    missing_cols = [col for col in required_cols if col not in df_original.columns]
                    messagebox.showerror("Error", f"Required columns missing: {missing_cols}")
                    df_original = None  # Reset
                    self.df_original_loaded = False
                    self.file_path_label.config(text="Error loading file.")
                    self.start_button.config(state="disabled")
                    return

                df_original.dropna(subset=['AA'], inplace=True)
                df_original.fillna(0, inplace=True)  # Fill any other potential NaN values in features

                all_amino_acids = sorted(df_original['AA'].unique().tolist())
                if len(all_amino_acids) < 1:
                    messagebox.showerror("Error", "No amino acids found in the 'AA' column.")
                    df_original = None
                    self.df_original_loaded = False
                    self.file_path_label.config(text="Error: No AAs found.")
                    self.start_button.config(state="disabled")
                    return

                self.df_original_loaded = True
                self.file_path_label.config(text=f"Loaded: {os.path.basename(file_path)} (AAs: {len(all_amino_acids)})")
                self.qty_spinner.config(to=len(all_amino_acids))  # Adjust max spinbox value

                # Automatically set output directory (optional, user can change later if needed)
                self.output_dir = os.path.join(os.path.dirname(file_path), "cluster_results")
                os.makedirs(self.output_dir, exist_ok=True)
                self.start_button.config(state="normal")
                self.status_label.config(text="Data loaded. Ready for analysis.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")
                df_original = None
                self.df_original_loaded = False
                self.file_path_label.config(text="Failed to load.")
                self.start_button.config(state="disabled")
        else:
            self.file_path_label.config(text="No file selected.")

    def update_qty_value(self):
        try:
            self.selected_qty = int(self.qty_spinner.get())
            if self.selected_qty < 1 or self.selected_qty > len(all_amino_acids):
                self.start_button.config(state="disabled")
                self.status_label.config(text=f"Invalid quantity. Max: {len(all_amino_acids)}")
            elif self.df_original_loaded:
                self.start_button.config(state="normal")
                self.status_label.config(text="Quantity selected. Ready to start analysis.")
        except ValueError:
            self.start_button.config(state="disabled")
            self.status_label.config(text="Invalid input. Please enter a number.")

    def start_analysis_thread(self):
        if not self.df_original_loaded:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        if self.selected_qty < 2:  # KMeans needs at least 2 clusters
            messagebox.showwarning("Warning", "Please select at least 2 amino acids for clustering.")
            return
        if self.selected_qty > len(all_amino_acids):
            messagebox.showwarning("Warning",
                                   f"Selected quantity ({self.selected_qty}) exceeds available amino acids ({len(all_amino_acids)}). Please adjust.")
            return

        # Disable controls during analysis
        self.start_button.config(state="disabled")
        self.load_button.config(state="disabled")
        self.qty_spinner.config(state="disabled")

        self.progress_bar['value'] = 0
        self.status_label.config(text="Starting analysis...")

        # Clear previous results from display and internal storage
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()
        global current_analysis_results, best_results_for_display
        current_analysis_results = []
        best_results_for_display = []

        # Run analysis in a separate thread to keep GUI responsive
        analysis_thread = threading.Thread(target=self.run_full_analysis)
        analysis_thread.start()

    def run_full_analysis(self):
        global current_analysis_results, best_results_for_display, all_amino_acids

        num_combinations_to_sample = 50000  # Fixed max sample number
        top_combinations_for_tuning_count = 5  # Number of top combinations from phase 1 to fine-tune
        top_display_count = 5  # Number of final optimized results to display

        # Generate sampled combinations
        sampled_combinations = []
        try:
            all_combs = list(itertools.combinations(all_amino_acids, self.selected_qty))
            sampled_combinations = random.sample(all_combs, min(num_combinations_to_sample, len(all_combs)))
        except ValueError as e:
            self.status_label.config(text=f"Error generating combinations: {e}. Aborting analysis.")
            messagebox.showerror("Error", f"Error generating combinations: {e}. Analysis aborted.")
            self.master.after(100, self.reset_gui_after_analysis)
            return

        # --- Phase 1: Initial Screening with Fixed Parameters ---
        self.status_label.config(text="Phase 1: Initial screening with fixed parameters...")
        self.progress_bar['mode'] = 'determinate'

        # Fixed t-SNE parameters for initial screening
        phase1_tsne_params = {'perplexity': 30, 'learning_rate': 200, 'max_iter': 1000}

        phase1_raw_results = []
        for i, combo in enumerate(sampled_combinations):
            self.progress_bar['value'] = (i / len(sampled_combinations)) * 40  # 40% for phase 1
            self.status_label.config(
                text=f"Phase 1: Processing combination {i + 1}/{len(sampled_combinations)} ({', '.join(combo)})")

            result = run_tsne_kmeans_analysis(list(combo), **phase1_tsne_params)
            if not result['filtered_df_empty']:
                phase1_raw_results.append(result)
            self.master.update_idletasks()  # Update GUI

        # Filter phase 1 results based on criteria
        phase1_filtered_results = []
        for result in phase1_raw_results:
            ari_valid = result.get('ari') is not None and not np.isnan(result['ari'])
            nmi_valid = result.get('nmi') is not None and not np.isnan(result['nmi'])
            silhouette_valid = result.get('silhouette') is not None and not np.isnan(result['silhouette'])

            if (ari_valid and result['ari'] > 0.7 and
                    nmi_valid and result['nmi'] > 0.75 and
                    silhouette_valid and result['silhouette'] > 0.3):
                phase1_filtered_results.append(result)

        if not phase1_filtered_results:
            self.status_label.config(text="No combinations met initial screening criteria. Analysis complete.")
            messagebox.showinfo("No Results",
                                "No clustering combinations met the initial ARI > 0.8, NMI > 0.75, and Silhouette > 0.5 criteria. Try relaxing criteria or check your data.")
            self.master.after(100, self.reset_gui_after_analysis)
            return

        # V4.0 Change: Sort phase 1 filtered results by Silhouette, then CH, then DB (descending)
        # This aligns with the final goal of maximizing Silhouette
        def phase1_sort_key(res):
            silhouette = res.get('silhouette', -np.inf)
            calinski_harabasz = res.get('calinski_harabasz', -np.inf)
            davies_bouldin = res.get('davies_bouldin', np.inf)
            return (silhouette, calinski_harabasz, -davies_bouldin)  # Silhouette and CH high, DB low

        phase1_filtered_results.sort(key=phase1_sort_key, reverse=True)

        # Get the top N combinations for tuning based on this new sorting
        top_combinations_for_tuning_details = phase1_filtered_results[:top_combinations_for_tuning_count]
        # Extract just the amino acid list for phase 2 iteration
        top_combinations_for_tuning = [res['amino_acids'] for res in top_combinations_for_tuning_details]

        if not top_combinations_for_tuning:
            self.status_label.config(text="No top combinations selected for fine-tuning. Analysis complete.")
            self.master.after(100, self.reset_gui_after_analysis)
            return

        # --- Phase 2: Fine-tuning Parameters for Selected Combinations ---
        self.status_label.config(
            text=f"Phase 2: Fine-tuning parameters for top {len(top_combinations_for_tuning)} combinations...")

        tsne_param_grid = {
            'perplexity': [10, 30, 50],
            'learning_rate': [100, 200, 500],
            'max_iter': [1000, 2000]
        }
        all_tsne_param_combinations = list(itertools.product(
            tsne_param_grid['perplexity'],
            tsne_param_grid['learning_rate'],
            tsne_param_grid['max_iter']
        ))

        # This list will store the BEST result for EACH of the top N amino acid combinations
        final_optimized_results = []

        total_tuning_runs = len(top_combinations_for_tuning) * len(all_tsne_param_combinations)
        current_tuning_run_count = 0

        for i, combo in enumerate(top_combinations_for_tuning):
            best_silhouette_for_combo = -np.inf
            best_calinski_for_combo = -np.inf
            best_davies_for_combo = np.inf  # Lower is better for DB
            best_result_for_current_combo = None

            for params_tuple in all_tsne_param_combinations:
                current_tuning_run_count += 1
                # Progress bar update for phase 2 (remaining 60%)
                self.progress_bar['value'] = 40 + (current_tuning_run_count / total_tuning_runs) * 60

                params = {'perplexity': params_tuple[0], 'learning_rate': params_tuple[1], 'max_iter': params_tuple[2]}
                self.status_label.config(
                    text=f"Tuning: {', '.join(combo)} with params {params} ({current_tuning_run_count}/{total_tuning_runs})")

                result = run_tsne_kmeans_analysis(list(combo), **params)

                if result['filtered_df_empty']:
                    continue  # Skip invalid results

                current_silhouette = result.get('silhouette') if result.get('silhouette') is not None and not np.isnan(
                    result['silhouette']) else -np.inf
                current_calinski = result.get('calinski_harabasz') if result.get(
                    'calinski_harabasz') is not None and not np.isnan(result['calinski_harabasz']) else -np.inf
                current_davies = result.get('davies_bouldin') if result.get(
                    'davies_bouldin') is not None and not np.isnan(result['davies_bouldin']) else np.inf

                # Determine if this is the best result for the current amino acid combination (tuning)
                is_better = False
                if current_silhouette > best_silhouette_for_combo:
                    is_better = True
                elif current_silhouette == best_silhouette_for_combo:
                    if current_calinski > best_calinski_for_combo:
                        is_better = True
                    elif current_calinski == best_calinski_for_combo:
                        if current_davies < best_davies_for_combo:  # Lower DB is better
                            is_better = True

                if is_better:
                    best_silhouette_for_combo = current_silhouette
                    best_calinski_for_combo = current_calinski
                    best_davies_for_combo = current_davies
                    best_result_for_current_combo = result

            if best_result_for_current_combo:  # If an optimal result was found for this combination
                final_optimized_results.append(best_result_for_current_combo)

            self.master.update_idletasks()  # Update GUI progress

        if not final_optimized_results:
            self.status_label.config(text="No combinations survived fine-tuning. Analysis complete.")
            messagebox.showinfo("No Results",
                                "After fine-tuning, no combinations yielded valid results. Try relaxing criteria or check your data.")
            self.master.after(100, self.reset_gui_after_analysis)
            return

        # --- Final Sorting of Optimized Results ---
        self.status_label.config(text="Final sorting of optimized results...")

        # Custom sorting based on criteria: Silhouette (priority >0.6), then CH (high), then DB (low)
        def final_sort_key(res):
            silhouette = res.get('silhouette', -np.inf)
            calinski_harabasz = res.get('calinski_harabasz', -np.inf)
            davies_bouldin = res.get('davies_bouldin', np.inf)

            # Prioritize Silhouette > 0.6 by giving it a higher primary sort value
            priority_group = 1 if silhouette > 0.6 else 0

            # Then sort by (priority_group, silhouette, calinski_harabasz, -davies_bouldin)
            return (priority_group, silhouette, calinski_harabasz, -davies_bouldin)

        final_optimized_results.sort(key=final_sort_key, reverse=True)

        # Select top N for final display
        best_results_for_display = final_optimized_results[:top_display_count]
        current_analysis_results = best_results_for_display  # Update global variable

        self.status_label.config(text="Analysis complete. Displaying top results...")
        self.progress_bar['value'] = 100
        self.display_results()
        self.master.after(100, self.reset_gui_after_analysis)  # Reset GUI state

    def reset_gui_after_analysis(self):
        """Re-enables controls and updates status after analysis completes."""
        self.start_button.config(state="normal")
        self.load_button.config(state="normal")
        self.qty_spinner.config(state="normal")
        self.status_label.config(text="Analysis finished. Ready for new analysis.")

    def display_results(self):
        # Clear previous results
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()

        if not best_results_for_display:
            tk.Label(self.results_scrollable_frame, text="No valid results to display.", font=("Arial", 12)).pack(
                pady=20)
            return

        for i, result in enumerate(best_results_for_display):
            result_frame = tk.LabelFrame(self.results_scrollable_frame,
                                         text=f"Result {i + 1} (Sil: {result['silhouette']:.4f})", bd=2, relief="solid",
                                         padx=10, pady=10)  # Update frame title to show Silhouette
            result_frame.pack(pady=10, fill="x", padx=5)

            tk.Label(result_frame, text=f"Amino Acids: {', '.join(result['amino_acids'])}",
                     font=("Arial", 10, "bold")).pack(anchor="w")
            tk.Label(result_frame,
                     text=f"t-SNE Params (for visualization): Perplexity={result['tsne_params']['perplexity']}, Learning Rate={result['tsne_params']['learning_rate']}, Max Iter={result['tsne_params']['max_iter']}",
                     font=("Arial", 10)).pack(anchor="w")
            # Update metrics display to include ARI and NMI
            tk.Label(result_frame,
                     text=f"Metrics: Silhouette={result['silhouette']:.4f}, CH={result['calinski_harabasz']:.0f}, DB={result['davies_bouldin']:.4f}, ARI={result['ari']:.4f}, NMI={result['nmi']:.4f}",
                     font=("Arial", 10)).pack(anchor="w")

            # Create a single figure with two subplots side-by-side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=100)  # Increased figure size for two plots

            # --- Plot 1: KMeans Clustering Results (KMeans Labels) ---
            plot_title_kmeans = f"KMeans Clustering (K={len(set(result['kmeans_labels']))}) on Original Features" \
                                f"\nSil: {result['silhouette']:.2f}, CH: {result['calinski_harabasz']:.0f}, DB: {result['davies_bouldin']:.2f}" \
                                f"\nARI: {result['ari']:.2f}, NMI: {result['nmi']:.2f}"

            num_clusters = len(set(result['kmeans_labels']))
            cmap_kmeans = plt.cm.get_cmap('tab20', num_clusters) if num_clusters <= 20 else plt.cm.get_cmap('gist_ncar',
                                                                                                            num_clusters)

            # Scatter plot using t-SNE coordinates for visualization, but colored by KMeans labels from original features
            ax1.scatter(result['tsne_coords'][:, 0], result['tsne_coords'][:, 1], c=result['kmeans_labels'],
                        cmap=cmap_kmeans, s=50, alpha=0.8)
            ax1.set_title(plot_title_kmeans, fontsize=12, fontweight='bold', fontname='Arial')
            ax1.set_xlabel('t-SNE Component 1', fontname='Arial', fontsize=10)  # Clarify x-axis is t-SNE
            ax1.set_ylabel('t-SNE Component 2', fontname='Arial', fontsize=10)  # Clarify y-axis is t-SNE
            ax1.tick_params(axis='both', which='major', labelsize=8)
            ax1.grid(True, linestyle='--', alpha=0.6)

            # --- Plot 2: True Amino Acid Labels on t-SNE ---
            plot_title_true = f"True Amino Acid Labels on t-SNE" \
                              f"\n(Comparison to KMeans on Original: ARI={result['ari']:.2f}, NMI={result['nmi']:.2f})"

            unique_true_labels_sorted = sorted(list(set(result['true_labels'])))
            num_true_labels = len(unique_true_labels_sorted)

            true_label_to_color_idx = {label: i for i, label in enumerate(unique_true_labels_sorted)}
            colors_for_points = [true_label_to_color_idx[label] for label in result['true_labels']]

            cmap_true = plt.cm.get_cmap('tab20', num_true_labels) if num_true_labels <= 20 else plt.cm.get_cmap(
                'gist_ncar', num_true_labels)

            scatter_true = ax2.scatter(result['tsne_coords'][:, 0], result['tsne_coords'][:, 1], c=colors_for_points,
                                       cmap=cmap_true, s=50, alpha=0.8)

            # Create custom legend handles for true labels
            handles = []
            for j, label_name in enumerate(unique_true_labels_sorted):
                color = cmap_true(j / (num_true_labels - 1) if num_true_labels > 1 else 0.5)
                handles.append(Patch(facecolor=color, edgecolor='black', label=label_name))
            ax2.legend(handles=handles, title="Amino Acid (True Label)", fontsize=10, title_fontsize=12,
                       loc="upper right")

            ax2.set_title(plot_title_true, fontsize=12, fontweight='bold', fontname='Arial')
            ax2.set_xlabel('t-SNE Component 1', fontname='Arial', fontsize=10)
            ax2.set_ylabel('t-SNE Component 2', fontname='Arial', fontsize=10)
            ax2.tick_params(axis='both', which='major', labelsize=8)
            ax2.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()  # Adjust subplots to fit into figure area.

            # Embed the combined plot in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=result_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(pady=5)
            canvas.draw()

            # --- Save plots and TXT for this specific result ---
            if self.output_dir:
                filename_prefix_aas = "_".join(aa.replace('L-', '') for aa in result['amino_acids'])

                # Plot filename - now includes Silhouette score in filename
                plot_filename_combined = f"Combined_Plot_Result_{i + 1}_K{len(set(result['kmeans_labels']))}_Sil{result['silhouette']:.4f}_{filename_prefix_aas}.png"
                fig.savefig(os.path.join(self.output_dir, plot_filename_combined), dpi=300, bbox_inches='tight')
                print(f"Saved combined plot: {os.path.join(self.output_dir, plot_filename_combined)}")

                # TXT file for this specific result
                results_filename = f"Clustering_Data_Result_{i + 1}_K{len(set(result['kmeans_labels']))}_Sil{result['silhouette']:.4f}_{filename_prefix_aas}.txt"
                results_file_path = os.path.join(self.output_dir, results_filename)

                with open(results_file_path, 'w') as f:
                    f.write(
                        f"KMeans Clustering Analysis Results (K={len(set(result['kmeans_labels']))} Clusters) on Original Features\n")
                    f.write(f"-----------------------------------------------------------\n")
                    f.write(f"Amino Acids in this Combination: {', '.join(result['amino_acids'])}\n")
                    f.write(
                        f"t-SNE Parameters (for visualization only): Perplexity={result['tsne_params']['perplexity']}, Learning Rate={result['tsne_params']['learning_rate']}, Max Iter={result['tsne_params']['max_iter']}\n")
                    f.write(f"Evaluation Metrics:\n")
                    f.write(f"  Silhouette Score: {result['silhouette']:.4f}\n")
                    f.write(f"  Calinski-Harabasz Index: {result['calinski_harabasz']:.4f}\n")
                    f.write(f"  Davies-Bouldin Index: {result['davies_bouldin']:.4f}\n")
                    f.write(f"  Adjusted Rand Index (ARI): {result['ari']:.4f}\n")
                    f.write(f"  Normalized Mutual Information (NMI): {result['nmi']:.4f}\n")
                    f.write(f"-----------------------------------------------------------\n")
                    f.write(f"Data Points (t-SNE Coordinates, KMeans Cluster, True Label):\n")
                    f.write(
                        f"{'Amino_Acid':<15}\t{'tSNE_Component_1':<20}\t{'tSNE_Component_2':<20}\t{'KMeans_Cluster_Label':<25}\n")
                    f.write(f"{'-' * 15}\t{'-' * 20}\t{'-' * 20}\t{'-' * 25}\n")  # Separator line

                    for j in range(len(result['true_labels'])):
                        f.write(
                            f"{result['true_labels'][j]:<15}\t{result['tsne_coords'][j, 0]:<20.8f}\t{result['tsne_coords'][j, 1]:<20.8f}\t{result['kmeans_labels'][j]:<25}\n")
                print(f"Saved TXT: {results_file_path}")

        # Ensure scrollable frame updates its size after all widgets are packed
        self.results_scrollable_frame.update_idletasks()
        self.results_canvas.config(scrollregion=self.results_canvas.bbox("all"))


def start_app():
    root = tk.Tk()
    app = AminoAcidClusterApp(root)
    root.mainloop()


if __name__ == "__main__":
    start_app()