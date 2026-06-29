import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import itertools
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

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
    """Opens a folder selection dialog for the user to choose a directory for saving results."""
    root = tk.Tk()
    root.withdraw()
    save_dir = filedialog.askdirectory(
        title="Select Directory to Save Results"
    )
    return save_dir

def find_optimal_amino_acid_combinations():
    """
    Finds optimal combinations of 3-6 amino acids that can be accurately classified by XGBoost.
    Modified to always take top N combinations per group size if 90% threshold isn't met.
    """
    input_file_path = select_file()

    if not input_file_path:
        messagebox.showinfo("Info", "No training data file selected, program exited.")
        return

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

    target_col = 'AA'

    # --- Wide to Long format conversion starts ---
    long_format_data = []

    for index, row in df_original.iterrows():
        aa_label = row['AA']
        concentration = row['浓度/uM'] # Keeping for completeness, though not a feature

        for chirality in chiralities:
            intensity_col_name = f'{chirality}_intensity'
            shift_col_name = f'{chirality}_shift'

            # Ensure the columns exist before accessing
            if intensity_col_name in row and shift_col_name in row:
                new_record = {
                    'AA': aa_label,
                    '浓度/uM': concentration,
                    '碳管手性': chirality,
                    'intensity': row[intensity_col_name],
                    'shift': row[shift_col_name]
                }
                long_format_data.append(new_record)
            else:
                print(f"Warning: Missing data for {intensity_col_name} or {shift_col_name} in row {index}. Skipping.")

    df_long = pd.DataFrame(long_format_data)
    # --- Wide to Long format conversion ends ---

    print("\nTransformed Long Format Data Info:")
    print(df_long.info())
    print("\nFirst 5 Rows of Transformed Long Format Data:")
    print(df_long.head())
    print(f"\nTotal samples after transformation: {len(df_long)} (Original samples * 4)")

    # Prepare features and labels
    feature_cols = ['intensity', 'shift']
    X_full = df_long[feature_cols].copy()
    y_labels_full = df_long[target_col].copy()

    # Handle missing values if any
    X_full.fillna(0, inplace=True)

    all_amino_acids = sorted(y_labels_full.unique())
    print(f"\nFound {len(all_amino_acids)} unique amino acids: {all_amino_acids}")

    # Results storage for each combination length
    results_by_length = {r: [] for r in range(3, 7)}
    all_top_combinations = [] # To store the final 20 (or fewer) results

    # Iterate through combinations of 3 to 6 amino acids
    for r in range(3, 7): # Combinations of 3, 4, 5, 6
        print(f"\n--- Checking combinations of size {r} ---")
        current_length_combinations = [] # Store all combinations for this length

        for combo in itertools.combinations(all_amino_acids, r):
            current_amino_acids = list(combo)

            # Filter data for current combination
            df_combo = df_long[df_long['AA'].isin(current_amino_acids)].copy()

            if df_combo.empty:
                continue # Skip empty combinations

            X_combo = df_combo[feature_cols].copy()
            y_labels_combo = df_combo[target_col].copy()

            # Encode labels specifically for this subset
            subset_label_encoder = LabelEncoder()
            y_combo = subset_label_encoder.fit_transform(y_labels_combo)

            # Scale features
            scaler = StandardScaler()
            X_scaled_combo = scaler.fit_transform(X_combo)

            # Check if there's enough data for cross-validation
            if len(np.unique(y_combo)) < 2:
                # Need at least 2 classes for classification
                continue
            # Also check if any class has fewer than n_splits samples for StratifiedKFold
            # Count occurrences of each class
            class_counts = pd.Series(y_combo).value_counts()
            if any(count < 5 for count in class_counts): # 5 is n_splits
                # print(f"  Skipping combo {combo}: At least one class has fewer than 5 samples for CV.")
                continue


            # StratifiedKFold for cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # XGBoost Model (using default parameters for quick evaluation)
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

            try:
                # Perform 5-fold cross-validation
                scores = cross_val_score(model, X_scaled_combo, y_combo, cv=skf, scoring='accuracy', n_jobs=-1)
                mean_accuracy = np.mean(scores)

                current_length_combinations.append({
                    'amino_acids': combo,
                    'mean_accuracy': mean_accuracy,
                    'num_classes': len(combo)
                })
                # print(f"  Combo: {combo}, Accuracy: {mean_accuracy:.4f}") # 打印所有组合，可选

            except Exception as e:
                print(f"  Error evaluating combo {combo}: {e}")
                continue

        # Sort all combinations for the current length by accuracy in descending order
        current_length_combinations.sort(key=lambda x: x['mean_accuracy'], reverse=True)

        # Take the top 5 combinations for this length
        top_5_for_this_length = current_length_combinations[:5]
        results_by_length[r] = top_5_for_this_length
        all_top_combinations.extend(top_5_for_this_length) # Add to the final list


    # Sort all collected top combinations (max 20) by accuracy in descending order
    all_top_combinations.sort(key=lambda x: x['mean_accuracy'], reverse=True)

    print("\n--- Final Results: Top Combinations per Size Group ---")
    if not all_top_combinations:
        print("No valid combinations found.")
    else:
        # Output the best from each group (max 20)
        print("Top combinations for each size group (3, 4, 5, 6), sorted overall by accuracy:")
        for i, result in enumerate(all_top_combinations):
            print(f"{i+1}. Amino Acids: {result['amino_acids']}, Classes: {result['num_classes']}, Accuracy: {result['mean_accuracy']:.4f}")

        # --- 添加保存路径选择 ---
        save_dir = get_save_directory() # 询问保存路径
        if save_dir:
            output_file = os.path.join(save_dir, "top_amino_acid_combinations_by_length.txt")
            with open(output_file, 'w') as f:
                f.write("Top Amino Acid Combinations (Sorted Overall by Accuracy):\n\n")
                if not all_top_combinations:
                    f.write("No valid combinations found.\n")
                else:
                    for i, result in enumerate(all_top_combinations):
                        f.write(f"{i+1}. Amino Acids: {result['amino_acids']}, Classes: {result['num_classes']}, Accuracy: {result['mean_accuracy']:.4f}\n")
            print(f"\nResults saved to: {output_file}")
        else:
            print("\nNo save directory selected, results not saved to file.")

if __name__ == "__main__":
    # Ensure XGBoost is installed
    try:
        import xgboost
    except ImportError:
        print("XGBoost is not installed. Please install it: pip install xgboost")
        sys.exit(1)

    find_optimal_amino_acid_combinations()