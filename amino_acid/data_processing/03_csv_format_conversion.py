import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import os
import re
import unicodedata


def standardize_column_name(col_name):
    """
    Standardizes column names by converting to lowercase, replacing common symbols
    and words with a consistent short form, and removing non-alphanumeric characters.
    """
    cleaned_name = col_name.lower()

    # Handle specific symbols and their common word variations
    cleaned_name = cleaned_name.replace('delta', 'd')
    cleaned_name = cleaned_name.replace('lambda', 'l')
    cleaned_name = cleaned_name.replace('mu', 'u')

    # Handle Unicode symbols directly
    cleaned_name = cleaned_name.replace('δ', 'd')
    cleaned_name = cleaned_name.replace('Δ', 'd')
    cleaned_name = cleaned_name.replace('λ', 'l')

    # Remove all characters that are not letters (a-z), numbers (0-9),
    # or Chinese characters (U+4e00 to U+9fff).
    cleaned_name = re.sub(r'[^\w\u4e00-\u9fff]', '', cleaned_name)

    return cleaned_name


def get_standardized_col_map(df_columns):
    """
    Creates a mapping from standardized internal column names to actual column names
    found in the DataFrame, allowing for flexible column identification.
    """
    col_map = {}
    expected_cols_variations = {
        'aa': ['aa'],
        '浓度um': ['浓度um', '浓度ui', '浓度u', '浓度'],
        'ii0i': ['ii0i', 'ii0i0', 'ii0'],
        'dl': ['dl', 'dlambda', 'delta']
    }

    for actual_col in df_columns:
        standardized_actual_col = standardize_column_name(actual_col)
        for std_internal_name, variations in expected_cols_variations.items():
            if standardized_actual_col in variations:
                col_map[std_internal_name] = actual_col
                break
    return col_map


def calculate_norm_std_dev(intensity_vals, shift_vals):
    """
    Calculates the standard deviation of the norms of (intensity, shift) pairs.
    """
    if len(intensity_vals) < 2 or len(shift_vals) < 2:  # Need at least 2 points to calculate std dev
        return np.inf  # Return infinity for insufficient data, so it's not chosen

    # Calculate the norm for each (intensity, shift) pair
    norms = np.sqrt(np.array(intensity_vals) ** 2 + np.array(shift_vals) ** 2)
    return np.std(norms)


def select_best_3_measurements(measurements):
    """
    Selects the 3 measurements with the minimum standard deviation of their norms.
    Each measurement is a tuple: (intensity, shift).
    If fewer than 3 measurements are available, all available are returned.
    """
    if len(measurements) <= 3:
        return measurements  # If 3 or fewer, return all

    best_std_dev = np.inf
    best_subset = []

    # Iterate through all combinations of 3 measurements
    from itertools import combinations
    for subset_indices in combinations(range(len(measurements)), 3):
        subset = [measurements[i] for i in subset_indices]

        subset_intensity = [m[0] for m in subset]
        subset_shift = [m[1] for m in subset]

        current_std_dev = calculate_norm_std_dev(subset_intensity, subset_shift)

        if current_std_dev < best_std_dev:
            best_std_dev = current_std_dev
            best_subset = subset

    return best_subset


def process_excel_to_csv_format(file_path):
    """
    Processes an Excel file: merges data from specified sheets,
    filters incomplete rows, selects the 3 best measurements per AA-concentration
    combination based on minimum error (standard deviation of norms),
    and converts to a unified CSV format.
    """
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        expected_sheets = ['(6,5)', '(7,5)', '(8,3)', 'S7-(6,5)']

        if not all(sheet in sheet_names for sheet in expected_sheets):
            missing_sheets = [sheet for sheet in expected_sheets if sheet not in sheet_names]
            messagebox.showerror("Error", f"Excel file is missing expected sheets: {', '.join(missing_sheets)}")
            return None

        all_hand_data = {}

        for sheet_name in expected_sheets:
            try:
                df = xls.parse(sheet_name)
                print(f"Original columns for sheet '{sheet_name}': {df.columns.tolist()}")

                col_map = get_standardized_col_map(df.columns)
                print(f"Standardized column map for sheet '{sheet_name}': {col_map}")

                required_found = True
                if 'aa' not in col_map: required_found = False
                if '浓度um' not in col_map: required_found = False
                if 'ii0i' not in col_map: required_found = False
                if 'dl' not in col_map: required_found = False

                if not required_found:
                    missing_cols_display = []
                    if 'aa' not in col_map: missing_cols_display.append('AA')
                    if '浓度um' not in col_map: missing_cols_display.append('浓度/uM')
                    if 'ii0i' not in col_map: missing_cols_display.append('(I-I0)/I or (I-I0)/I0')
                    if 'dl' not in col_map: missing_cols_display.append('Δλ')
                    messagebox.showerror("Error",
                                         f"Sheet '{sheet_name}' is missing required columns: {', '.join(missing_cols_display)}.")
                    return None

                df_standardized = df[[col_map['aa'], col_map['浓度um'], col_map['ii0i'], col_map['dl']]].copy()
                df_standardized.columns = ['AA', '浓度/uM', '(I-I0)/I', 'Δλ']

                df_standardized['浓度/uM'] = pd.to_numeric(df_standardized['浓度/uM'], errors='coerce')
                df_standardized['(I-I0)/I'] = pd.to_numeric(df_standardized['(I-I0)/I'], errors='coerce')
                df_standardized['Δλ'] = pd.to_numeric(df_standardized['Δλ'], errors='coerce')

                all_hand_data[sheet_name] = df_standardized.dropna(subset=['AA', '浓度/uM', '(I-I0)/I', 'Δλ']).copy()
                print(f"Successfully read sheet '{sheet_name}'.")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while processing sheet '{sheet_name}': {e}")
                return None

        all_aas = set()
        all_concs = set()
        for df in all_hand_data.values():
            all_aas.update(df['AA'].unique())
            all_concs.update(df['浓度/uM'].unique())

        final_data_list = []

        for aa in sorted(list(all_aas)):
            for conc in sorted(list(all_concs)):
                combined_measurements_for_selection = {}  # Stores all (intensity, shift) pairs per hand for selection

                for hand in expected_sheets:
                    if hand in all_hand_data:
                        filtered_df = all_hand_data[hand][
                            (all_hand_data[hand]['AA'] == aa) &
                            (all_hand_data[hand]['浓度/uM'] == conc)
                            ]
                        # Collect all (intensity, shift) tuples for the current hand, AA, and conc
                        measurements = list(zip(filtered_df['(I-I0)/I'].tolist(), filtered_df['Δλ'].tolist()))
                        combined_measurements_for_selection[hand] = measurements
                    else:
                        combined_measurements_for_selection[hand] = []

                # Now, for each hand, select the best 3 measurements
                selected_measurements_per_hand = {}
                for hand, measurements in combined_measurements_for_selection.items():
                    selected_measurements_per_hand[hand] = select_best_3_measurements(measurements)

                # Determine the maximum number of selected measurements across all hands for this AA-conc combo
                max_selected_measurements = 0
                for measurements in selected_measurements_per_hand.values():
                    max_selected_measurements = max(max_selected_measurements, len(measurements))

                # Build rows for the final DataFrame
                if max_selected_measurements > 0:
                    for i in range(max_selected_measurements):
                        row = {'AA': aa, '浓度/uM': conc}
                        for hand in expected_sheets:
                            # Safely get intensity and shift, filling NaN if not available
                            intensity_val = selected_measurements_per_hand[hand][i][0] if i < len(
                                selected_measurements_per_hand[hand]) else np.nan
                            shift_val = selected_measurements_per_hand[hand][i][1] if i < len(
                                selected_measurements_per_hand[hand]) else np.nan

                            row[f'{hand}_intensity'] = intensity_val
                            row[f'{hand}_shift'] = shift_val
                        final_data_list.append(row)
                else:
                    # If no valid measurements found after selection (e.g., all dropped), add a row with NaNs
                    row = {'AA': aa, '浓度/uM': conc}
                    for hand in expected_sheets:
                        row[f'{hand}_intensity'] = np.nan
                        row[f'{hand}_shift'] = np.nan
                    final_data_list.append(row)

        if final_data_list:
            return pd.DataFrame(final_data_list)
        else:
            messagebox.showinfo("Tip", "No valid data found to generate CSV.")
            return None

    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {file_path}")
        return None
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while reading or processing the Excel file: {e}")
        return None


def select_file():
    """Opens a file dialog to select an Excel file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select the Excel file containing chiral carbon tube data",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    root.destroy()
    return file_path


def select_save_path(default_filename="processed_filtered_data.csv"):
    """Opens a file dialog to select the save path and filename for the processed data."""
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        title="Select the path and filename to save the processed data",
        defaultextension=".csv",
        initialfile=default_filename,
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    return save_path


def main():
    """Main function to control the script's execution flow."""
    excel_file_path = select_file()

    if excel_file_path:
        processed_df = process_excel_to_csv_format(excel_file_path)

        if processed_df is not None:
            print("\nProcessed and filtered data (first 10 rows):")
            print(processed_df.head(10))

            save_file_path = select_save_path()
            if save_file_path:
                try:
                    processed_df.to_csv(save_file_path, index=False, encoding='utf-8')
                    messagebox.showinfo("Success",
                                        f"Data successfully processed, filtered, and saved to:\n{save_file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred while saving the file: {e}")
            else:
                messagebox.showinfo("Info", "Save path not selected, data not saved.")
        else:
            messagebox.showinfo("Info", "No valid data generated.")
    else:
        messagebox.showinfo("Info", "No Excel file selected, program exited.")


if __name__ == "__main__":
    main()
