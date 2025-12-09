import sys
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import re
import numpy as np
import matplotlib.colors as mcolors
import os
import glob

def plot_transfer_data(file, name, cc, ax):
    """Plot transfer characteristics (Id vs Vgs) with sqrt(Id) and V_th/SS extraction."""
    sheets = [sheet for sheet in file.keys() if sheet not in ['Calc', 'Settings']]
    p = np.pi

    # Define colors, they are shades of a given color
    aux0 = plt.get_cmap(cc)(np.linspace(0.8, 0.9, len(sheets)))
    colors = [mcolors.to_hex(color) for color in aux0]

    num0 = re.search(r'\d+(\.\d+)?', name)
    if num0:
        TE = float(num0.group())

    # Define a list of markers for distinction
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'p', '8']
    for i, (sheet, color) in enumerate(zip(sheets, colors)):
        df = file[sheet]
        ax.set_yscale('log')
        marker = markers[i % len(markers)]
        ax.plot(df['GateV'], df['DrainI'], color=color, marker=marker)#, label=sheet)

    # Plot sqrt(Id) on a secondary y-axis
    ax2 = ax.twinx()
    for sheet, color in zip(sheets, colors):
        df = file[sheet]
        if 'GateV' not in df.columns or 'DrainI' not in df.columns:
            continue
        v = pd.to_numeric(df['GateV'], errors='coerce').values
        i_vals = pd.to_numeric(df['DrainI'], errors='coerce').values
        # Mask invalid entries
        mask = np.isfinite(v) & np.isfinite(i_vals)
        if mask.sum() < 2:
            continue
        v_clean = v[mask]
        i_clean = i_vals[mask]
        
        sqrt_id = np.sqrt(i_clean)

        # Linear-region detection using sliding window linear regression
        n = len(v_clean)
        min_win = max(5, int(0.1 * n))
        win = max(min_win, int(0.3 * n))
        if win >= n:
            win = n

        best_r2 = -np.inf
        best_slice = None
        for start in range(0, n - win + 1):
            xw = v_clean[start:start + win]
            yw = sqrt_id[start:start + win]
            # require variability
            if np.allclose(yw, yw[0]):
                continue
            coef = np.polyfit(xw, yw, 1)
            ypred = np.polyval(coef, xw)
            ss_res = np.sum((yw - ypred) ** 2)
            ss_tot = np.sum((yw - np.mean(yw)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            if r2 > best_r2:
                best_r2 = r2
                best_slice = (start, start + win, coef)

        # Compute V_th first (needed for SS calculation)
        x_intercept = np.nan
        if best_slice is not None and best_r2 > 0.90:
            s0, s1, coef = best_slice
            m, b = coef[0], coef[1]
            if abs(m) > 1e-12:
                x_intercept = -b / m
        
        # Calculate Subthreshold Swing (SS) using positive I and V < V_th
        ss_value = np.nan
        if not np.isnan(x_intercept):
            # Filter: positive current AND voltage below V_th
            ss_mask = (i_clean > 0) & (v_clean < x_intercept)
            if ss_mask.sum() > 5:  # Need at least a few points
                v_ss = v_clean[ss_mask]
                i_ss = i_clean[ss_mask]
                log_i_ss = np.log10(i_ss)
                
                # Linear fit: log10(I) = m*V + b, so SS = 1/m * 1000 mV/decade
                ss_coef = np.polyfit(v_ss, log_i_ss, 1)
                ss_slope = ss_coef[0]
                if abs(ss_slope) > 1e-9:
                    ss_value = (1.0 / ss_slope) * 1000  # Convert to mV/decade

        # Plot sqrt(Id) raw data (lighter alpha)
        ax2.plot(v_clean, sqrt_id, color=color, linestyle='-', linewidth=2, alpha=0.75)

        # If a good linear region was found, fit and plot the fit and x-intercept
        if best_slice is not None and best_r2 > 0.90:
            s0, s1, coef = best_slice
            m, b = coef[0], coef[1]
            # skip non-positive slope
            if abs(m) > 1e-12:
                # Extrapolate line from fitted region to x-intercept
                v_start = min(v_clean[s0], x_intercept)
                v_end = max(v_clean[s1 - 1], x_intercept)
                v_fit = np.linspace(v_start, v_end, 100)
                y_fit = m * v_fit + b
                ax2.plot(v_fit, y_fit, color=color, linestyle='--', linewidth=2.0, alpha=0.5)
                
                ymax = ax.get_ylim()[0]
                
                # Display V_th and SS together
                if not np.isnan(ss_value):
                    label_text = f'V$_{{th}}$={x_intercept:.2f} V'
                else:
                    label_text = f'V$_{{th}}$={x_intercept:.2f} V'

                ax.text(x_intercept, ymax, label_text, rotation=0,
                        verticalalignment='bottom', horizontalalignment='left',
                        color=color, fontsize=12)
        else:
            # if no good linear region, still plot raw sqrt data more prominently
            ax2.plot(v_clean, sqrt_id, color=color, linestyle='-', alpha=0.6)
    
    ax2.set_ylabel(r'Sqrt(Ids) (A$^{1/2}$)', fontsize=16)
    ax2.tick_params(axis='y', labelsize=14)

    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Vgs (V)', fontsize=16)
    ax.set_ylabel('Ids (A)', fontsize=16)
    ax.set_title('Transfer Characteristics', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=14, framealpha=1)


def plot_output_data(file, name, cc, ax):
    """Plot output characteristics (Id vs Vds)."""
    sheets = [sheet for sheet in file.keys() if sheet not in ['Calc', 'Settings']]
    p = np.pi

    # Define colors, they are shades of a given color
    aux0 = plt.get_cmap(cc)(np.linspace(0.8, 0.9, len(sheets)))
    colors = [mcolors.to_hex(color) for color in aux0]

    num0 = re.search(r'\d+(\.\d+)?', name)
    if num0:
        TE = float(num0.group())

    # Define a list of markers for distinction
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'p', '8']
    for i, (sheet, color) in enumerate(zip(sheets, colors)):
        df = file[sheet]
        marker = markers[i % len(markers)]
        ax.plot(df['DrainV'], df['DrainI'], color=color, marker=marker)#, label=sheet)

    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Drain Voltage (V)', fontsize=16)
    ax.set_ylabel('Drain Current (A)', fontsize=16)
    ax.set_title('Output Characteristics', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=14, framealpha=1)


def detect_measurement_type(filename):
    """
    Automatically detect if file is transfer or output based on filename.
    
    Transfer: contains 'vgs' and 'id' (gate sweep)
    Output: contains 'vds' and 'id' (drain sweep)
    Skip: contains 'after' in filename
    
    Returns: 'transfer', 'output', or 'skip'
    """
    filename_lower = filename.lower()
    
    # Skip files with 'after' in the name
    if 'after' in filename_lower:
        return 'skip'
    
    # Check for output characteristics (Vds-Id)
    if ('vds' in filename_lower and 'id' in filename_lower):
        return 'output'
    
    # Check for transfer characteristics (Vgs-Id)
    if ('vgs' in filename_lower and 'id' in filename_lower):
        return 'transfer'
    
    # Fallback: skip unknown files
    print(f"Warning: Could not auto-detect type for '{filename}'. Skipping.")
    return 'skip'


def main():
    """Main execution with automatic mode detection from filenames."""
    if len(sys.argv) < 2:
        print("Usage: python FET-plot-combined.py <directory|file1.xls> [file2.xls ...]")
        print("\nAutomatic mode detection from filename:")
        print("  - Files with 'vgs' and 'id' → Transfer characteristics")
        print("  - Files with 'vds' and 'id' → Output characteristics")
        print("  - Files with 'after' → Skipped")
        print("\nExamples:")
        print("  python FET-plot-combined.py L40um")
        print("  python FET-plot-combined.py vgs-id#1@2.xls vds-id#1@2.xls")
        print("  python FET-plot-combined.py *.xls")
        sys.exit(1)
    
    # Check if first argument is a directory
    first_arg = sys.argv[1]
    if os.path.isdir(first_arg):
        # Get all .xls files from the directory
        file_blocks = glob.glob(os.path.join(first_arg, '*.xls'))
        if not file_blocks:
            print(f"No .xls files found in directory: {first_arg}")
            sys.exit(1)
        print(f"Found {len(file_blocks)} .xls file(s) in {first_arg}")
    else:
        # Treat arguments as file names
        file_blocks = sys.argv[1:]
    
    # Group files by type
    transfer_files = []
    output_files = []
    skipped_files = []
    
    for file_name in file_blocks:
        file_type = detect_measurement_type(file_name)
        if file_type == 'transfer':
            transfer_files.append(file_name)
        elif file_type == 'output':
            output_files.append(file_name)
        else:  # skip
            skipped_files.append(file_name)
    
    print(f"\nDetected {len(transfer_files)} transfer file(s) and {len(output_files)} output file(s)")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} file(s): {', '.join(skipped_files)}")
    if transfer_files:
        print(f"Transfer files: {', '.join(transfer_files)}")
    if output_files:
        print(f"Output files: {', '.join(output_files)}")
    print("-" * 60)
    
    # Calculate total subplots needed
    num_transfer = len(transfer_files)
    num_output = len(output_files)
    total_plots = num_transfer + num_output
    
    if total_plots == 0:
        print("No valid files found.")
        sys.exit(1)
    
    # Determine layout
    cols = math.ceil(math.sqrt(total_plots))
    rows = math.ceil(total_plots / cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 6 * rows))
    
    # Flatten axes for easy indexing
    axes = axes.flatten() if total_plots > 1 else [axes]
    color_name = 'gray_r'
    
    plot_idx = 0
    
    # Process transfer files
    for file_name in transfer_files:
        print(f"Processing transfer: {file_name}")
        data = pd.read_excel(file_name, index_col=False, sheet_name=None)
        aux = file_name.split('.')[0]
        aux0 = file_name.split('-')[1] if '-' in file_name else 'Device'
        plot_transfer_data(data, aux0, color_name, axes[plot_idx])
        plot_idx += 1
    
    # Process output files
    for file_name in output_files:
        print(f"Processing output: {file_name}")
        data = pd.read_excel(file_name, index_col=False, sheet_name=None)
        aux = file_name.split('.')[0]
        aux0 = file_name.split('-')[1] if '-' in file_name else 'Device'
        plot_output_data(data, aux0, color_name, axes[plot_idx])
        plot_idx += 1
    
    # Hide unused subplots
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # save figure

    title_fig = input('Name of the figure: ')
    fig.savefig(title_fig + '.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
