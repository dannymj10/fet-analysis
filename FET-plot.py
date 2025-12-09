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

# Plotting constants
PLOT_LINEWIDTH = 2
PLOT_MARKERSIZE = 4
FONT_SIZE_SMALL = 10
FONT_SIZE_MEDIUM = 11
FONT_SIZE_NORMAL = 12
FONT_SIZE_LARGE = 14
FONT_SIZE_XLARGE = 16
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'p', '8']

# Helper functions
def get_colors_and_markers(num_items, colormap='gray_r'):
    """Get colors and markers for plotting multiple items.
    
    Args:
        num_items: Number of items to plot
        colormap: Matplotlib colormap name
        
    Returns:
        tuple: (colors list, markers list)
    """
    aux0 = plt.get_cmap(colormap)(np.linspace(0.8, 0.9, num_items))
    colors = [mcolors.to_hex(color) for color in aux0]
    return colors, MARKERS

def detect_dual_sweep(v_clean):
    """Detect if voltage data represents a dual sweep (up then down).
    
    Args:
        v_clean: Cleaned voltage array
        
    Returns:
        tuple: (is_dual_sweep, max_idx) where is_dual_sweep is bool and max_idx is the turning point
    """
    max_idx = np.argmax(v_clean)
    is_dual_sweep = max_idx > 0 and max_idx < len(v_clean) - 1
    return is_dual_sweep, max_idx

def calculate_vth_and_ss(v_clean, i_clean):
    """Calculate V_th and SS from transfer curve data using steepest slope method.
    
    Returns:
        tuple: (vth, ss, best_slice) where best_slice contains fit parameters
    """
    sqrt_id = np.sqrt(np.abs(i_clean))
    
    # Linear-region detection using sliding window
    n = len(v_clean)
    min_win = max(5, int(0.1 * n))
    win = max(min_win, int(0.3 * n))
    if win >= n:
        win = n
    
    best_slope = -np.inf
    best_slice = None
    for start in range(0, n - win + 1):
        xw = v_clean[start:start + win]
        yw = sqrt_id[start:start + win]
        if np.allclose(yw, yw[0]):
            continue
        coef = np.polyfit(xw, yw, 1)
        slope = coef[0]
        
        # Calculate R² to ensure reasonable linearity
        ypred = np.polyval(coef, xw)
        ss_res = np.sum((yw - ypred) ** 2)
        ss_tot = np.sum((yw - np.mean(yw)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        
        # Find steepest slope with acceptable linearity (R² > 0.85)
        if slope > best_slope and r2 > 0.85:
            best_slope = slope
            best_slice = (start, start + win, coef)
    
    # Compute V_th
    x_intercept = np.nan
    if best_slice is not None:
        s0, s1, coef = best_slice
        m, b = coef[0], coef[1]
        if abs(m) > 1e-12:
            x_intercept = -b / m
    
    # Calculate SS using positive I and V < V_th
    ss_value = np.nan
    if not np.isnan(x_intercept):
        ss_mask = (i_clean > 0) & (v_clean < x_intercept)
        if ss_mask.sum() > 5:
            v_ss = v_clean[ss_mask]
            i_ss = i_clean[ss_mask]
            log_i_ss = np.log10(i_ss)
            ss_coef = np.polyfit(v_ss, log_i_ss, 1)
            ss_slope = ss_coef[0]
            if abs(ss_slope) > 1e-9:
                ss_value = (1.0 / ss_slope) * 1000  # Convert to mV/decade
    
    return x_intercept, ss_value, best_slice

def calculate_ion_ioff(i_clean, v_clean, vth):
    """Calculate Ion/Ioff ratio.
    
    Ion: Maximum current
    Ioff: Minimum current below V_th
    """
    if np.isnan(vth) or len(i_clean) == 0:
        return np.nan, np.nan, np.nan
    
    i_abs = i_clean #np.abs(i_clean)
    ion = np.max(i_abs)
    
    # Find Ioff as minimum current below V_th
    mask_below_vth = v_clean < vth
    if mask_below_vth.sum() > 0:
        ioff = np.min(i_abs[mask_below_vth])
    else:
        ioff = np.min(i_abs)
    
    ratio = np.abs(ion / ioff) if ioff > 0 else np.nan
    return ion, ioff, ratio

def calculate_gm_max(v_clean, i_clean):
    """Calculate maximum transconductance (gm = dI/dV) and corresponding Vgs.
    
    Returns:
        tuple: (gm_max, v_at_gm_max)
    """
    if len(v_clean) < 2:
        return np.nan, np.nan
    
    gm = np.gradient(i_clean, v_clean)
    gm_abs = np.abs(gm)
    max_idx = np.argmax(gm_abs)
    gm_max = gm_abs[max_idx]
    v_at_gm_max = v_clean[max_idx]
    return gm_max, v_at_gm_max

def create_wl_annotation(ax, name, vth_ss_values=None, sheets=None):
    """Create W/L annotation text box, optionally with V_th and SS values."""
    w_match = re.search(r'W(\d+)', name, re.IGNORECASE)
    l_match = re.search(r'L(\d+)', name, re.IGNORECASE)
    if not (w_match and l_match):
        return
    
    w_val = w_match.group(1)
    l_val = l_match.group(1)
    
    text_lines = [f'W/L = {w_val}/{l_val}']
    
    # Add V_th and SS if provided (for saturation curves)
    if vth_ss_values and sheets:
        for sheet in sheets:
            if sheet in vth_ss_values:
                vth = vth_ss_values[sheet]['vth']
                ss = vth_ss_values[sheet]['ss']
                if not np.isnan(vth):
                    text_lines.append(f'V$_{{th}}$ = {vth:.2f} V')
                    if not np.isnan(ss):
                        text_lines.append(f'SS = {ss:.1f} mV/dec')
    
    combined_text = '\n'.join(text_lines)
    
    from matplotlib.offsetbox import AnchoredText
    text_box = AnchoredText(combined_text, loc='lower right', 
                           prop=dict(size=FONT_SIZE_MEDIUM), frameon=True)
    text_box.patch.set_boxstyle("round,pad=0.1")
    text_box.patch.set_facecolor('white')
    text_box.patch.set_alpha(0.95)
    ax.add_artist(text_box)

# Main plotting functions
def plot_transfer_data(file, name, cc, ax, filename=""):
    """Plot transfer characteristics (Id vs Vgs) with sqrt(Id) and V_th/SS extraction."""
    sheets = [sheet for sheet in file.keys() if sheet not in ['Calc', 'Settings']]

    # Get colors and markers
    colors, markers = get_colors_and_markers(len(sheets), cc)

    # Check if this is a linear file
    is_linear = 'linear' in filename.lower()

    # Plot Id vs Vgs for all sheets
    for i, (sheet, color) in enumerate(zip(sheets, colors)):
        df = file[sheet]
        ax.set_yscale('log')
        marker = markers[i % len(markers)]
        
        # Get DrainV value for label
        drain_v_label = None
        if 'DrainV' in df.columns:
            drain_vals = pd.to_numeric(df['DrainV'], errors='coerce').dropna()
            if len(drain_vals) > 0:
                drain_v_label = drain_vals.iloc[0]
        
        label = f'Vds = {drain_v_label:.1f} V' if drain_v_label is not None else sheet

        # Plot Id vs Vgs (transfer characteristics) - only forward sweep if dual sweep
        v = pd.to_numeric(df['GateV'], errors='coerce').values
        i_vals = pd.to_numeric(df['DrainI'], errors='coerce').values
        mask = np.isfinite(v) & np.isfinite(i_vals)
        v_clean = v[mask]
        i_clean = i_vals[mask]
        
        # Detect dual sweep
        is_dual_sweep, max_idx = detect_dual_sweep(v_clean)
        
        if is_dual_sweep:
            # Plot only forward sweep
            v_plot = v_clean[:max_idx + 1]
            i_plot = np.abs(i_clean[:max_idx + 1])
        else:
            # Plot full data for single sweep
            v_plot = v_clean
            i_plot = np.abs(i_clean)
        
        ax.plot(v_plot, i_plot, color=color, marker=marker, label=label)

 
    # Store V_th and SS values for plotting
    vth_ss_values = {}
    
    # Calculate V_th for all sheets
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
        
        # Detect dual sweep: check if voltage goes up then down
        is_dual_sweep, max_idx = detect_dual_sweep(v_clean)
        
        if is_dual_sweep:
            # Split into forward and reverse sweeps
            v_forward = v_clean[:max_idx + 1]
            i_forward = i_clean[:max_idx + 1]
            v_reverse = v_clean[max_idx:]
            i_reverse = i_clean[max_idx:]
            
            # Calculate V_th for forward sweep
            vth_fwd, ss_fwd, best_slice_fwd = calculate_vth_and_ss(v_forward, i_forward)
            
            # Calculate V_th for reverse sweep
            vth_rev, ss_rev, best_slice_rev = calculate_vth_and_ss(v_reverse, i_reverse)
            
            # Use forward sweep as primary, store both
            x_intercept, ss_value, best_slice = vth_fwd, ss_fwd, best_slice_fwd
            sqrt_id = np.sqrt(np.abs(i_clean))
            
            # Calculate Ion/Ioff ratio using full data
            ion, ioff, ion_ioff_ratio = calculate_ion_ioff(i_clean, v_clean, x_intercept)
            
            # Calculate gm_max and V@Gm_max using full data
            gm_max, v_at_gm_max = calculate_gm_max(v_clean, i_clean)
            
            # Store values including both forward and reverse V_th
            vth_ss_values[sheet] = {
                'vth': x_intercept,
                'vth_fwd': vth_fwd,
                'vth_rev': vth_rev,
                'ss': ss_value,
                'ss_fwd': ss_fwd,
                'ss_rev': ss_rev,
                'color': color,
                'best_slice': best_slice,
                'v_clean': v_clean,
                'sqrt_id': sqrt_id,
                'ion': ion,
                'ioff': ioff,
                'ion_ioff_ratio': ion_ioff_ratio,
                'gm_max': gm_max,
                'v_at_gm_max': v_at_gm_max,
                'is_dual_sweep': True
            }
        else:
            # Single sweep - original calculation
            x_intercept, ss_value, best_slice = calculate_vth_and_ss(v_clean, i_clean)
            sqrt_id = np.sqrt(np.abs(i_clean))
            
            # Calculate Ion/Ioff ratio
            ion, ioff, ion_ioff_ratio = calculate_ion_ioff(i_clean, v_clean, x_intercept)
            
            # Calculate gm_max and V@Gm_max
            gm_max, v_at_gm_max = calculate_gm_max(v_clean, i_clean)
            
            # Store values for plotting
            vth_ss_values[sheet] = {
                'vth': x_intercept, 
                'ss': ss_value, 
                'color': color,
                'best_slice': best_slice,
                'v_clean': v_clean,
                'sqrt_id': sqrt_id,
                'ion': ion,
                'ioff': ioff,
                'ion_ioff_ratio': ion_ioff_ratio,
                'gm_max': gm_max,
                'v_at_gm_max': v_at_gm_max,
                'is_dual_sweep': False
            }

    # Only plot gm for linear files
    if not is_linear:
        # For saturation files, plot sqrt(DrainI) on secondary y-axis
        ax2 = ax.twinx()
        
        for sheet, color in zip(sheets, colors):
            # Reuse data from vth_ss_values instead of re-reading
            if sheet not in vth_ss_values:
                continue
            
            v_clean = vth_ss_values[sheet]['v_clean']
            
            # Detect dual sweep and use only forward sweep for plotting
            is_dual_sweep, max_idx = detect_dual_sweep(v_clean)
            
            if is_dual_sweep:
                v_plot = v_clean[:max_idx + 1]
                # Recalculate i_plot from original data since we need abs
                df = file[sheet]
                i_vals = pd.to_numeric(df['DrainI'], errors='coerce').values
                mask = np.isfinite(v_clean) & np.isfinite(i_vals)[:len(v_clean)]
                i_clean = i_vals[mask]
                i_plot = i_clean[:max_idx + 1]
            else:
                v_plot = v_clean
                # Get i_clean from original data
                df = file[sheet]
                i_vals = pd.to_numeric(df['DrainI'], errors='coerce').values
                mask = np.isfinite(v_clean) & np.isfinite(i_vals)[:len(v_clean)]
                i_plot = i_vals[mask]
            
            sqrt_id = np.sqrt(np.abs(i_plot))
            
            # Plot sqrt(Id) on secondary axis
            ax2.plot(v_plot, sqrt_id, color=color, linestyle='-', linewidth=PLOT_LINEWIDTH, alpha=0.75)
            
            # Plot the extrapolation line if V_th was calculated
            best_slice = vth_ss_values[sheet]['best_slice']
            vth = vth_ss_values[sheet]['vth']
            
            if best_slice is not None and not np.isnan(vth):
                s0, s1, coef = best_slice
                m, b = coef[0], coef[1]
                
                if abs(m) > 1e-12:
                    # Create extrapolation line from V_th to the fitted region
                    v_start = min(v_plot[s0], vth)
                    v_end = max(v_plot[min(s1 - 1, len(v_plot) - 1)], v_plot[-1])
                    v_fit = np.linspace(v_start, v_end, 100)
                    y_fit = m * v_fit + b
                    
                    # Plot the fitted/extrapolation line
                    ax2.plot(v_fit, y_fit, color=color, linestyle='--', linewidth=PLOT_LINEWIDTH, alpha=0.7)
                    
        ax2.set_ylabel(r'$\sqrt{I_{ds}}$ (A$^{1/2}$)', fontsize=FONT_SIZE_XLARGE)
        ax2.tick_params(axis='y', labelsize=FONT_SIZE_LARGE)
        
        ax.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
        ax.set_xlabel('Vgs (V)', fontsize=FONT_SIZE_XLARGE)
        ax.set_ylabel('Ids (A)', fontsize=FONT_SIZE_XLARGE)
        ax.legend(loc='upper left', fontsize=FONT_SIZE_LARGE, framealpha=1)
        
        #print(vth_ss_values) # For debugging
        create_wl_annotation(ax, name, vth_ss_values, sheets)
        return vth_ss_values

    # For linear files, continue with gm plotting
    # Plot gm (transconductance) on a secondary y-axis ONLY for linear files
    ax3 = ax.twinx()
    
    for sheet, color in zip(sheets, colors):
        df = file[sheet]
        if 'GateV' not in df.columns or 'DrainI' not in df.columns:
            continue
        v = pd.to_numeric(df['GateV'], errors='coerce').values
        i_vals = pd.to_numeric(df['DrainI'], errors='coerce').values
        mask = np.isfinite(v) & np.isfinite(i_vals)
        if mask.sum() < 2:
            continue
        v_clean = v[mask]
        i_clean = i_vals[mask]
        
        # Calculate gm = dI/dV
        gm = np.gradient(i_clean, v_clean)
        
        # Plot gm
        ax3.plot(v_clean, gm, color=color, linestyle='-', linewidth=PLOT_LINEWIDTH, alpha=0.75)
    
    ax3.set_ylabel(r'Gm (S)', fontsize=FONT_SIZE_XLARGE)
    ax3.tick_params(axis='y', labelsize=FONT_SIZE_LARGE)
    # Create W/L annotation using helper function
    create_wl_annotation(ax, name)

def plot_output_data(file, name, cc, ax):
    """Plot output characteristics (Id vs Vds) for all gate voltages."""
    sheets = [sheet for sheet in file.keys() if sheet not in ['Calc', 'Settings']]
    
    # Get colors and markers
    colors, markers = get_colors_and_markers(len(sheets), cc)
    
    for sheet in sheets:
        df = file[sheet]
        
        # Find all DrainV(n), DrainI(n), and GateV(n) column sets
        drain_v_cols = [col for col in df.columns if col.startswith('DrainV')]
        drain_i_cols = [col for col in df.columns if col.startswith('DrainI')]
        gate_v_cols = [col for col in df.columns if col.startswith('GateV')]
        
        # Extract sweep numbers and match pairs
        sweep_data = []
        for v_col in drain_v_cols:
            # Extract number from column name (e.g., 'DrainV(1)' -> '1')
            v_match = re.search(r'DrainV\((\d+)\)', v_col)
            if v_match:
                sweep_num = v_match.group(1)
                i_col = f'DrainI({sweep_num})'
                g_col = f'GateV({sweep_num})'
                if i_col in drain_i_cols:
                    # Get the gate voltage value (first non-NaN value from GateV column)
                    gate_voltage = None
                    if g_col in gate_v_cols:
                        gate_vals = pd.to_numeric(df[g_col], errors='coerce').dropna()
                        if len(gate_vals) > 0:
                            gate_voltage = gate_vals.iloc[0]
                    sweep_data.append((v_col, i_col, sweep_num, gate_voltage))
        
        # If no numbered columns found, fall back to 'DrainV' and 'DrainI'
        if not sweep_data and 'DrainV' in df.columns and 'DrainI' in df.columns:
            gate_voltage = None
            if 'GateV' in df.columns:
                gate_vals = pd.to_numeric(df['GateV'], errors='coerce').dropna()
                if len(gate_vals) > 0:
                    gate_voltage = gate_vals.iloc[0]
            sweep_data = [('DrainV', 'DrainI', '1', gate_voltage)]
        
        # Plot each sweep 
        color = 'black'
        for i, (v_col, i_col, sweep_num, gate_voltage) in enumerate(sweep_data):
            marker = markers[i % len(markers)]
            if gate_voltage is not None:
                label = f'Vgs = {gate_voltage:.1f} V'
            else:
                label = f'Sweep {sweep_num}'
            ax.plot(df[v_col], df[i_col], color=color, marker=marker, label=label, 
                   markersize=PLOT_MARKERSIZE, linestyle='-', linewidth=PLOT_LINEWIDTH)

    ax.tick_params(axis='both', labelsize=FONT_SIZE_LARGE)
    ax.set_xlabel('Vds (V)', fontsize=FONT_SIZE_XLARGE)
    ax.set_ylabel('Ids (A)', fontsize=FONT_SIZE_XLARGE)
    ax.legend(loc='best', fontsize=FONT_SIZE_SMALL, framealpha=1)
    
    # Extract W and L values from name parameter (e.g., "W100um/L40um")
    w_match = re.search(r'W(\d+)', name, re.IGNORECASE)
    l_match = re.search(r'L(\d+)', name, re.IGNORECASE)
    if w_match and l_match:
        w_val = w_match.group(1)
        l_val = l_match.group(1)
        from matplotlib.offsetbox import AnchoredText
        text_box = AnchoredText(f'W/L = {w_val}/{l_val}', loc='lower right', 
                               prop=dict(size=FONT_SIZE_NORMAL), frameon=True)
        text_box.patch.set_boxstyle("round,pad=0.1")
        text_box.patch.set_facecolor('white')
        text_box.patch.set_alpha(1)
        ax.add_artist(text_box)

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
    
    # Global metrics collection for single CSV output
    global_metrics = []
    
    # Check if first argument is a directory
    first_arg = sys.argv[1]
    if os.path.isdir(first_arg):
        # Check for subdirectories
        subdirs = [d for d in os.listdir(first_arg) if os.path.isdir(os.path.join(first_arg, d))]
        
        if subdirs:
            # Process each subdirectory separately
            for subdir in subdirs:
                subdir_path = os.path.join(first_arg, subdir)
                subdir_files = glob.glob(os.path.join(subdir_path, '*.xls'))
                
                if subdir_files:
                    print(f"\n{'='*60}")
                    print(f"Processing directory: {subdir}")
                    print(f"{'='*60}")
                    # Pass both parent and subdir: e.g., "W100um/L40um"
                    combined_label = f"{os.path.basename(first_arg)}/{subdir}"
                    metrics = process_files(subdir_files, combined_label)
                    if metrics:
                        global_metrics.extend(metrics)
            
            # Export single CSV for all data
            if global_metrics:
                metrics_df = pd.DataFrame(global_metrics)
                csv_filename = f"{os.path.basename(first_arg)}_all_metrics.csv"
                metrics_df.to_csv(csv_filename, index=False, float_format='%.4e')
                print(f"\n{'='*60}")
                print(f"All metrics exported to: {csv_filename}")
                print(f"Total measurements: {len(metrics_df)}")
                print(f"{'='*60}")
                print("\nSummary:")
                print(metrics_df.to_string(index=False))
            return
        else:
            # No subdirectories, process files in the directory
            file_blocks = glob.glob(os.path.join(first_arg, '*.xls'))
            if not file_blocks:
                print(f"No .xls files found in directory: {first_arg}")
                sys.exit(1)
            print(f"Found {len(file_blocks)} .xls file(s) in {first_arg}")
            metrics = process_files(file_blocks, first_arg)
            if metrics:
                global_metrics.extend(metrics)
    else:
        # Treat arguments as file names
        file_blocks = sys.argv[1:]
        metrics = process_files(file_blocks, "Combined")
        if metrics:
            global_metrics.extend(metrics)
    
    # Export single CSV for standalone processing
    if global_metrics and not os.path.isdir(first_arg):
        metrics_df = pd.DataFrame(global_metrics)
        csv_filename = "FET_all_metrics.csv"
        metrics_df.to_csv(csv_filename, index=False, float_format='%.4e')
        print(f"\n{'='*60}")
        print(f"All metrics exported to: {csv_filename}")
        print(f"Total measurements: {len(metrics_df)}")
        print(f"{'='*60}")

def process_files(file_blocks, label):
    """Process a list of files and create plots."""
    
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
    
    # Collect all metrics for CSV export
    all_metrics = []
    
    # Extract W and L from label
    w_match = re.search(r'W(\d+)', label, re.IGNORECASE)
    l_match = re.search(r'L(\d+)', label, re.IGNORECASE)
    w_val = w_match.group(1) if w_match else 'N/A'
    l_val = l_match.group(1) if l_match else 'N/A'
    
    # Dictionary to accumulate metrics by Vds value (sheet name)
    metrics_by_vds = {}
    
    # Process transfer files
    for file_name in transfer_files:
        print(f"Processing transfer: {file_name}")
        data = pd.read_excel(file_name, index_col=False, sheet_name=None)
        metrics = plot_transfer_data(data, label, color_name, axes[plot_idx], file_name)
        
        # Add metrics to collection
        if metrics:
            for sheet, values in metrics.items():
                if sheet not in metrics_by_vds:
                    # Check if this is dual sweep data
                    if values.get('is_dual_sweep', False):
                        metrics_by_vds[sheet] = {
                            'W (um)': w_val,
                            'L (um)': l_val,
                            'V_th_fwd (V)': values.get('vth_fwd', np.nan),
                            'V_th_rev (V)': values.get('vth_rev', np.nan),
                            'SS_fwd (mV/dec)': values.get('ss_fwd', np.nan),
                            'SS_rev (mV/dec)': values.get('ss_rev', np.nan),
                            'Ion/Ioff': values['ion_ioff_ratio'],
                            'V@Gm_max (V)': values['v_at_gm_max'],
                            'Gm_max (S)': values['gm_max']
                        }
                    else:
                        metrics_by_vds[sheet] = {
                            'W (um)': w_val,
                            'L (um)': l_val,
                            'V_th (V)': values['vth'],
                            'SS (mV/dec)': values['ss'],
                            'Ion/Ioff': values['ion_ioff_ratio'],
                            'V@Gm_max (V)': values['v_at_gm_max'],
                            'Gm_max (S)': values['gm_max']
                        }
                else:
                    # Update existing entry with values from current file
                    if values.get('is_dual_sweep', False):
                        # Update with dual sweep values
                        if 'vth_fwd' in values and not np.isnan(values['vth_fwd']):
                            metrics_by_vds[sheet]['V_th_fwd (V)'] = values['vth_fwd']
                        if 'vth_rev' in values and not np.isnan(values['vth_rev']):
                            metrics_by_vds[sheet]['V_th_rev (V)'] = values['vth_rev']
                        if 'ss_fwd' in values and not np.isnan(values['ss_fwd']):
                            metrics_by_vds[sheet]['SS_fwd (mV/dec)'] = values['ss_fwd']
                        if 'ss_rev' in values and not np.isnan(values['ss_rev']):
                            metrics_by_vds[sheet]['SS_rev (mV/dec)'] = values['ss_rev']
                    else:
                        # Update with single sweep values
                        if not np.isnan(values['vth']):
                            metrics_by_vds[sheet]['V_th (V)'] = values['vth']
                        if not np.isnan(values['ss']):
                            metrics_by_vds[sheet]['SS (mV/dec)'] = values['ss']
                    
                    # Update common metrics
                    if not np.isnan(values['ion_ioff_ratio']):
                        metrics_by_vds[sheet]['Ion/Ioff'] = values['ion_ioff_ratio']
                    if not np.isnan(values['v_at_gm_max']):
                        metrics_by_vds[sheet]['V@Gm_max (V)'] = values['v_at_gm_max']
                    if not np.isnan(values['gm_max']):
                        metrics_by_vds[sheet]['Gm_max (S)'] = values['gm_max']
        
        plot_idx += 1
    
    # Convert metrics dictionary to list
    for sheet_metrics in metrics_by_vds.values():
        all_metrics.append(sheet_metrics)
    
    # Process output files
    for file_name in output_files:
        print(f"Processing output: {file_name}")
        data = pd.read_excel(file_name, index_col=False, sheet_name=None)
        plot_output_data(data, label, color_name, axes[plot_idx])
        plot_idx += 1
    
    # Hide unused subplots
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save figure
    w_match_fig = re.search(r'(W\d+um)', label, re.IGNORECASE)
    l_match_fig = re.search(r'(L\d+um)', label, re.IGNORECASE)
    if w_match_fig and l_match_fig:
        fig_name = f"IV_{w_match_fig.group(1)}_{l_match_fig.group(1)}"
    else:
        fig_name = f"IV_{label.replace('/', '_')}"
    
    fig.savefig(f'{fig_name}.png', bbox_inches='tight', dpi=300)
    print(f"Figure saved as: {fig_name}.png")
    
    plt.show()
    
    # Return metrics for global collection
    return all_metrics

if __name__ == "__main__":
    main()