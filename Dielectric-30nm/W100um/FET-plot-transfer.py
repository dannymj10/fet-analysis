import sys
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import re
import numpy as np
import matplotlib.colors as mcolors

def plot_data(file, name, cc, ax):
    #name: size of the TE
    #cc: color. It's an input variable
    #name1: file details eg. anneal, Temp, ...
    """Plots data subplots (ax)."""
    sheets = [sheet for sheet in file.keys() if sheet not in ['Calc', 'Settings']]
    p = np.pi

    # Define colors, they are shades of a given color
    aux0 = plt.get_cmap(cc)(np.linspace(0.8, 0.9, len(sheets)))
    colors = [mcolors.to_hex(color) for color in aux0]
    ####################################

    num0 = re.search(r'\d+(\.\d+)?', name) # Match number in the 'name' variable
    #num1 = re.search(r'\d+(\.\d+)?', name1)
    if num0:
        TE = float(num0.group())
        #VIA = float(num1.group())

    # Define a list of markers for distinction
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'p', '8']
    for i, (sheet, color) in enumerate(zip(sheets, colors)):
        df = file[sheet]
        ax.set_yscale('log')
        marker = markers[i % len(markers)]
        #ax.plot(df['V'], abs(df['I']), color=color, marker=marker)
        ax.plot(df['V'], df['I'], color=color, marker=marker, label=sheet)

    # Plot derivative dI/dV on a secondary y-axis (robust to NaNs)
    ax2 = ax.twinx()
    for sheet, color in zip(sheets, colors):
        df = file[sheet]
        if 'V' not in df.columns or 'I' not in df.columns:
            continue
        v = pd.to_numeric(df['V'], errors='coerce').values
        i_vals = pd.to_numeric(df['I'], errors='coerce').values
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
                    label_text = f'V$_{{th}}$={x_intercept:.2f} V'#, SS={abs(ss_value):.0f} mV/dec'
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

    #ymin, ymax = ax.get_ylim()
    #ax.set_ylim([ymin/1000, ymax * 2])

    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Vgs (V)', fontsize=16)
    ax.set_ylabel('Ids (A)', fontsize=16)
    
    #ax.grid()
    #ax.text(0.95, 0.05,  f'$\\mathrm{{\\Phi}} = \\mathrm{{ {TE}~\\mu m }}$',
    #          transform=ax.transAxes,
    #          va='bottom', ha='right', color='black', fontsize=12,
    #          bbox = dict(facecolor = 'white', alpha = 1))
    #ax.set_title(name1)
    ax.legend(loc='upper left', fontsize=14, framealpha=1)

# Program execution
file_blocks = sys.argv[1:]  # Get list of file names from command line
num_files = len(file_blocks)

# Determine number of rows and columns for the grid
cols = math.ceil(math.sqrt(num_files))  # number of columns
#cols = num_files #for 1 row
rows = math.ceil(num_files / cols) #required rows
#rows = 1 #for 1 row
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 6 * rows))
#fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 4)) # for 1 row

# Flatten axes for easy indexing
axes = axes.flatten() if num_files > 1 else [axes]
color_name = 'gray_r' #input('write the color you want:')
##'gray' / 'gray_r' (alias for Greys)
##'gist_gray' / 'gist_gray_r' (grayscale, includes black)

#voltage = float(input('Voltage at which you want to extract the data (e.g., 0.0): '))
#txt_file = f'Current_extracted_V{voltage}.txt'


for i, file_name in enumerate(file_blocks):
    data = pd.read_excel(file_name, index_col=False, sheet_name=None)
    aux = file_name.split('.')[0] #name of the file
    aux0 = file_name.split('-')[1] #size of the TE
    #aux1 = file_name.split('-')[2]
    #data_extraction(data, txt_file, aux0, voltage)
    plot_data(data, aux0, color_name, axes[i])#,aux1)

# Hide unused subplots (if num_files is not a perfect square)
for j in range(num_files, rows*cols):
    fig.delaxes(axes[j])

plt.tight_layout()
title_fig = input('Name of the figure: ')
fig.savefig(title_fig+'.png', bbox_inches='tight', dpi=300)
plt.show()