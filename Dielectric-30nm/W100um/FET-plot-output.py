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
        marker = markers[i % len(markers)]
        ax.plot(df['V'], df['I'], color=color, marker=marker, label=sheet)

    #ymin, ymax = ax.get_ylim()
    #ax.set_ylim([ymin/1000, ymax * 2])

    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Drain Voltage (V)', fontsize=16)
    ax.set_ylabel('Drain Current (A)', fontsize=16)
    
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
#title_fig = input('Name of the figure: ')
#fig.savefig(title_fig+'.png', bbox_inches='tight', dpi=300)
plt.show()