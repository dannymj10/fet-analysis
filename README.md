# FET Analysis Tool

A Python script for analyzing and visualizing Field-Effect Transistor (FET) measurements, specifically transfer and output characteristics.

## Features

- **Automatic Mode Detection**: Automatically detects measurement type (transfer or output) from filenames
- **Transfer Characteristics Analysis**:
  - Plots Ids vs Vgs with logarithmic scale
  - Calculates threshold voltage (V_th) using linear extrapolation method
  - Computes subthreshold swing (SS) in mV/decade
  - Calculates Ion/Ioff ratio
  - Determines maximum transconductance (Gm_max) and corresponding Vgs
  - Plots √Ids for saturation measurements or Gm for linear measurements
  - Supports dual-sweep measurements (forward and reverse)
  
- **Output Characteristics Analysis**:
  - Plots Ids vs Vds for multiple gate voltages
  - Automatically extracts gate voltage values from data

- **Batch Processing**:
  - Process entire directories or individual files
  - Support for nested directory structures
  - Automatic W/L ratio extraction and display

- **Data Export**:
  - Exports comprehensive metrics to CSV files
  - Includes V_th, SS, Ion/Ioff, Gm_max, and V@Gm_max for all measurements

## Requirements

```
pandas
matplotlib
numpy
```

Install dependencies:
```bash
pip install pandas matplotlib numpy
```

## Usage

### Process a directory with subdirectories:
```bash
python FET-plot.py W100um
```
This will process all `.xls` files in subdirectories and generate a single CSV with all metrics.

### Process specific files:
```bash
python FET-plot.py vgs-id-linear.xls vgs-id-sat.xls vds-id.xls
```

### Process all .xls files in current directory:
```bash
python FET-plot.py *.xls
```

## File Naming Convention

The script automatically detects measurement types from filenames:
- **Transfer characteristics**: Files containing 'vgs' and 'id' (e.g., `vgs-id-linear.xls`)
- **Output characteristics**: Files containing 'vds' and 'id' (e.g., `vds-id.xls`)
- **Skip**: Files containing 'after' are automatically skipped

## Output

- **PNG figures**: High-resolution (300 DPI) plots with automatically detected W/L ratios
- **CSV files**: Comprehensive metrics including:
  - W/L dimensions
  - V_th (threshold voltage)
  - SS (subthreshold swing)
  - Ion/Ioff ratio
  - Gm_max (maximum transconductance)
  - V@Gm_max (gate voltage at maximum transconductance)
  - For dual-sweep: separate V_th and SS for forward and reverse sweeps

## Data Format

Input files should be Excel (.xls) format with columns:
- **Transfer**: `GateV`, `DrainI`, `DrainV`
- **Output**: `DrainV(n)`, `DrainI(n)`, `GateV(n)` where n is the sweep number

## Example

```bash
# Process a directory structure like:
# W100um/
#   L40um/
#     vgs-id-linear.xls
#     vgs-id-sat.xls
#     vds-id.xls
#   L60um/
#     ...

python FET-plot.py W100um
```

This will generate:
- Individual plots for each L value (e.g., `IV_W100um_L40um.png`)
- A single CSV file with all metrics (`W100um_all_metrics.csv`)

## License

MIT License

## Author

Developed for MSE6348-F25 TFT measurements analysis
