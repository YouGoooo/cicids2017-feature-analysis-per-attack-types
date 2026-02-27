import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters (keep same window sizing as original script)
INPUT_CSV = './data/cicids2017_filtered.csv'
OUT_DIR = './results'
WINDOW_SIZE = 50000
STEP = WINDOW_SIZE          # keep same behaviour (set < WINDOW_SIZE for overlap)
LABEL_COL = 'Attack Type'

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
print("Dataset shape:", df.shape)

if LABEL_COL not in df.columns:
    raise RuntimeError(f"Expected label column '{LABEL_COL}' not found.")

# Ensure deterministic row slicing by resetting the index
df = df.reset_index(drop=True)
df = df.dropna(subset=[LABEL_COL])
y_all = df[LABEL_COL].astype(str).copy()
n_rows = len(df)

# build windows (same logic)
windows = []
start = 0
while start < n_rows:
    end = min(start + WINDOW_SIZE, n_rows)
    windows.append((start, end))
    if end == n_rows:
        break
    start += STEP

print(f"Computed {len(windows)} windows (window_size={WINDOW_SIZE}, step={STEP})")

# determine all attack types across dataset so every window has same index
attack_types = sorted(y_all.unique().tolist())

# prepare DataFrames: counts and percentages (per-window)
cols = [f"window_{i}" for i in range(len(windows))]
counts_df = pd.DataFrame(0, index=attack_types, columns=cols, dtype=int)
percent_df = pd.DataFrame(0.0, index=attack_types, columns=cols, dtype=float)

t0 = time.perf_counter()

print("\nProcessing Windows and Displaying Percentages:")
print("="*50)

for i, (s, e) in enumerate(windows):
    y_win = y_all.iloc[s:e]
    if y_win.empty:
        continue
    
    # get counts for all known attack types (fill missing with 0)
    counts = y_win.value_counts().reindex(attack_types, fill_value=0).astype(int)
    counts_df[f"window_{i}"] = counts
    
    total = counts.sum()
    if total > 0:
        # Calculate percentages
        percents = (counts / float(total)) * 100.0
        percent_df[f"window_{i}"] = percents

        # --- MODIFICATION START: Print percentages for this window ---
        print(f"\n--- Window {i} (Rows {s} to {e}) ---")
        # Filter to show only attacks present in this window (> 0%)
        # and sort them descending for better readability
        active_attacks = percents[percents > 0].sort_values(ascending=False)
        
        for attack_name, val in active_attacks.items():
            print(f"  {attack_name:<25}: {val:.2f}%")
        # --- MODIFICATION END ---

print("="*50)
print(f"Computed counts and percentages in {time.perf_counter()-t0:.2f}s")

# Save CSV outputs
csv_counts = os.path.join(OUT_DIR, 'attacks_count_per_window.csv')
csv_percent = os.path.join(OUT_DIR, 'attacks_percent_per_window.csv')
counts_df.to_csv(csv_counts)
percent_df.to_csv(csv_percent)
print("Saved counts csv:", csv_counts)
print("Saved percentages csv:", csv_percent)

# Heatmap: counts (log-scaled for visibility if counts vary a lot)
fig = plt.figure(figsize=(14, max(6, len(attack_types) * 0.3)))
vals = counts_df.values.astype(float)
# add small epsilon to avoid log(0)
eps = 1e-6
im = plt.imshow(np.log1p(vals + eps), aspect='auto', cmap='magma', interpolation='nearest')
plt.colorbar(im, label='log(1 + count)')
plt.yticks(range(len(attack_types)), attack_types, fontsize=8)
plt.xticks(range(len(cols)), cols, rotation=45, ha='right', fontsize=8)
plt.title('Attack counts per window (log scale)')
plt.tight_layout()
png_heat = os.path.join(OUT_DIR, 'attacks_count_per_window_heatmap.png')
plt.savefig(png_heat, dpi=150)
plt.close(fig)
print("Saved heatmap:", png_heat)

# Optional: small summary print
summary = counts_df.sum(axis=1).sort_values(ascending=False)
print("\nTotal counts per attack type (dataset-wide):")
print(summary)