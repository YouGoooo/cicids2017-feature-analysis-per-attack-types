"""Comprehensive Feature Importance and Attack Consensus Analysis.

This script computes feature importance scores across sliding windows of network traffic
data and generates consensus signatures for different attack types. It calculates
importance using four methods (Random Forest, Permutation, ANOVA, Mutual Information),
aggregates the results, and computes a Jaccard similarity matrix between attack
signatures based on the top-K most important features.

Generated PNG files (saved in results/png/):
- heatmap_rf.png: Heatmap of Random Forest importance scores per window.
- heatmap_permutation.png: Heatmap of Permutation importance scores per window.
- heatmap_anova.png: Heatmap of ANOVA F-value scores per window.
- heatmap_mutual_info.png: Heatmap of Mutual Information scores per window.
- global_comparison_4_methods.png: Horizontal bar charts comparing overall feature importance across all methods.
- methods_correlation_matrix.png: Spearman correlation matrix comparing the four scoring methods.
- consensus_attack_similarity_jaccard.png: Heatmap of the Jaccard similarity index between attack signatures.

Generated CSV files (saved in results/csv/):
- scores_randomforest.csv: Raw feature importance scores from Random Forest.
- scores_permutation.csv: Raw feature importance scores from Permutation testing.
- scores_anova.csv: Raw F-value scores from ANOVA.
- scores_mutual_info.csv: Raw Mutual Information scores.
- feature_importances_aggregated_all.csv: Aggregated mean scores for all methods.
- jaccard_matrix.csv: Raw Jaccard similarity matrix values between attack types.

Estimated time to run: ~10-15 minutes on a standard machine.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_classif, mutual_info_classif
from itertools import groupby

# --- WARNING HANDLING ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection")

# --- PARAMETERS ---
INPUT_CSV = './data/cicids2017_filtered.csv'
OUT_DIR = './results'
CSV_DIR = os.path.join(OUT_DIR, 'csv')
PNG_DIR = os.path.join(OUT_DIR, 'png')
WINDOW_SIZE = 50000
STEP = WINDOW_SIZE
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 5
TOP_K_FEATURES = 20  # Number of top features to define an attack signature
BENIGN_LABEL = "BENIGN"

SHORT_LABEL_MAP = {
    "Web Attack - Brute Force": "Brute Force",
    "Web Attack - XSS": "XSS",
    "Web Attack - Sql Injection": "SQL Inj",
    "DoS GoldenEye": "GoldenEye",
    "DoS Hulk": "Hulk",
    "DoS Slowhttptest": "SlowHttp",
    "DoS slowloris": "Slowloris",
    "FTP-Patator": "FTP-Pat",
    "SSH-Patator": "SSH-Pat"
}

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --- UTILITY FUNCTIONS ---
def normalize_series(s):
    """Scale a pandas Series to [0,1]; return zeros if constant."""
    if s.max() > s.min():
        return (s - s.min()) / (s.max() - s.min())
    return s * 0.0

def normalize_array(arr):
    """Scale a numpy array to [0,1] while handling NaNs."""
    arr = np.nan_to_num(arr)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)

def build_windows(n_rows, window_size, step):
    """Return list of (start,end) tuples for sliding windows."""
    windows = []
    start = 0
    while start < n_rows:
        end = min(start + window_size, n_rows)
        windows.append((start, end))
        if end == n_rows:
            break
        start += step
    return windows

def get_contiguous_regions(labels_list):
    """Group consecutive identical labels into (start, end, label) tuples."""
    regions = []
    if not labels_list:
        return regions
    current_idx = 0
    for label, group in groupby(labels_list):
        group_len = len(list(group))
        regions.append((current_idx, current_idx + group_len - 1, label))
        current_idx += group_len
    return regions

def plot_heatmap_with_annotations(df_scores, region_map, title_str, filename):
    """Save a heatmap of feature scores with vertical attack annotations."""
    if df_scores is None or df_scores.empty:
        return

    features_list = df_scores.index.tolist()
    fig_h = max(10, len(features_list) * 0.15)
    fig_w = max(14, len(df_scores.columns) * 0.3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    v_max = df_scores.values.max()
    v_max = v_max if v_max > 0 else 1.0

    im = ax.imshow(df_scores.values, aspect='auto', cmap='viridis',
                   interpolation='nearest', vmin=0, vmax=v_max)

    cbar = plt.colorbar(im, pad=0.01)
    cbar.set_label('Normalized Importance (0-1)', size=12)

    step_y = 2 if len(features_list) > 50 else 1
    ax.set_yticks(range(0, len(features_list), step_y))
    ax.set_yticklabels([features_list[j] for j in range(0, len(features_list), step_y)], fontsize=9)

    ax.set_xticks(range(len(df_scores.columns)))
    ax.set_xticklabels(df_scores.columns, rotation=45, ha='right', fontsize=9)
    ax.set_title(title_str, pad=25, fontsize=16)

    levels = [-1.5, -4.0]
    for i, (idx_start, idx_end, label) in enumerate(region_map):
        if label == BENIGN_LABEL:
            if idx_end < len(df_scores.columns) - 1:
                ax.axvline(x=idx_end + 0.5, color='white', linestyle=':', alpha=0.3)
            continue
        display_label = SHORT_LABEL_MAP.get(label, label)
        color = '#D62728'
        center_x = (idx_start + idx_end) / 2.0
        y_pos = levels[i % 2]

        ax.text(center_x, y_pos, display_label, ha='center', va='bottom',
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        line_y = y_pos + 0.5
        ax.plot([idx_start, idx_end], [line_y, line_y], color=color, linewidth=2.0, clip_on=False)
        if idx_end < len(df_scores.columns) - 1:
            ax.axvline(x=idx_end + 0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(PNG_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Heatmap: {filename}")

# --- MAIN DATA PROCESSING ---

print("Loading dataset:", INPUT_CSV)
try:
    df = pd.read_csv(INPUT_CSV)
    print("Dataset shape:", df.shape)
except FileNotFoundError:
    print(f"Error: File {INPUT_CSV} not found. Please check path.")
    df = pd.DataFrame(columns=['Attack Type'])

LABEL_COL = 'Attack Type'
if LABEL_COL not in df.columns:
    if df.shape[0] > 0:
        raise RuntimeError(f"Expected label column '{LABEL_COL}' not found.")
    else:
        print("Empty DataFrame or file not found, skipping logic checks.")

df = df.dropna(subset=[LABEL_COL])
X_all = df.select_dtypes(include=[np.number]).copy()
y_all = df[LABEL_COL].copy()

if X_all.shape[1] == 0 and df.shape[0] > 0:
    raise RuntimeError("No numeric feature columns found.")

# Prepare sliding windows
n_rows = len(df)
windows = build_windows(n_rows, WINDOW_SIZE, STEP)
print(f"Computed {len(windows)} windows (window_size={WINDOW_SIZE}, step={STEP})")

features = X_all.columns.tolist()
col_names = [f"win_{i+1}" for i in range(len(windows))]

# Initialize DataFrames to collect per-window scores for each method
scores_rf = pd.DataFrame(0.0, index=features, columns=col_names)
scores_perm = pd.DataFrame(0.0, index=features, columns=col_names)
scores_anova = pd.DataFrame(0.0, index=features, columns=col_names)
scores_mi = pd.DataFrame(0.0, index=features, columns=col_names)

window_detected_labels = []
attack_consensus_data = {}  # Stores unified feature importance vectors by attack type

t_total = time.perf_counter()

for i, (s, e) in enumerate(windows):
    win_num = i + 1
    col_name = f"win_{win_num}"

    X_win = X_all.iloc[s:e]
    y_win = y_all.iloc[s:e]

    # Detect the primary attack label within the current window
    counts = y_win.value_counts()
    attack_candidates = counts.index[counts.index != BENIGN_LABEL].tolist()
    primary_label = attack_candidates[0] if len(attack_candidates) > 0 else BENIGN_LABEL
    window_detected_labels.append(primary_label)

    n_classes = y_win.nunique()
    print(f"\nWindow {win_num}: rows [{s}:{e}] => Label: '{primary_label}' (Classes: {n_classes})")

    if len(X_win) < 5 or n_classes < 2:
        print("  -> Skipping (too few samples or only 1 class).")
        continue

    # Initialize execution timers for performance tracking
    t_rf = t_perm = t_anova = t_mi = 0.0

    # 1. Random Forest Feature Importance
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    try:
        rf.fit(X_win, y_win)
        imp_rf = pd.Series(rf.feature_importances_, index=features).fillna(0.0)
        scores_rf[col_name] = normalize_series(imp_rf)
    except Exception as ex:
        print(f"  -> RF error: {ex}")
    t_rf = time.perf_counter() - t0

    # 2. Permutation Importance
    t0 = time.perf_counter()
    try:
        perm_res = permutation_importance(rf, X_win, y_win, n_repeats=3, random_state=42, n_jobs=-1)
        imp_perm = pd.Series(perm_res.importances_mean, index=features).fillna(0.0)
        imp_perm[imp_perm < 0] = 0
        scores_perm[col_name] = normalize_series(imp_perm)
    except Exception as ex:
        print(f"  -> Permutation error: {ex}")
    t_perm = time.perf_counter() - t0

    # 3. ANOVA F-value
    t0 = time.perf_counter()
    try:
        f_scores, _ = f_classif(X_win, y_win)
        imp_anova = pd.Series(f_scores, index=features).fillna(0.0)
        scores_anova[col_name] = normalize_series(imp_anova)
    except Exception as ex:
        print(f"  -> ANOVA error: {ex}")
    t_anova = time.perf_counter() - t0

    # 4. Mutual Information
    t0 = time.perf_counter()
    try:
        mi_scores = mutual_info_classif(X_win, y_win, discrete_features='auto', 
                                        random_state=42, n_neighbors=3)
        imp_mi = pd.Series(mi_scores, index=features).fillna(0.0)
        scores_mi[col_name] = normalize_series(imp_mi)
    except Exception as ex:
        print(f"  -> Mutual Info error: {ex}")
    t_mi = time.perf_counter() - t0

    print(f"  -> [Times] RF: {t_rf:.3f}s | Perm: {t_perm:.3f}s | ANOVA: {t_anova:.3f}s | MutualInfo: {t_mi:.3f}s")

    # Aggregate vectors to create a unified consensus for the current attack
    if primary_label != BENIGN_LABEL and len(X_win) >= 10 and n_classes >= 2:
        cons_vec = normalize_array(rf.feature_importances_) + \
                   normalize_array(perm_res.importances_mean) + \
                   normalize_array(f_scores) + \
                   normalize_array(mi_scores)
        attack_consensus_data.setdefault(primary_label, []).append(cons_vec)

print(f"\nFinished processing all windows in {time.perf_counter()-t_total:.2f}s")

# --- SAVE NUMERICAL DATA (CSV) ---
scores_rf.to_csv(os.path.join(CSV_DIR, 'scores_randomforest.csv'))
scores_perm.to_csv(os.path.join(CSV_DIR, 'scores_permutation.csv'))
scores_anova.to_csv(os.path.join(CSV_DIR, 'scores_anova.csv'))
scores_mi.to_csv(os.path.join(CSV_DIR, 'scores_mutual_info.csv'))

agg_means = pd.DataFrame({
    'RF': scores_rf.mean(axis=1),
    'Permutation': scores_perm.mean(axis=1),
    'ANOVA': scores_anova.mean(axis=1),
    'MutualInfo': scores_mi.mean(axis=1)
})
agg_means.to_csv(os.path.join(CSV_DIR, 'feature_importances_aggregated_all.csv'))

# --- GENERATE PLOTS (PNG) ---
dynamic_attack_map = get_contiguous_regions(window_detected_labels)
plot_heatmap_with_annotations(scores_rf, dynamic_attack_map, "Random Forest (Gini)", "heatmap_rf.png")
plot_heatmap_with_annotations(scores_perm, dynamic_attack_map, "Permutation Importance", "heatmap_permutation.png")
plot_heatmap_with_annotations(scores_anova, dynamic_attack_map, "ANOVA F-value", "heatmap_anova.png")
plot_heatmap_with_annotations(scores_mi, dynamic_attack_map, "Mutual Information", "heatmap_mutual_info.png")

print("Generating Global Comparison Plot...")

agg_means['Sum'] = agg_means.sum(axis=1)
agg_sorted = agg_means.sort_values('Sum', ascending=True)
agg_sorted_clean = agg_sorted.drop(columns=['Sum'])

n_features = len(agg_sorted_clean)
fig_h = max(12, n_features * 0.25)
fig, axes = plt.subplots(1, 4, figsize=(20, fig_h), sharey=True)
methods = ['RF', 'Permutation', 'ANOVA', 'MutualInfo']
colors = ['steelblue', 'orange', 'seagreen', 'purple']
for i, col in enumerate(methods):
    ax = axes[i]
    ax.barh(agg_sorted_clean.index, agg_sorted_clean[col], color=colors[i], height=0.7)
    ax.set_title(col, fontsize=14)
    ax.set_xlabel('Score', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    if i > 0:
        ax.tick_params(axis='y', which='both', left=False)

plt.suptitle("Global Feature Importance Comparison", fontsize=18, y=1.005)
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, 'global_comparison_4_methods.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved Global Plot.")

print("Generating Correlation Matrix of Methods...")

corr_matrix = agg_sorted_clean.corr(method='spearman')
fig_corr, ax_corr = plt.subplots(figsize=(8, 7))
im = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=0, vmax=1)
for ii in range(len(corr_matrix)):
    for jj in range(len(corr_matrix)):
        ax_corr.text(jj, ii, f"{corr_matrix.iloc[ii, jj]:.2f}", ha="center", va="center", color="black", fontsize=12, fontweight='bold')

ax_corr.set_xticks(np.arange(len(corr_matrix.columns)))
ax_corr.set_yticks(np.arange(len(corr_matrix.index)))
ax_corr.set_xticklabels(corr_matrix.columns, fontsize=11)
ax_corr.set_yticklabels(corr_matrix.index, fontsize=11)
plt.colorbar(im, label='Spearman Rank Correlation', fraction=0.046, pad=0.04)
plt.title("Correlation Similarity Matrix (Method Consensus)", fontsize=14, pad=15)
plt.tight_layout()
out_corr = os.path.join(PNG_DIR, 'methods_correlation_matrix.png')
plt.savefig(out_corr, dpi=150, bbox_inches='tight')
plt.close(fig_corr)
print(f"Saved Correlation Matrix: {out_corr}")

# --- ATTACK CONSENSUS AND JACCARD SIMILARITY ---
print("\nComputing consensus signatures and Jaccard matrix...")

attack_signatures = {}
for label, vecs in attack_consensus_data.items():
    mean_vec = np.mean(vecs, axis=0)
    s_consensus = pd.Series(mean_vec, index=features)
    top_k = s_consensus.nlargest(TOP_K_FEATURES)
    attack_signatures[label] = set(top_k.index.tolist())
    print(f"\n==== Signature for '{label}' ({len(vecs)} windows) ====")
    for rank, (fname, score) in enumerate(top_k.items(), 1):
        print(f"{rank:02d}. {fname:<50} (Score: {score:.4f})")

labels_present = sorted(attack_signatures.keys())
n = len(labels_present)
jaccard_mat = np.zeros((n,n))
for ii in range(n):
    for jj in range(n):
        a = attack_signatures[labels_present[ii]]
        b = attack_signatures[labels_present[jj]]
        inter = len(a & b)
        union = len(a | b)
        jaccard_mat[ii,jj] = inter/union if union>0 else 0.0

df_jaccard = pd.DataFrame(jaccard_mat, index=labels_present, columns=labels_present)

plt.figure(figsize=(14,12))
sns.heatmap(df_jaccard, annot=True, fmt=".2f", cmap="YlOrRd",
            vmin=0, vmax=1, square=True, linewidths=.5,
            cbar_kws={"shrink": .8, "label": "Jaccard Index"})
plt.title(f"Attack Similarity Matrix (Consensus of 4 Methods)\nTop {TOP_K_FEATURES} Features Comparison", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

# Force Jaccard matrix plots and CSVs into their respective subdirectories
out_path = os.path.join(PNG_DIR, 'consensus_attack_similarity_jaccard.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved Matrix Plot: {out_path}")

csv_path = os.path.join(CSV_DIR, 'jaccard_matrix.csv')
df_jaccard.to_csv(csv_path)
print(f"Saved Matrix CSV: {csv_path}")

print("Done.")