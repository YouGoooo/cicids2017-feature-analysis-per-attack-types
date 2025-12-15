import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Parameters (adjust as needed) ---
INPUT_CSV = './data/cicids2017_filtered.csv'
OUT_DIR = './results'
WINDOW_SIZE = 50000
STEP = WINDOW_SIZE          # set STEP < WINDOW_SIZE for overlapping windows
K = 10                      # number of top features per window for SelectKBest
TOP_PLOT = 40               # number of top features to show in comparison plot
USE_RF = True
RF_N_ESTIMATORS = 100       # Increased from 50
RF_MAX_DEPTH = 5            # Decreased from 10 to prevent overfitting/speed up
USE_PERM = True             # compute permutation importance
PERM_N_REPEATS = 5          # Reduced to 5 for speed (adjust to 10 for precision)
PERM_SCORING = 'accuracy'   # scoring used for permutation_importance
SORT_BY = 'mean_score_rf'   # choose 'mean_score_skb' or 'mean_score_rf' or 'mean_score_perm'

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading dataset: {INPUT_CSV}")
if not os.path.exists(INPUT_CSV):
    # Dummy data generation for testing if file doesn't exist
    print("WARNING: Input file not found. Generating dummy data for demonstration...")
    data_size = 100000
    df = pd.DataFrame(np.random.rand(data_size, 20), columns=[f'feat_{i}' for i in range(20)])
    df['Attack Type'] = np.random.choice(['BENIGN', 'DDoS', 'PortScan'], size=data_size)
else:
    df = pd.read_csv(INPUT_CSV)

print("Dataset shape:", df.shape)

LABEL_COL = 'Attack Type'
if LABEL_COL not in df.columns:
    raise RuntimeError(f"Expected label column '{LABEL_COL}' not found.")

df = df.dropna(subset=[LABEL_COL])
X_all = df.select_dtypes(include=[np.number]).copy()
y_all = df[LABEL_COL].copy()

if X_all.shape[1] == 0:
    raise RuntimeError("No numeric feature columns found.")

# Build windows
n_rows = len(df)
windows = []
start = 0
while start < n_rows:
    end = min(start + WINDOW_SIZE, n_rows)
    windows.append((start, end))
    if end == n_rows:
        break
    start += STEP

print(f"Computed {len(windows)} windows (window_size={WINDOW_SIZE}, step={STEP})")

features = X_all.columns.tolist()
window_names = [f"window_{i}" for i in range(len(windows))]

# Initialize DataFrames for scores
scores_skb = pd.DataFrame(0.0, index=features, columns=window_names)
scores_rf = pd.DataFrame(0.0, index=features, columns=window_names) if USE_RF else None
scores_perm = pd.DataFrame(0.0, index=features, columns=window_names) if USE_PERM else None
scores_mi = pd.DataFrame(0.0, index=features, columns=window_names)

t_total = time.perf_counter()

for i, (s, e) in enumerate(windows):
    X_win = X_all.iloc[s:e]
    y_win = y_all.iloc[s:e]
    n_classes = y_win.nunique()
    
    print(f"\nWindow {i}: rows [{s}:{e}] => samples={len(X_win)}, classes={n_classes}")
    
    # Skip invalid windows
    if n_classes < 2 or len(X_win) < 2:
        print("  -> single class or too few samples: skipping.")
        continue

    # --- 1. SelectKBest ---
    t0 = time.perf_counter()
    # Handle NaN/Inf in features just in case
    X_win = X_win.fillna(0)
    
    skb = SelectKBest(score_func=f_classif, k=min(K, X_win.shape[1]))
    try:
        skb.fit(X_win, y_win)
        win_scores = pd.Series(skb.scores_, index=X_win.columns).fillna(0.0)
        
        # Normalize
        if win_scores.max() > win_scores.min():
            win_scores = (win_scores - win_scores.min()) / (win_scores.max() - win_scores.min())
        else:
            win_scores = 0.0 * win_scores
        
        scores_skb[f"window_{i}"] = win_scores
    except Exception as ex:
        print("  -> SelectKBest error:", ex)

    t1 = time.perf_counter()
    print(f"  SelectKBest done in {t1-t0:.2f}s")

    # --- 2. RandomForest Importance ---
    rf_model = None
    if USE_RF:
        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                    random_state=42, n_jobs=-1, class_weight='balanced_subsample')
        try:
            rf.fit(X_win, y_win)
            rf_scores = pd.Series(rf.feature_importances_, index=X_win.columns).fillna(0.0)
            rf_model = rf

            # Accuracy check
            y_pred_rf = rf.predict(X_win)
            acc_rf = accuracy_score(y_win, y_pred_rf)
            print(f"  RandomForest accuracy (train) = {acc_rf:.4f}")
            
            # Confusion Matrix (optional print)
            try:
                unique_labels = sorted(y_win.unique())
                cm = confusion_matrix(y_win, y_pred_rf, labels=unique_labels)
                # print(f"  Confusion Matrix shapes: {cm.shape}")
            except Exception as e:
                print(f"  Could not compute confusion matrix: {e}")

            # Normalize
            if rf_scores.max() > rf_scores.min():
                rf_scores = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min())
            else:
                rf_scores = 0.0 * rf_scores
                
            scores_rf[f"window_{i}"] = rf_scores

        except Exception as ex:
            print("  -> RandomForest error:", ex)
            rf_model = None
        
        t1 = time.perf_counter()
        print(f"  RandomForest done in {t1-t0:.2f}s")

    # --- 3. Permutation Importance ---
    if USE_PERM:
        t0 = time.perf_counter()
        # Use the RF trained above, or fit a new one if RF wasn't used/failed
        model_for_perm = rf_model
        
        if model_for_perm is None:
            model_for_perm = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                                    random_state=42, n_jobs=-1, class_weight='balanced_subsample')
            try:
                model_for_perm.fit(X_win, y_win)
            except Exception as ex:
                print("  -> Permutation model training error:", ex)
                model_for_perm = None

        if model_for_perm is not None:
            try:
                perm_res = permutation_importance(model_for_perm, X_win, y_win,
                                                  n_repeats=PERM_N_REPEATS, random_state=42,
                                                  n_jobs=-1, scoring=PERM_SCORING)
                perm_mean = pd.Series(perm_res.importances_mean, index=X_win.columns).fillna(0.0)
                
                # Normalize
                if perm_mean.max() > perm_mean.min():
                    perm_scores = (perm_mean - perm_mean.min()) / (perm_mean.max() - perm_mean.min())
                else:
                    perm_scores = 0.0 * perm_mean
                
                scores_perm[f"window_{i}"] = perm_scores
                
            except Exception as ex:
                print("  -> permutation_importance error:", ex)
        
        t1 = time.perf_counter()
        print(f"  Permutation importance done in {t1-t0:.2f}s")

    # --- 4. Mutual Information ---
    t0 = time.perf_counter()
    try:
        # mutual_info_classif requires discrete features usually, but handles continuous too.
        # It can be slow on large data.
        mi_scores = mutual_info_classif(X_win, y_win, discrete_features='auto', random_state=42)
        mi_series = pd.Series(mi_scores, index=X_win.columns).fillna(0.0)
        
        # Normalize
        if mi_series.max() > mi_series.min():
            mi_normalized = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min())
        else:
            mi_normalized = 0.0 * mi_series
            
        scores_mi[f"window_{i}"] = mi_normalized
    except Exception as ex:
        print("  -> Mutual Info error:", ex)

    t1 = time.perf_counter()
    print(f"  Mutual Information done in {t1-t0:.2f}s")

print(f"\nFinished processing windows in {time.perf_counter()-t_total:.2f}s")

# --- Save Results ---

# Helper to save and print
def save_csv(df_in, name):
    path = os.path.join(OUT_DIR, name)
    df_in.to_csv(path)
    print(f"Saved: {path}")

save_csv(scores_skb, 'feature_importances_per_window_selectkbest.csv')
save_csv(scores_mi, 'feature_importances_per_window_mutual_information.csv')

if USE_RF:
    save_csv(scores_rf, 'feature_importances_per_window_randomforest.csv')
if USE_PERM:
    save_csv(scores_perm, 'feature_importances_per_window_permutation.csv')

# --- Aggregate Results ---
agg_df = pd.DataFrame()
agg_df['mean_score_skb'] = scores_skb.mean(axis=1)
agg_df['max_score_skb'] = scores_skb.max(axis=1)
agg_df['mean_score_mi'] = scores_mi.mean(axis=1)
agg_df['max_score_mi'] = scores_mi.max(axis=1)

if USE_RF:
    agg_df['mean_score_rf'] = scores_rf.mean(axis=1)
    agg_df['max_score_rf'] = scores_rf.max(axis=1)

if USE_PERM:
    agg_df['mean_score_perm'] = scores_perm.mean(axis=1)
    agg_df['max_score_perm'] = scores_perm.max(axis=1)

agg_df = agg_df.fillna(0.0).sort_values(by='mean_score_skb', ascending=False)
save_csv(agg_df, 'feature_importances_per_window_aggregated.csv')

# --- Plotting ---

global_cmap = 'viridis'

def plot_heatmap(df_scores, title, filename):
    if df_scores is None or df_scores.empty:
        return
    plt.figure(figsize=(14, max(6, len(features) * 0.12)))
    v_max = df_scores.values.max()
    v_max = v_max if v_max > 0 else 1e-9
    
    im = plt.imshow(df_scores.values, aspect='auto', cmap=global_cmap,
                    interpolation='nearest', vmin=0.0, vmax=v_max)
    plt.colorbar(im, label='Normalized importance (0-1)')
    
    n_feats = len(df_scores.index)
    step_y = max(1, n_feats // 40)
    plt.yticks(range(0, n_feats, step_y), [df_scores.index[j] for j in range(0, n_feats, step_y)], fontsize=7)
    plt.xticks(range(len(df_scores.columns)), df_scores.columns, rotation=45, ha='right', fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved plot: {filename}")

plot_heatmap(scores_skb, "SelectKBest — Feature Importance", 'heatmap_selectkbest.png')
plot_heatmap(scores_mi, "Mutual Info — Feature Importance", 'heatmap_mutual_info.png')
if USE_RF:
    plot_heatmap(scores_rf, "RandomForest — Feature Importance", 'heatmap_randomforest.png')
if USE_PERM:
    plot_heatmap(scores_perm, "Permutation — Feature Importance", 'heatmap_permutation.png')

# Global Comparison Plot
cmp_cols = ['mean_score_skb', 'mean_score_mi']
if USE_RF: cmp_cols.append('mean_score_rf')
if USE_PERM: cmp_cols.append('mean_score_perm')

cmp_df = agg_df[cmp_cols].copy()
# Normalize for visualization (mean centering to 0.5)
for col in cmp_cols:
    m = cmp_df[col].mean()
    if m > 0:
        cmp_df[col] = (cmp_df[col] / m) * 0.5

# Sort for plotting
sort_col_final = SORT_BY if SORT_BY in cmp_df.columns else cmp_cols[0]
cmp_df_sorted = cmp_df.sort_values(sort_col_final, ascending=False)

# Plot top N features only if list is too long
if len(cmp_df_sorted) > TOP_PLOT:
    plot_df = cmp_df_sorted.head(TOP_PLOT)
    title_suffix = f"(Top {TOP_PLOT})"
else:
    plot_df = cmp_df_sorted
    title_suffix = "(All)"

plt.figure(figsize=(15, 8))
plot_df.plot(kind='bar', width=0.8, figsize=(15, 8))
plt.ylabel('Relative Importance (Scaled)')
plt.title(f'Global Importance Comparison {title_suffix}')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'global_comparison.png'), dpi=150)
plt.close()
print("Saved global comparison plot.")

# Time Series Random Plot
N_RANDOM_FEATURES = 5
rng = np.random.RandomState(42)
selected_ts_features = rng.choice(features, size=min(N_RANDOM_FEATURES, len(features)), replace=False)
ts_idx = range(len(windows))

for feat in selected_ts_features:
    plt.figure(figsize=(10, 4))
    plt.plot(ts_idx, scores_skb.loc[feat], label='SelectKBest', marker='o')
    plt.plot(ts_idx, scores_mi.loc[feat], label='MutInfo', marker='x')
    if USE_RF:
        plt.plot(ts_idx, scores_rf.loc[feat], label='RandomForest', marker='s')
    if USE_PERM:
        plt.plot(ts_idx, scores_perm.loc[feat], label='Permutation', marker='^')
    
    plt.title(f"Feature: {feat}")
    plt.legend()
    plt.tight_layout()
    safe_name = re.sub(r'[^A-Za-z0-9]', '_', feat)
    plt.savefig(os.path.join(OUT_DIR, f"ts_{safe_name}.png"))
    plt.close()

print("\nDone.")