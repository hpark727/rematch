import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def load_data(belief_index_path, belief_signal_index_path):
    # Load JSONs
    with open(belief_index_path, "r") as f:
        b_index = json.load(f)

    with open(belief_signal_index_path, "r") as f:
        bs_index = json.load(f)

    df_b  = pd.DataFrame(b_index)   # LLM index
    df_bs = pd.DataFrame(bs_index)  # concept-signal index

    print('df_b columns: ' + str(df_b.columns.tolist()))
    print('df_bs columns: ' + str(df_bs.columns.tolist()))

    df_b["llm_row"] = np.arange(len(df_b))
    df_bs["signal_row"] = np.arange(len(df_bs))

    # Keep only what you want from the signal side to avoid duplicate columns
    sig_cols = ["belief_id", "signal_row"]
    for c in ["text"]:  # include if it exists in df_bs
        if c in df_bs.columns:
            sig_cols.append(c)

    # Merge on belief_id (real join)
    main = df_b.merge(df_bs[sig_cols], on="belief_id", how="left")

    print(main.shape)
    return main

df = load_data("llm_belief_embeddings/belief_index.json", "llm_belief_embeddings/belief_signal_index.json")

dboed = df.groupby("agent_type").get_group("DBOEDSCOTUSJudgmentPredictionAgent").copy()
boed = df.groupby("agent_type").get_group("BOEDSCOTUSJudgmentPredictionAgent").copy()

print('DBOED shape:', dboed.shape)
print('BOED shape:', boed.shape)

def build_trajectory_map(df, step_col="step_index", vec_row_col="signal_row"):
    if vec_row_col not in df.columns:
        raise ValueError(f"{vec_row_col} not in df columns. Available: {df.columns.tolist()}")

    df2 = df.dropna(subset=[vec_row_col]).copy()
    df2[vec_row_col] = df2[vec_row_col].astype(int)

    traj_map = {}
    for tid, g in df2.groupby("trajectory_id", sort=False):
        g = g.sort_values(step_col)
        rows = g[vec_row_col].to_numpy()
        if len(rows) >= 2:
            traj_map[tid] = rows
    return traj_map

traj_map_dboed = build_trajectory_map(dboed)
traj_map_boed   = build_trajectory_map(boed)

# print('Number of DBOED trajectories:', len(traj_map_dboed))
# print('Number of BOED trajectories:', len(traj_map_boed))

# print('dboed traj: ', list(traj_map_dboed.items())[0])

def get_embeddings(embedding_path):
    emb = np.load(embedding_path)
    print(f"Loaded embeddings from {embedding_path}, shape={emb.shape}, dtype={emb.dtype}")
    return emb

emb = get_embeddings("concept_graph_embeddings/concept_embeddings.npy")
sig = get_embeddings("concept_graph_embeddings/belief_signals.npy")

print('emb shape:', emb.shape)
print('sig shape:', sig.shape)

vector_embeddings = sig @ emb

print('vector_embeddings shape:', vector_embeddings.shape)

# sanity check making sure that total entries in trajectories match
totsteps = 0
for i, rows in traj_map_dboed.items():
    totsteps += len(rows)
for i, rows in traj_map_boed.items():
    totsteps += len(rows)

print('Total trajectory steps:', totsteps)

def l2_normalize(A, axis=1, eps=1e-10):
    norms = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / np.maximum(norms, eps)

norm = l2_normalize(vector_embeddings, axis=1)

print('norm shape:', norm.shape)

def cos(A, B):
    return cosine_similarity(A, B)
       

def get_row(tid, traj_map):
    return traj_map[tid]

EPS = 1e-12
# compute adjacency cosine similarity sequences
def trajectory_adj_cos_sequences(trajectory_map, embeddings, pad_value=np.nan, min_steps=2):
    """
    Returns a wide DataFrame:
      - rows = trajectories
      - columns = adj_cos_0, adj_cos_1, ... (padded with NaN)
    """
    Xn = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), EPS, None)
    n_rows = embeddings.shape[0]

    # compute sequences
    seqs = {}
    max_len = 0
    for tid, idxs in trajectory_map.items():
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs) < min_steps:
            continue
        if np.any((idxs < 0) | (idxs >= n_rows)):
            raise IndexError(f"Trajectory {tid} has out-of-range indices.")
        V = Xn[idxs]
        adj = np.sum(V[:-1] * V[1:], axis=1)  # length T-1
        seqs[tid] = adj
        max_len = max(max_len, len(adj))

    # pad to same length
    data = {}
    for tid, adj in seqs.items():
        padded = np.full((max_len,), pad_value, dtype=float)
        padded[:len(adj)] = adj
        data[tid] = padded

    df_seq = pd.DataFrame.from_dict(data, orient="index")
    df_seq.index.name = "trajectory_id"
    df_seq.columns = [f"adj_cos_{i}" for i in range(max_len)]
    return df_seq

# histogram plotting function
def plot_adj_cos_histogram(seq_df, agent_type, bins=50):
    adj_cos_values = dboed.values.flatten()
    adj_cos_values = adj_cos_values[~np.isnan(adj_cos_values)]

    plt.figure(figsize=(8, 5))
    sns.histplot(adj_cos_values, bins=bins, kde=True)
    plt.title(f'Adjacency Cosine Similarity Histogram for {agent_type}')
    plt.xlabel('Adjacency Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
def stats(name, A):
    A = np.asarray(A)
    finite = np.isfinite(A)
    print(f"{name}: shape={A.shape} dtype={A.dtype}")
    print(f"  finite: {finite.mean()*100:.2f}%  (nan={np.isnan(A).sum()}, inf={(~finite & ~np.isnan(A)).sum()})")
    if finite.any():
        Af = A[finite]
        print(f"  min={Af.min():.3g} max={Af.max():.3g} mean={Af.mean():.3g} std={Af.std():.3g}")
        print(f"  absmax={np.max(np.abs(Af)):.3g}")

# modify dboed trajectory map to separate by belief type
dboed_design_map = {}
dboed_task_map = {}

for tid, step_rows in traj_map_dboed.items():
    tid = str(tid)
    dboed_design_map[tid] = []
    dboed_task_map[tid] = []

    for idx in step_rows:
        bt = df.loc[idx, "belief_type"]
        if bt == "design_beliefs":
            dboed_design_map[tid].append(idx)
        elif bt == "task_beliefs":
            dboed_task_map[tid].append(idx)
        else:
            print("Unexpected belief_type:", tid, idx, bt)

print('dboed_design_map size:', len(dboed_design_map))
print('dboed_task_map size:', len(dboed_task_map))

# check that the design map indices correspond to design beliefs
check = True
for tid, idx_list in dboed_design_map.items():
    for idx in idx_list:
        if df.loc[idx, "belief_type"] != "design_beliefs":  
            check = False
            print("Mismatch in design_map:", tid, idx, df.loc[idx, "belief_type"])
            break
    if not check:
        break

if not check:
    print("Error: design map contains non-design belief!")
else:
    print("Design map check passed.")

# check that the task map indices correspond to task beliefs
check = True
for tid, idx_list in dboed_task_map.items():
    for idx in idx_list:
        if df.loc[idx, "belief_type"] != "task_beliefs":  
            check = False
            print("Mismatch in task_map:", tid, idx, df.loc[idx, "belief_type"])
            break
    if not check:
        break

if not check:
    print("Error: task map contains non-task belief!")
else:
    print("Task map check passed.")

dboed_design_sim = trajectory_adj_cos_sequences(dboed_design_map, vector_embeddings, pad_value=np.nan, min_steps=2)
dboed_task_sim   = trajectory_adj_cos_sequences(dboed_task_map,   vector_embeddings, pad_value=np.nan, min_steps=2)
dboed_df = trajectory_adj_cos_sequences(traj_map_dboed, vector_embeddings, pad_value=np.nan, min_steps=2)
boed_df = trajectory_adj_cos_sequences(traj_map_boed,   vector_embeddings, pad_value=np.nan, min_steps=2)

from stats_and_plots import CosineRunAnalyzer as sp

an = sp(dboed_design_sim, dboed_task_sim, boed_df)
an.plot_overlay_time_series_mean_std(
    {
        "DBOED Design": dboed_design_sim,
        "DBOED Task": dboed_task_sim,
        "BOED": boed_df,
    },
    title="Adjacent cosine similarity over time (mean ± 1 st. dev.)",
    show_band=True,
    band_alpha=0.20,
    clip_y=(0.9, 1.0),   # optional zoom; tweak/remove as needed
)
an.show(block=True)

# Classification with RBF-SVM and LOOCV

dboed_design_sim['agent_type'] = 'DBOED_Design'
dboed_task_sim['agent_type']   = 'DBOED_Task'
boed_df['agent_type']          = 'BOED'

combined = pd.concat([dboed_design_sim, dboed_task_sim, boed_df], axis=0)
print('combined shape' , combined.shape)

X = combined.drop(columns=['agent_type']).to_numpy()

row_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(row_means, inds[1])

y = combined['agent_type'].to_numpy()

loo = LeaveOneOut()

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False))
])

y_pred = cross_val_predict(pipe, X, y, cv=loo)

print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred, digits=3))


# Deep Learning Binary Classifier with LOOCV
# from sklearn.impute import SimpleImputer
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import LabelEncoder

# dboed_design_sim["agent_type"] = "DBOED"
# dboed_task_sim["agent_type"]   = "DBOED"
# boed_df["agent_type"]          = "BOED"

# combined = pd.concat([dboed_design_sim, dboed_task_sim, boed_df], axis=0).reset_index(drop=True)

# X = combined.drop(columns=["agent_type"]).to_numpy()
# y = combined["agent_type"].to_numpy()

# X = X.astype(np.float64, copy=False)

# le = LabelEncoder()
# y_enc = le.fit_transform(y)   # e.g., BOED->0, DBOED->1 (order depends on le.classes_)

# loo = LeaveOneOut()

# pipe = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),
#     ("scaler", StandardScaler()),
#     ("clf", MLPClassifier(
#         hidden_layer_sizes=(64, 32),
#         activation="relu",
#         solver="adam",
#         alpha=1e-4,
#         learning_rate_init=1e-3,
#         max_iter=2000,
#         early_stopping=True,      
#         n_iter_no_change=20,
#         random_state=42
#     ))
# ])

# y_pred_enc = cross_val_predict(pipe, X, y_enc, cv=loo)

# y_pred = le.inverse_transform(y_pred_enc)
# y_true = le.inverse_transform(y_enc)

# print(confusion_matrix(y_true, y_pred, labels=["DBOED", "BOED"]))
# print(classification_report(y_true, y_pred, digits=3))

# Permutation test for significance
import numpy as np

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

def permutation_test_loocv_accuracy(X, y, n_perms=500, random_state=0, estimator=None):
    X = np.asarray(X)
    y = np.asarray(y)

    if estimator is None:
        raise ValueError("Pass your Pipeline as estimator=...")

    loo = LeaveOneOut()

    # Observed
    y_pred = cross_val_predict(estimator, X, y, cv=loo)
    obs_acc = (y_pred == y).mean()

    rng = np.random.default_rng(random_state)
    perm_accs = np.empty(n_perms, dtype=float)

    for b in range(n_perms):
        print(f"Permutation {b+1}/{n_perms}", end="\r")

        y_perm = rng.permutation(y)
        y_perm_pred = cross_val_predict(estimator, X, y_perm, cv=loo)
        perm_accs[b] = (y_perm_pred == y_perm).mean()

    p_value = (1.0 + np.sum(perm_accs >= obs_acc)) / (n_perms + 1.0)
    return obs_acc, perm_accs, p_value


obs_acc, perm_accs, p = permutation_test_loocv_accuracy(
    X, y, n_perms=2000, random_state=0, estimator=pipe
)

print("\n--- Permutation test ---")
print(f"Observed LOOCV accuracy: {obs_acc:.3f}")
print(f"Permutation mean accuracy: {perm_accs.mean():.3f} (std={perm_accs.std():.3f})")
print(f"Empirical one-sided p-value: {p:.6f}")
print("Perm acc quantiles [50%, 90%, 95%, 99%]:",
      np.round(np.quantile(perm_accs, [0.50, 0.90, 0.95, 0.99]), 3))