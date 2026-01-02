import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def build_trajectory_map(df, step_col="step_number", vec_row_col="signal_row"):
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

