import json
import numpy as np
import pandas as pd

BELIEF_INDEX_PATH = "llm_belief_embeddings/belief_index.json"
C_EMB = "concept_graph_embeddings/concept_embeddings.npy"
B_SIG = "concept_graph_embeddings/belief_signals.npy" 

with open(BELIEF_INDEX_PATH, "r") as f:
    idx = json.load(f)
df = pd.DataFrame(idx)

c_emb = np.load(C_EMB)
b_sig = np.load(B_SIG)

def stats(name, A):
    A = np.asarray(A)
    finite = np.isfinite(A)
    print(f"{name}: shape={A.shape} dtype={A.dtype}")
    print(f"  finite: {finite.mean()*100:.2f}%  (nan={np.isnan(A).sum()}, inf={(~finite & ~np.isnan(A)).sum()})")
    if finite.any():
        Af = A[finite]
        print(f"  min={Af.min():.3g} max={Af.max():.3g} mean={Af.mean():.3g} std={Af.std():.3g}")
        print(f"  absmax={np.max(np.abs(Af)):.3g}")


print(stats("b_sig", b_sig)) 
print(stats("c_emb", c_emb))

b_sig = b_sig.astype(np.float32, copy=False)
c_emb = c_emb.astype(np.float32, copy=False)

row_sum = b_sig.sum(axis=1, keepdims=True)
b_sig_norm = b_sig / (row_sum + 1e-8)  

X = b_sig_norm @ c_emb
y_codes = df['agent_type'].to_numpy()
groups = df['trajectory_id'].to_numpy()

def group_train_test_split(groups, test_size=0.2, seed=42):
    """
    Split indices into train/test such that no group appears in both.
    groups: array-like (n_samples,) e.g. trajectory_id per row
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_test = int(np.ceil(test_size * len(uniq)))
    test_groups = set(uniq[:n_test])

    test_mask = np.isin(groups, list(test_groups))
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


train_idx, test_idx = group_train_test_split(groups, test_size=0.2, seed=42)

X_train, y_train = X[train_idx], y_codes[train_idx]
X_test,  y_test  = X[test_idx],  y_codes[test_idx]

assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
print("train groups:", len(np.unique(groups[train_idx])),
      "test groups:", len(np.unique(groups[test_idx])))
