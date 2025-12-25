import json
import numpy as np
import pandas as pd

# Load JSONs
with open("llm_belief_embeddings/belief_index.json", "r") as f:
    b_index = json.load(f)

with open("concept_graph_embeddings/belief_signal_index.json", "r") as f:
    bs_index = json.load(f)

df_b  = pd.DataFrame(b_index)   # LLM index
df_bs = pd.DataFrame(bs_index)  # concept-signal index

print('df_b columns: ' + str(df_b.columns.tolist()))
print('df_bs columns: ' + str(df_bs.columns.tolist()))


# Add row pointers (super useful later to index into .npy matrices)
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
#print("signal coverage:", main["signal_row"].notna().mean())
#print(main.columns)

target = main['agent_type']
