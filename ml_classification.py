import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

X = b_sig #@ c_emb
y_codes = df['agent_type'].to_numpy()
groups = df['trajectory_id'].to_numpy()

from collections import defaultdict
import numpy as np

def stratified_group_split(groups, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    y = np.asarray(y)

    # one label per group (assumes all rows in a group share the same label)
    group_to_label = {}
    for g in np.unique(groups):
        lab = y[groups == g]
        group_to_label[g] = lab[0]

    # bucket groups by label
    buckets = defaultdict(list)
    for g, lab in group_to_label.items():
        buckets[lab].append(g)

    test_groups = set()
    for lab, gs in buckets.items():
        gs = np.array(gs)
        rng.shuffle(gs)
        n_test = int(np.ceil(test_size * len(gs)))
        test_groups.update(gs[:n_test])

    test_mask = np.isin(groups, list(test_groups))
    return np.where(~test_mask)[0], np.where(test_mask)[0]



train_idx, test_idx = stratified_group_split(groups, y_codes, test_size=0.2, seed=42)

X_train, y_train = X[train_idx], y_codes[train_idx]
X_test,  y_test  = X[test_idx],  y_codes[test_idx]

assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
print("train groups:", len(np.unique(groups[train_idx])),
      "test groups:", len(np.unique(groups[test_idx])))

def standard_scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = standard_scale(X_train, X_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

def dummy_baseline(X_train_scaled, y_train, X_test_scaled, y_test):
    from sklearn.dummy import DummyClassifier
    dum = DummyClassifier(strategy="most_frequent")
    dum.fit(X_train_scaled, y_train)
    print("dummy acc:", dum.score(X_test_scaled, y_test))


def SVM_classification(X_train_scaled, y_train, X_test_scaled, y_test):
    clf = SVC(kernel='poly', random_state=42)
    clf.fit(X_train_scaled, y_train)
    prediction = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, prediction)
    print(f"SVM classification accuracy: {acc*100:.2f}%")

def MLP_classification(X_train_scaled, y_train, X_test_scaled, y_test):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
    clf.fit(X_train_scaled, y_train)
    prediction = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, prediction)
    print(f"MLP classification accuracy: {acc*100:.2f}%")

def l_reg_classification(X_train_scaled, y_train, X_test_scaled, y_test):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train_scaled, y_train)
    prediction = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, prediction)
    print(f"Logistic Regression classification accuracy: {acc*100:.2f}%")

def random_forest_classification(X_train_scaled, y_train, X_test_scaled, y_test):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    prediction = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, prediction)
    print(f"Random Forest classification accuracy: {acc*100:.2f}%")

print('test set size:', X_test_scaled.shape[0])
dummy_baseline(X_train_scaled, y_train, X_test_scaled, y_test)
SVM_classification(X_train_scaled, y_train, X_test_scaled, y_test)
MLP_classification(X_train_scaled, y_train, X_test_scaled, y_test)
l_reg_classification(X_train_scaled, y_train, X_test_scaled, y_test)
random_forest_classification(X_train_scaled, y_train, X_test_scaled, y_test)

train_labels = set(y_train)
test_labels  = set(y_test)
print("labels only in test:", sorted(test_labels - train_labels))
print("labels only in train:", sorted(train_labels - test_labels))

