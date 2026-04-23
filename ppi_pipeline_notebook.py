# %% [markdown]
# # Production-ready PPI Network Analysis Pipeline (igraph + pandas)
#
# This notebook-style script implements an end-to-end protein-protein interaction (PPI)
# analysis workflow:
#
# 1. Load TSV/CSV PPI edge list
# 2. Standardize columns + clean data
# 3. Filter by confidence threshold (0.7 or 700, inferred from score scale)
# 4. Build an undirected weighted graph
# 5. Extract largest connected component (LCC)
# 6. Detect communities (Leiden if available, otherwise Louvain)
# 7. Compute cartography metrics (within-module z-score and participation coefficient)
# 8. Assign Guimera-Amaral roles (R1-R7)
# 9. Compute local centrality metrics (MCC, DMNC, MNC)
# 10. Compute global centrality metrics
# 11. Save all outputs and produce publication-friendly plots

# %% [markdown]
# ## 0) Package installation cell
#
# Run this once in Jupyter if packages are missing:
#
# ```python
# !pip install -U python-igraph pandas numpy scipy matplotlib seaborn plotly matplotlib-venn kaleido
# ```

# %%
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib_venn import venn3


# %% [markdown]
# ## 1) Config cell (edit this section only)

# %%
@dataclass
class Config:
    # Input / output
    input_file: str = "ppi_input.csv"   # <- set path to your CSV/TSV file
    output_dir: str = "output_ppi"

    # Column mapping (edit if your file uses different names)
    source_col: str = "source"
    target_col: str = "target"
    score_col: str = "confidence"

    # Analysis options
    top_n_for_venn: int = 30
    use_weights: bool = True


CFG = Config()


# %% [markdown]
# ## 2) Utility functions
#
# ### Formulas used
#
# **Within-module degree z-score** for node $i$ in module $s$:
#
# $z_i = \dfrac{k_{i,s} - \mu_{k,s}}{\sigma_{k,s}}$
#
# where $k_{i,s}$ is node $i$'s intra-module degree (or weighted intra-module strength if `use_weights=True`),
# and $\mu_{k,s},\sigma_{k,s}$ are the module-wise mean and standard deviation of intra-module degree.
#
# **Participation coefficient**:
#
# $P_i = 1 - \sum_s \left(\dfrac{k_{i,s}}{k_i}\right)^2$
#
# where $k_{i,s}$ is node $i$'s connectivity to module $s$ and $k_i$ is total degree/strength.
#
# **Role assignment (Guimera-Amaral):**
#
# - Non-hubs: $z<2.5$
#   - R1: $P \le 0.05$
#   - R2: $0.05 < P \le 0.62$
#   - R3: $0.62 < P \le 0.80$
#   - R4: $P > 0.80$
# - Hubs: $z\ge2.5$
#   - R5: $P \le 0.30$
#   - R6: $0.30 < P \le 0.75$
#   - R7: $P > 0.75$
#
# **Local centrality assumptions (cytoHubba-inspired):**
#
# - **MNC**: size of the largest connected component in the 1-hop neighborhood subgraph of a node (excluding the node).
# - **DMNC**: for the same maximal neighborhood component with $n$ nodes and $m$ edges,
#   $\text{DMNC}=\frac{m}{n^{1.7}}$ (if $n>0$, else 0). Exponent 1.7 follows common cytoHubba usage.
# - **MCC**: sum over maximal cliques containing node $v$:
#   $\text{MCC}(v)=\sum_{C\ni v} (|C|-1)!$

# %%
def detect_separator(path: str) -> str:
    """Detect CSV/TSV separator safely via csv.Sniffer; fallback by extension/heuristic."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(8192)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        if path.lower().endswith(".tsv"):
            return "\t"
        if sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def load_and_standardize(cfg: Config) -> pd.DataFrame:
    sep = detect_separator(cfg.input_file)
    print(f"Detected separator: {repr(sep)}")

    df = pd.read_csv(cfg.input_file, sep=sep)
    required = [cfg.source_col, cfg.target_col, cfg.score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {df.columns.tolist()}"
        )

    out = df[[cfg.source_col, cfg.target_col, cfg.score_col]].copy()
    out.columns = ["source", "target", "score"]

    # Safe type coercion + NA handling
    out["source"] = out["source"].astype(str).str.strip()
    out["target"] = out["target"].astype(str).str.strip()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["source", "target", "score"])
    out = out[(out["source"] != "") & (out["target"] != "")]

    return out


def remove_self_loops_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["source"] != df["target"]].copy()

    # Canonical undirected edge key
    pair = np.sort(out[["source", "target"]].values, axis=1)
    out["u"] = pair[:, 0]
    out["v"] = pair[:, 1]

    # Keep maximum score among duplicates (same undirected edge)
    out = out.groupby(["u", "v"], as_index=False)["score"].max()
    out = out.rename(columns={"u": "source", "v": "target"})
    return out


def infer_threshold(scores: pd.Series) -> float:
    max_score = float(scores.max())
    if max_score <= 1.0:
        return 0.7
    return 700.0


def build_igraph(edges: pd.DataFrame, use_weights: bool = True) -> ig.Graph:
    vertices = pd.Index(edges["source"]).append(pd.Index(edges["target"])).unique().tolist()
    idx = {v: i for i, v in enumerate(vertices)}
    edge_tuples = [(idx[s], idx[t]) for s, t in zip(edges["source"], edges["target"])]

    g = ig.Graph(n=len(vertices), edges=edge_tuples, directed=False)
    g.vs["name"] = vertices
    if use_weights:
        g.es["weight"] = edges["score"].astype(float).tolist()
    else:
        g.es["weight"] = [1.0] * g.ecount()
    return g


def extract_lcc_graph(g: ig.Graph) -> ig.Graph:
    comps = g.components(mode="weak")
    giant = comps.giant()
    return giant


def graph_to_edge_df(g: ig.Graph) -> pd.DataFrame:
    rows = []
    for e in g.es:
        s, t = e.tuple
        rows.append((g.vs[s]["name"], g.vs[t]["name"], float(e["weight"])) )
    return pd.DataFrame(rows, columns=["source", "target", "score"])


def detect_communities(g: ig.Graph, use_weights: bool = True):
    w = g.es["weight"] if use_weights else None
    # Leiden preferred
    try:
        comm = g.community_leiden(weights=w, objective_function="modularity")
        method = "leiden"
    except Exception:
        comm = g.community_multilevel(weights=w)
        method = "louvain_multilevel"
    return comm, method


def within_module_z_and_participation(
    g: ig.Graph, membership: List[int], use_weights: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    n = g.vcount()
    w = np.array(g.es["weight"], dtype=float) if use_weights else np.ones(g.ecount(), dtype=float)

    # Build adjacency list with weights
    neigh = [[] for _ in range(n)]
    for eid, e in enumerate(g.es):
        u, v = e.tuple
        ww = w[eid]
        neigh[u].append((v, ww))
        neigh[v].append((u, ww))

    modules = np.array(membership)
    z = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)

    # Intra-module degree/strength per node
    k_intra = np.zeros(n, dtype=float)
    for i in range(n):
        mi = modules[i]
        k_intra[i] = sum(ww for j, ww in neigh[i] if modules[j] == mi)

    # z-score by module
    for m in np.unique(modules):
        idx = np.where(modules == m)[0]
        vals = k_intra[idx]
        mu = vals.mean() if len(vals) else 0.0
        sigma = vals.std(ddof=0) if len(vals) else 0.0
        if sigma == 0:
            z[idx] = 0.0
        else:
            z[idx] = (vals - mu) / sigma

    # Participation coefficient
    for i in range(n):
        total = sum(ww for _, ww in neigh[i])
        if total <= 0:
            p[i] = 0.0
            continue

        by_mod: Dict[int, float] = {}
        for j, ww in neigh[i]:
            mj = modules[j]
            by_mod[mj] = by_mod.get(mj, 0.0) + ww

        frac_sq_sum = sum((k_is / total) ** 2 for k_is in by_mod.values())
        p[i] = 1.0 - frac_sq_sum

    return z, p


def assign_role(z: float, p: float) -> str:
    if z < 2.5:
        if p <= 0.05:
            return "R1"
        if p <= 0.62:
            return "R2"
        if p <= 0.80:
            return "R3"
        return "R4"
    else:
        if p <= 0.30:
            return "R5"
        if p <= 0.75:
            return "R6"
        return "R7"


def mnc_dmnc_per_node(g: ig.Graph) -> Tuple[np.ndarray, np.ndarray]:
    n = g.vcount()
    mnc = np.zeros(n, dtype=float)
    dmnc = np.zeros(n, dtype=float)

    for v in range(n):
        nbrs = g.neighbors(v)
        if len(nbrs) == 0:
            continue

        sub = g.induced_subgraph(nbrs)
        comps = sub.components(mode="weak")
        if len(comps) == 0:
            continue
        giant = comps.giant()

        nn = giant.vcount()
        mm = giant.ecount()
        mnc[v] = float(nn)
        dmnc[v] = float(mm / (nn ** 1.7)) if nn > 0 else 0.0

    return mnc, dmnc


def mcc_per_node(g: ig.Graph) -> np.ndarray:
    n = g.vcount()
    out = np.zeros(n, dtype=float)
    cliques = g.maximal_cliques()

    for c in cliques:
        size = len(c)
        if size < 2:
            continue
        contrib = math.factorial(size - 1)
        for v in c:
            out[v] += contrib
    return out


def make_role_color_map() -> Dict[str, str]:
    return {
        "R1": "green",
        "R2": "violet",
        "R3": "blue",
        "R4": "orange",
        "R5": "red",
        "R6": "cyan",
        "R7": "gold",
    }


# %% [markdown]
# ## 3) Data loading + preprocessing

# %%
sns.set_theme(style="whitegrid")
out_dir = Path(CFG.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

raw = load_and_standardize(CFG)
raw_edges_n = len(raw)

clean = remove_self_loops_and_duplicates(raw)
threshold = infer_threshold(clean["score"])
filtered = clean[clean["score"] >= threshold].copy()
filtered_edges_n = len(filtered)

if filtered.empty:
    raise ValueError(
        f"No edges left after filtering with threshold={threshold}. Check your score scale or threshold settings."
    )

filtered.to_csv(out_dir / "filtered_edges.csv", index=False)
print(f"Raw edges: {raw_edges_n}")
print(f"Filtered edges: {filtered_edges_n} (threshold={threshold})")


# %% [markdown]
# ## 4) Graph construction + LCC extraction

# %%
g = build_igraph(filtered, use_weights=CFG.use_weights)
lcc = extract_lcc_graph(g)
lcc_edges = graph_to_edge_df(lcc)
lcc_edges.to_csv(out_dir / "lcc_edges.csv", index=False)

print(f"LCC nodes: {lcc.vcount()}")
print(f"LCC edges: {lcc.ecount()}")


# %% [markdown]
# ## 5) Community detection and cartography metrics

# %%
comm, comm_method = detect_communities(lcc, use_weights=CFG.use_weights)
membership = comm.membership
n_modules = len(set(membership))

z, p = within_module_z_and_participation(lcc, membership, use_weights=CFG.use_weights)
roles = [assign_role(zz, pp) for zz, pp in zip(z, p)]


# %% [markdown]
# ## 6) Centrality metrics (local + global)

# %%
# Local (cytoHubba-inspired)
mcc = mcc_per_node(lcc)
dmnc, mnc = None, None
mnc, dmnc = mnc_dmnc_per_node(lcc)

# Global
weights = lcc.es["weight"] if CFG.use_weights else None

degree = np.array(lcc.degree())
weighted_degree = np.array(lcc.strength(weights=weights))
betweenness = np.array(lcc.betweenness(weights=weights, directed=False))
closeness = np.array(lcc.closeness(weights=weights, normalized=True))

# For weighted eigenvector centrality, igraph uses strength-aware method if weights provided
eigen = np.array(lcc.eigenvector_centrality(weights=weights, directed=False, scale=True))
coreness = np.array(lcc.coreness(mode="all"))


# %% [markdown]
# ## 7) Final node-wise metrics table

# %%
node_df = pd.DataFrame(
    {
        "node_id": lcc.vs["name"],
        "degree": degree,
        "weighted_degree": weighted_degree,
        "community_id": membership,
        "within_module_z": z,
        "participation_coeff": p,
        "role": roles,
        "MCC": mcc,
        "DMNC": dmnc,
        "MNC": mnc,
        "eigenvector_centrality": eigen,
        "closeness_centrality": closeness,
        "betweenness_centrality": betweenness,
        "k_core": coreness,
    }
)

node_df = node_df.sort_values(["role", "within_module_z", "participation_coeff"], ascending=[True, False, False])
node_df.to_csv(out_dir / "node_metrics.csv", index=False)


# %% [markdown]
# ## 8) Plotting: 2D cartography

# %%
role_colors = make_role_color_map()
fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

# Background role regions
ax.axvspan(0.00, 0.05, ymin=0.0, ymax=(2.5 + 2) / (5 + 2), color="green", alpha=0.08)
ax.axvspan(0.05, 0.62, ymin=0.0, ymax=(2.5 + 2) / (5 + 2), color="violet", alpha=0.08)
ax.axvspan(0.62, 0.80, ymin=0.0, ymax=(2.5 + 2) / (5 + 2), color="blue", alpha=0.08)
ax.axvspan(0.80, 1.00, ymin=0.0, ymax=(2.5 + 2) / (5 + 2), color="orange", alpha=0.08)
ax.axvspan(0.00, 0.30, ymin=(2.5 + 2) / (5 + 2), ymax=1.0, color="red", alpha=0.08)
ax.axvspan(0.30, 0.75, ymin=(2.5 + 2) / (5 + 2), ymax=1.0, color="cyan", alpha=0.08)
ax.axvspan(0.75, 1.00, ymin=(2.5 + 2) / (5 + 2), ymax=1.0, color="gold", alpha=0.08)

for role, sub in node_df.groupby("role"):
    ax.scatter(
        sub["participation_coeff"],
        sub["within_module_z"],
        s=40,
        alpha=0.85,
        c=role_colors.get(role, "gray"),
        edgecolor="k",
        linewidth=0.3,
        label=role,
    )

# Guide lines
ax.axhline(2.5, color="black", linestyle="--", linewidth=1)
for xv in [0.05, 0.30, 0.62, 0.75, 0.80]:
    ax.axvline(xv, color="gray", linestyle="--", linewidth=0.8)

ax.set_xlim(-0.02, 1.02)
ymax = max(5, float(np.nanmax(node_df["within_module_z"])) + 0.5)
ax.set_ylim(-2, ymax)
ax.set_xlabel("Participation coefficient (P)")
ax.set_ylabel("Within-module degree z-score")
ax.set_title("Cartography of PPI network nodes")
ax.legend(title="Role", loc="upper right", ncol=2, frameon=True)

fig.savefig(out_dir / "cartography_2d.png", dpi=300, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 9) Plotting: 3D cartography (interactive + static PNG if possible)

# %%
z_metric = "degree"
fig3d = px.scatter_3d(
    node_df,
    x="participation_coeff",
    y="within_module_z",
    z=z_metric,
    color="role",
    color_discrete_map=role_colors,
    hover_data=["node_id", "community_id", "eigenvector_centrality"],
    title=f"3D cartography: P vs z vs {z_metric}",
)
fig3d.update_traces(marker=dict(size=4, opacity=0.9))
fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=35))

fig3d.write_html(str(out_dir / "cartography_3d.html"))

# Static image export requires kaleido
try:
    fig3d.write_image(str(out_dir / "cartography_3d.png"), width=1200, height=900, scale=2)
    print("Saved cartography_3d.png")
except Exception as e:
    print(f"Could not export static 3D PNG (likely missing kaleido): {e}")


# %% [markdown]
# ## 10) Local centrality overlap (Venn)

# %%
def top_nodes(metric_col: str, n: int) -> set:
    return set(node_df.nlargest(n, metric_col)["node_id"].tolist())


top_n = int(CFG.top_n_for_venn)
set_mcc = top_nodes("MCC", top_n)
set_dmnc = top_nodes("DMNC", top_n)
set_mnc = top_nodes("MNC", top_n)

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
venn3(
    [set_mcc, set_dmnc, set_mnc],
    set_labels=(f"MCC top{top_n}", f"DMNC top{top_n}", f"MNC top{top_n}"),
    ax=ax,
)
ax.set_title("Overlap of top-ranked local centrality nodes")
fig.savefig(out_dir / "local_centrality_venn.png", dpi=300, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 11) Export summary

# %%
role_counts = node_df["role"].value_counts().reindex([f"R{i}" for i in range(1, 8)], fill_value=0)

summary_lines = [
    "PPI ANALYSIS SUMMARY",
    "=====================",
    f"Input file: {CFG.input_file}",
    f"Confidence threshold used: {threshold}",
    f"Community method: {comm_method}",
    "",
    f"Raw edges: {raw_edges_n}",
    f"Filtered edges: {filtered_edges_n}",
    f"LCC nodes: {lcc.vcount()}",
    f"LCC edges: {lcc.ecount()}",
    f"Modules: {n_modules}",
    "",
    "Role counts:",
]
summary_lines.extend([f"  {r}: {int(c)}" for r, c in role_counts.items()])

summary_text = "\n".join(summary_lines)
(out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

print(summary_text)


# %% [markdown]
# ## 12) How to run (cell-by-cell)
#
# 1. Run the package installation cell (if needed).
# 2. Edit the `Config` values in the config cell (`input_file`, column mappings, `output_dir`, etc.).
# 3. Run all cells from top to bottom.
# 4. Collect outputs from `output_dir`:
#    - `filtered_edges.csv`
#    - `lcc_edges.csv`
#    - `node_metrics.csv`
#    - `cartography_2d.png`
#    - `cartography_3d.html`
#    - `cartography_3d.png` (if kaleido available)
#    - `local_centrality_venn.png`
#    - `summary.txt`
#
# ---
#
# ### Troubleshooting notes
#
# - **No edges after filtering**: your confidence column may be on an unexpected scale; inspect min/max values.
# - **Missing column error**: update `source_col`, `target_col`, and `score_col` in config.
# - **3D PNG not saved**: install `kaleido` (`pip install kaleido`) or use the HTML output.
# - **Large graph performance**: MCC via maximal cliques can be expensive; for huge networks, consider restricting to LCC only (already done here) or sampling for exploratory analysis.
