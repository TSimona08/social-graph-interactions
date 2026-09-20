"""
Week 1 'bootstrap' go-nuts analysis of the Marvel superheroes network
(02805 shared playground, week-1 frozen snapshot).

Loads week1_edges.tsv + week1_nodes.tsv, builds a directed graph that
keeps the isolated nodes, and produces:
  - basic stats (nodes, edges, isolates, components)
  - in/out degree distributions (linear + log-log)
  - top nodes by in-degree and out-degree
  - a full network drawing, colored by component / isolate status
  - a table of isolated nodes and of components outside the giant one
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import json
from pathlib import Path

# Repo layout: this script lives in analysis/, data lives at repo root,
# figures are written straight into the site so they're ready to publish.
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = str(REPO_ROOT)
OUT = str(REPO_ROOT / "docs" / "assets" / "img" / "week1")
Path(OUT).mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- load ---
# NB: the header row itself is prefixed with "# " (e.g. "# source\ttarget"),
# so comment="#" would eat it along with the real comment lines. Skip the
# leading comment block manually and name columns explicitly instead.
def count_leading_comments(path):
    with open(path) as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return skip

# nodes.tsv: real header row is NOT "#"-prefixed -> skip comments, keep header
nodes = pd.read_csv(f"{BASE}/week1_nodes.tsv", sep="\t", skiprows=count_leading_comments(f"{BASE}/week1_nodes.tsv"))

# edges.tsv: real header row IS "#"-prefixed ("# source\ttarget") -> skip
# comments *and* that header line, and name columns explicitly.
edges_skip = count_leading_comments(f"{BASE}/week1_edges.tsv")
edges = pd.read_csv(f"{BASE}/week1_edges.tsv", sep="\t", skiprows=edges_skip, names=["source", "target"])

print("nodes columns:", list(nodes.columns))
print("edges columns:", list(edges.columns))
print("n nodes (roster):", len(nodes))
print("n edges:", len(edges))

G = nx.DiGraph()
G.add_nodes_from(nodes["node_id"])
G.add_edges_from(zip(edges["source"], edges["target"]))

name_of = dict(zip(nodes["node_id"], nodes["name"]))

n = G.number_of_nodes()
m = G.number_of_edges()
isolates = list(nx.isolates(G))
print(f"\nG: {n} nodes, {m} edges, {len(isolates)} isolates")

# --- degree distributions ---
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

in_vals = np.array(list(in_deg.values()))
out_vals = np.array(list(out_deg.values()))

print("\nin-degree: mean=%.2f max=%d" % (in_vals.mean(), in_vals.max()))
print("out-degree: mean=%.2f max=%d" % (out_vals.mean(), out_vals.max()))

top_in = sorted(in_deg.items(), key=lambda kv: -kv[1])[:10]
top_out = sorted(out_deg.items(), key=lambda kv: -kv[1])[:10]
print("\nTop 10 in-degree (most linked-to):")
for nid, d in top_in:
    print(f"  {name_of[nid]:35s} {d}")
print("\nTop 10 out-degree (link out the most):")
for nid, d in top_out:
    print(f"  {name_of[nid]:35s} {d}")

# --- linear degree histograms ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
bins = np.arange(0, max(in_vals.max(), out_vals.max()) + 2) - 0.5
axes[0].hist(in_vals, bins=bins, color="#d6493a", edgecolor="white")
axes[0].set_title("In-degree distribution")
axes[0].set_xlabel("in-degree")
axes[0].set_ylabel("number of characters")
axes[1].hist(out_vals, bins=bins, color="#3a6fd6", edgecolor="white")
axes[1].set_title("Out-degree distribution")
axes[1].set_xlabel("out-degree")
fig.tight_layout()
fig.savefig(f"{OUT}/degree_hist_linear.png", bbox_inches="tight")
plt.close(fig)

# --- log-log degree distributions (using log-binned or rank-based CCDF style) ---
def loglog_pdf_ax(ax, vals, color, label):
    vals = vals[vals > 0]
    counts = np.bincount(vals)
    degs = np.nonzero(counts)[0]
    freqs = counts[degs] / counts[degs].sum()
    ax.scatter(degs, freqs, color=color, s=30, label=label, alpha=0.85, edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree (log)")
    ax.set_ylabel("P(degree) (log)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
loglog_pdf_ax(axes[0], in_vals, "#d6493a", "in-degree")
axes[0].set_title("In-degree, log–log")
loglog_pdf_ax(axes[1], out_vals, "#3a6fd6", "out-degree")
axes[1].set_title("Out-degree, log–log")
fig.tight_layout()
fig.savefig(f"{OUT}/degree_hist_loglog.png", bbox_inches="tight")
plt.close(fig)

# --- components ---
UG = G.to_undirected()
components = list(nx.connected_components(UG))
components.sort(key=len, reverse=True)
giant = components[0]
others = components[1:]

print(f"\n{len(components)} connected components (undirected).")
print(f"Giant component: {len(giant)} nodes ({len(giant)/n*100:.1f}% of network)")
print(f"Other components (excl. isolates): ")
for c in others:
    if len(c) > 1:
        print(f"  size {len(c)}: {[name_of[x] for x in c]}")

print(f"\nIsolated nodes ({len(isolates)}):")
for nid in isolates:
    print(f"  {name_of[nid]}")

# --- SCC / WCC on directed graph too ---
n_scc = nx.number_strongly_connected_components(G)
n_wcc = nx.number_weakly_connected_components(G)
giant_scc = max(nx.strongly_connected_components(G), key=len)
print(f"\nStrongly connected components: {n_scc}, giant SCC size: {len(giant_scc)}")
print(f"Weakly connected components: {n_wcc}")

# --- network drawing ---
# Build the layout by hand in three zones so the picture stays legible:
#   1) giant component -> spring layout, filling a central disk
#   2) small non-trivial components -> tiny spring layouts, parked to the side
#   3) isolates -> tidy grid along the bottom (they have no edges to be
#      pulled anywhere meaningful, so scattering them via spring_layout
#      just wastes canvas)
fig, ax = plt.subplots(figsize=(12, 9))

pos = {}

giant_sub = G.subgraph(giant)
giant_pos = nx.spring_layout(giant_sub, seed=42, k=1.6 / np.sqrt(len(giant)), iterations=200)
# scale giant component to roughly fill a radius-1 disk centered at origin
gx = np.array([p[0] for p in giant_pos.values()])
gy = np.array([p[1] for p in giant_pos.values()])
scale = 1.0 / max(gx.std(), gy.std())
for k, (x, y) in giant_pos.items():
    pos[k] = (x * scale, y * scale + 0.3)

# small components: stack them to the upper-left of the main blob
small_comps = [c for c in others if len(c) > 1]
for i, comp in enumerate(small_comps):
    sub = G.subgraph(comp)
    sp = nx.spring_layout(sub, seed=1, k=0.8)
    cx, cy = -3.2, 1.0 - i * 1.6
    for k, (x, y) in sp.items():
        pos[k] = (x * 0.5 + cx, y * 0.5 + cy)

# isolates: neat grid along the bottom
n_per_row = 9
for i, nid in enumerate(isolates):
    row, col = divmod(i, n_per_row)
    pos[nid] = (-3.2 + col * 0.85, -2.6 - row * 0.55)

node_colors = []
node_sizes = []
for nid in G.nodes():
    tot_deg = in_deg[nid] + out_deg[nid]
    if nid in isolates:
        node_colors.append("#bbbbbb")
    elif nid in giant:
        node_colors.append("#d6493a")
    else:
        node_colors.append("#3a9e6f")
    node_sizes.append(15 + tot_deg * 9)

nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=0.4, alpha=0.5, arrows=False)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=0.5, edgecolors="white")

# label only the top-degree hubs, nudged so they don't stack on each other
top_hub_ids = [nid for nid, _ in sorted(
    {k: in_deg[k] + out_deg[k] for k in G.nodes()}.items(), key=lambda kv: -kv[1]
)[:8]]
manual_offsets = {
    0: (6, 6), 1: (6, 6), 2: (6, -14), 3: (-70, 6),
    4: (6, -14), 5: (-60, -14), 6: (6, 14), 7: (-70, -6),
}
for i, nid in enumerate(top_hub_ids):
    x, y = pos[nid]
    dx, dy = manual_offsets.get(i, (6, 6))
    ax.annotate(name_of[nid].split(" (")[0], (x, y), fontsize=9, fontweight="bold",
                xytext=(dx, dy), textcoords="offset points")

ax.annotate("small islands", (-3.2, 1.9), fontsize=9, color="#3a9e6f", ha="center", fontweight="bold")
ax.annotate("isolated nodes (no links either way)", (-3.2 + 4*0.85, -2.35), fontsize=9, color="#888888", ha="center", fontweight="bold")

ax.set_title("The Marvel superheroes web (Wikipedia links, week-1 snapshot)\n"
             "red = giant component (277 nodes) · green = other islands · gray = isolated · size = total degree", fontsize=12)
ax.axis("off")
ax.set_xlim(-4.2, 2.5)
fig.tight_layout()
fig.savefig(f"{OUT}/network_full.png", bbox_inches="tight")
plt.close(fig)

# --- save summary json for the write-up ---
summary = {
    "n_nodes": n,
    "n_edges": m,
    "n_isolates": len(isolates),
    "isolate_names": [name_of[x] for x in isolates],
    "n_components_undirected": len(components),
    "giant_component_size": len(giant),
    "other_components": [[name_of[x] for x in c] for c in others if len(c) > 1],
    "n_scc": n_scc,
    "giant_scc_size": len(giant_scc),
    "n_wcc": n_wcc,
    "top_in_degree": [[name_of[nid], d] for nid, d in top_in],
    "top_out_degree": [[name_of[nid], d] for nid, d in top_out],
    "mean_in_degree": float(in_vals.mean()),
    "mean_out_degree": float(out_vals.mean()),
    "max_in_degree": int(in_vals.max()),
    "max_out_degree": int(out_vals.max()),
}
summary_dir = REPO_ROOT / "analysis" / "output"
summary_dir.mkdir(parents=True, exist_ok=True)
with open(summary_dir / "week1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Figures written to", OUT)
print("Summary written to", summary_dir / "week1_summary.json")
