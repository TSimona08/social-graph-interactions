import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "analysis" / "bonus_turing" / "output"
OUT = REPO_ROOT / "docs" / "assets" / "img" / "bonus-turing"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

data = json.load(open(DATA_DIR / "turing_network.json"))
G = nx.DiGraph()
G.add_nodes_from(data["nodes"])
G.add_edges_from(data["edges"])

n, m = G.number_of_nodes(), G.number_of_edges()
isolates = list(nx.isolates(G))
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
in_vals = np.array(list(in_deg.values()))
out_vals = np.array(list(out_deg.values()))

print(f"n={n} m={m} isolates={len(isolates)}")
print("isolates:", isolates)
print("mean in-deg:", in_vals.mean(), "max in-deg:", in_vals.max())
print("mean out-deg:", out_vals.mean(), "max out-deg:", out_vals.max())

top_in = sorted(in_deg.items(), key=lambda kv: -kv[1])[:10]
top_out = sorted(out_deg.items(), key=lambda kv: -kv[1])[:10]
print("\nTop in-degree:")
for name, d in top_in: print(f"  {name:30s} {d}")
print("\nTop out-degree:")
for name, d in top_out: print(f"  {name:30s} {d}")

UG = G.to_undirected()
comps = sorted(nx.connected_components(UG), key=len, reverse=True)
print(f"\n{len(comps)} components. Giant: {len(comps[0])} ({len(comps[0])/n*100:.1f}%)")
for c in comps[1:]:
    if len(c) > 1:
        print("  extra component:", c)

# --- degree histograms ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
bins = np.arange(0, max(in_vals.max(), out_vals.max()) + 2) - 0.5
axes[0].hist(in_vals, bins=bins, color="#2b3ea8", edgecolor="white")
axes[0].set_title("In-degree"); axes[0].set_xlabel("in-degree"); axes[0].set_ylabel("count")
axes[1].hist(out_vals, bins=bins, color="#1c8a5a", edgecolor="white")
axes[1].set_title("Out-degree"); axes[1].set_xlabel("out-degree")
fig.tight_layout()
fig.savefig(OUT / "degree_hist.png", bbox_inches="tight")
plt.close(fig)

# --- network drawing ---
# Three explicit zones, laid out on a fixed canvas so nothing overlaps or
# runs off-frame: giant component fills a wide central/right region, small
# islands are stacked in individually-bounded boxes on the left, isolates
# sit in a grid at the bottom left.
fig, ax = plt.subplots(figsize=(13, 9))

CANVAS_XMIN, CANVAS_XMAX = -7.5, 7.5
CANVAS_YMIN, CANVAS_YMAX = -5.3, 4.8

giant = comps[0]
giant_sub = G.subgraph(giant)
gpos = nx.spring_layout(giant_sub, seed=7, k=2.5 / np.sqrt(len(giant)), iterations=400)
gx = np.array([p[0] for p in gpos.values()])
gy = np.array([p[1] for p in gpos.values()])
gx = (gx - gx.mean()) / (gx.max() - gx.min())  # normalize to [-0.5, 0.5]
gy = (gy - gy.mean()) / (gy.max() - gy.min())
# giant component occupies the right ~2/3 of the canvas
pos = {k: (0.5 + gx[i] * 7.5, gy[i] * 9.0) for i, k in enumerate(gpos.keys())}

small_comps = [c for c in comps[1:] if len(c) > 1]
box_h = 8.6 / max(len(small_comps), 1)
for i, comp in enumerate(small_comps):
    sub = G.subgraph(comp)
    sp = nx.spring_layout(sub, seed=3, k=0.8, iterations=100)
    sx = np.array([p[0] for p in sp.values()])
    sy = np.array([p[1] for p in sp.values()])
    span_x = max(sx.max() - sx.min(), 1e-6)
    span_y = max(sy.max() - sy.min(), 1e-6)
    cy_top = 4.3 - i * box_h
    for k, (x, y) in sp.items():
        nx_ = (x - sx.mean()) / span_x * 1.1 - 6.2
        ny_ = (y - sy.mean()) / span_y * (box_h * 0.6) + cy_top - box_h / 2
        pos[k] = (nx_, ny_)

n_iso_cols = 4
for i, nid in enumerate(isolates):
    row, col = divmod(i, n_iso_cols)
    pos[nid] = (-7.1 + col * 0.55, -1.6 - row * 0.5)

small_comp_nodes = {nid for c in small_comps for nid in c}
node_colors = []
for nid in G.nodes():
    if nid in isolates:
        node_colors.append("#bbbbbb")
    elif nid in small_comp_nodes:
        node_colors.append("#1c8a5a")
    else:
        node_colors.append("#b3151a")
node_sizes = [40 + (in_deg[nid] + out_deg[nid]) * 40 for nid in G.nodes()]

nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#ccc", width=0.6, alpha=0.6,
                        arrows=True, arrowsize=6, connectionstyle="arc3,rad=0.05")
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                        linewidths=0.5, edgecolors="white")

top_hubs = [nid for nid, _ in sorted(
    {k: in_deg[k] + out_deg[k] for k in G.nodes()}.items(), key=lambda kv: -kv[1]
)[:8]]
for nid in top_hubs:
    x, y = pos[nid]
    ax.annotate(nid, (x, y), fontsize=8.5, fontweight="bold",
                xytext=(6, 6), textcoords="offset points")

ax.annotate("deep learning\npioneers", (-6.2, 4.3), fontsize=8.5, color="#1c8a5a",
            fontweight="bold", ha="center", va="bottom")
ax.annotate("isolated\n(no in-category links)", (-6.85, -1.3), fontsize=8.5,
            color="#888888", fontweight="bold", ha="left", va="bottom")

ax.set_title(
    f"Turing Award laureates — Wikipedia link network\n"
    f"{n} laureates, {m} links · red = giant component ({len(giant)}) · "
    f"green = other islands · gray = isolated",
    fontsize=12,
)
ax.axis("off")
ax.set_xlim(CANVAS_XMIN, CANVAS_XMAX)
ax.set_ylim(CANVAS_YMIN, CANVAS_YMAX)
fig.tight_layout()
fig.savefig(OUT / "network.png", bbox_inches="tight")
plt.close(fig)

summary = {
    "n_nodes": n, "n_edges": m, "n_isolates": len(isolates), "isolates": isolates,
    "n_components": len(comps), "giant_size": len(comps[0]),
    "top_in": top_in, "top_out": top_out,
    "mean_in": float(in_vals.mean()), "mean_out": float(out_vals.mean()),
    "max_in": int(in_vals.max()), "max_out": int(out_vals.max()),
}
json.dump(summary, open(DATA_DIR / "summary.json", "w"), indent=2)
print("\nDone.")
