"""
Week 4 — communities, weights and the philosophers.

02805 shared playground, week-4 philosophers snapshot: 1444 philosophers on
English Wikipedia born before 1900, 11136 directed links between their
articles, each with a weight (repeat links = a stronger tie).

This script asks one question: does WEIGHTING the network change the
communities Louvain finds? It builds the undirected graph two ways —
edges collapsed to unweighted (just "linked or not") and edges summed into
a weight (how many times) — runs Louvain on each, compares them with
normalized mutual information, and finds the philosophers whose community
assignment flips between the two.

Outputs:
  - console stats (NMI, sizes, movers)
  - docs/assets/img/week4/communities_unweighted.png
  - docs/assets/img/week4/communities_weighted.png
  - docs/assets/img/week4/movers.png
  - docs/assets/data/week4_network.json   (layout + both community labels)
  - docs/assets/data/week4_movers.json    (philosophers who moved, with why)
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import json
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMG_OUT = REPO_ROOT / "docs" / "assets" / "img" / "week4"
DATA_OUT = REPO_ROOT / "docs" / "assets" / "data"
IMG_OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RNG_SEED = 42


def count_leading_comments(path):
    with open(path) as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return skip


# --- load ---
nodes_path = REPO_ROOT / "week4_philosophers_nodes.tsv"
edges_path = REPO_ROOT / "week4_philosophers_edges.tsv"

nodes = pd.read_csv(nodes_path, sep="\t", skiprows=count_leading_comments(nodes_path))
edges = pd.read_csv(edges_path, sep="\t", skiprows=count_leading_comments(edges_path) + 1,
                     names=["source", "target", "weight"])
edges["weight"] = edges["weight"].astype(int)

print("nodes:", len(nodes), "edges (directed):", len(edges))

name_of = dict(zip(nodes["node_id"], nodes["name"]))
era_of = dict(zip(nodes["node_id"], nodes["era"]))
subfields_of = dict(zip(nodes["node_id"], nodes["subfields"]))
desc_of = dict(zip(nodes["node_id"], nodes["description"]))

# --- build undirected graphs: unweighted (binary) vs weighted (summed) ---
# Sum both directions' weight into one undirected tie, per the data's own
# README note ("sum both directions for an undirected weighted network").
UW = nx.Graph()   # unweighted: edge exists or it doesn't
W = nx.Graph()    # weighted: edge weight = sum of both directions' link counts
UW.add_nodes_from(nodes["node_id"])
W.add_nodes_from(nodes["node_id"])

for _, row in edges.iterrows():
    a, b, wt = row["source"], row["target"], row["weight"]
    if a == b:
        continue
    UW.add_edge(a, b)
    if W.has_edge(a, b):
        W[a][b]["weight"] += wt
    else:
        W.add_edge(a, b, weight=wt)

print("undirected edges:", UW.number_of_edges())

# --- restrict to the giant component (communities on isolates are noise) ---
components = sorted(nx.connected_components(UW), key=len, reverse=True)
giant_nodes = components[0]
print("components:", len(components), "giant component size:", len(giant_nodes))
print("next 5 component sizes:", [len(c) for c in components[1:6]])

UWg = UW.subgraph(giant_nodes).copy()
Wg = W.subgraph(giant_nodes).copy()

# --- Louvain: unweighted vs weighted ---
comms_uw = nx.community.louvain_communities(UWg, weight=None, seed=RNG_SEED)
comms_w = nx.community.louvain_communities(Wg, weight="weight", seed=RNG_SEED)

print(f"\nunweighted Louvain: {len(comms_uw)} communities, sizes {sorted((len(c) for c in comms_uw), reverse=True)}")
print(f"weighted Louvain:   {len(comms_w)} communities, sizes {sorted((len(c) for c in comms_w), reverse=True)}")

label_uw = {}
for i, c in enumerate(comms_uw):
    for n in c:
        label_uw[n] = i
label_w = {}
for i, c in enumerate(comms_w):
    for n in c:
        label_w[n] = i

ordered_nodes = list(giant_nodes)
labels_uw = [label_uw[n] for n in ordered_nodes]
labels_w = [label_w[n] for n in ordered_nodes]
nmi = normalized_mutual_info_score(labels_uw, labels_w)
print(f"\nNMI(unweighted, weighted) = {nmi:.4f}")

# --- era composition of each weighted community (sanity check: does it track known history?) ---
print("\nWeighted communities by dominant era:")
for i, c in enumerate(sorted(comms_w, key=len, reverse=True)):
    eras = Counter(era_of.get(n, "?") for n in c)
    top_era, top_n = eras.most_common(1)[0]
    print(f"  community {i}: n={len(c):4d}  dominant era: {top_era} ({top_n}/{len(c)})")

# --- movers: philosophers whose community changes composition between the two runs ---
# Build a contingency mapping: for each weighted community, which unweighted
# community contributes the most members (the "parent"). A node "moves" if
# it's NOT in its weighted community's parent unweighted community.
comms_w_sorted = sorted(comms_w, key=len, reverse=True)
comms_uw_sorted = sorted(comms_uw, key=len, reverse=True)

label_w_sorted = {}
for wi, c in enumerate(comms_w_sorted):
    for n in c:
        label_w_sorted[n] = wi
label_uw_sorted = {}
for ui, c in enumerate(comms_uw_sorted):
    for n in c:
        label_uw_sorted[n] = ui

parent_of_w = {}  # weighted community index -> majority unweighted community index (both in sorted numbering)
for wi, c in enumerate(comms_w_sorted):
    cnt = Counter(label_uw_sorted[n] for n in c)
    parent_of_w[wi] = cnt.most_common(1)[0][0]

movers = []
for n in ordered_nodes:
    wi = label_w_sorted[n]
    ui = label_uw_sorted[n]
    if ui != parent_of_w[wi]:
        movers.append(n)

print(f"\nMovers (community changes between unweighted and weighted): {len(movers)} / {len(ordered_nodes)} ({100*len(movers)/len(ordered_nodes):.1f}%)")

# rank movers by the weight they carry on their strongest edge (the tie
# heavy enough to plausibly have pulled them into a different community)
mover_info = []
for n in movers:
    nbrs = list(Wg.neighbors(n)) if n in Wg else []
    if not nbrs:
        continue
    strongest = max(nbrs, key=lambda m: Wg[n][m]["weight"])
    mover_info.append({
        "node_id": n,
        "name": name_of.get(n, n),
        "era": era_of.get(n, ""),
        "unweighted_community": int(label_uw_sorted[n]),
        "weighted_community": int(label_w_sorted[n]),
        "strongest_tie": name_of.get(strongest, strongest),
        "strongest_tie_weight": int(Wg[n][strongest]["weight"]),
    })
mover_info.sort(key=lambda d: -d["strongest_tie_weight"])

print("\nTop 10 movers by strongest single tie weight:")
for m in mover_info[:10]:
    print(f"  {m['name']:35s} uw#{m['unweighted_community']:2d} -> w#{m['weighted_community']:2d}  "
          f"(strongest tie: {m['strongest_tie']}, weight {m['strongest_tie_weight']})")

# --- layout (spring layout on the weighted giant component, fixed seed) ---
print("\ncomputing layout...")
pos = nx.spring_layout(Wg, weight="weight", k=1.2 / np.sqrt(len(Wg)), seed=RNG_SEED, iterations=80)

# --- static figures: side-by-side unweighted vs weighted communities ---
def community_colors(labels_sorted_map, n_comms):
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(n_comms, 1))
    return {i: cmap(i % 20) for i in range(n_comms)}

colors_uw = community_colors(label_uw_sorted, len(comms_uw_sorted))
colors_w = community_colors(label_w_sorted, len(comms_w_sorted))


def draw_communities(G, label_map, colors, title, path, mover_set=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    node_colors = [colors[label_map[n]] for n in G.nodes()]
    sizes = [70 if (mover_set and n in mover_set) else 16 for n in G.nodes()]
    edgecolors = ["#2b2420" if (mover_set and n in mover_set) else "none" for n in G.nodes()]
    linewidths = [1.1 if (mover_set and n in mover_set) else 0 for n in G.nodes()]

    # edges (very faint)
    for a, b in G.edges():
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#c9bfa8", linewidth=0.25, alpha=0.35, zorder=1)

    ax.scatter(xs, ys, c=node_colors, s=sizes, edgecolors=edgecolors, linewidths=linewidths, zorder=2)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


mover_set = set(movers)
draw_communities(UWg, label_uw_sorted, colors_uw, f"Unweighted Louvain — {len(comms_uw_sorted)} communities",
                  IMG_OUT / "communities_unweighted.png")
draw_communities(Wg, label_w_sorted, colors_w, f"Weighted Louvain — {len(comms_w_sorted)} communities",
                  IMG_OUT / "communities_weighted.png")
draw_communities(Wg, label_w_sorted, colors_w,
                  f"{len(movers)} philosophers who moved communities (outlined)",
                  IMG_OUT / "movers.png", mover_set=mover_set)

# --- JSON export for the interactive explorer ---
network_export = {
    "meta": {
        "n_nodes": len(ordered_nodes),
        "n_edges": Wg.number_of_edges(),
        "n_communities_unweighted": len(comms_uw_sorted),
        "n_communities_weighted": len(comms_w_sorted),
        "nmi": round(float(nmi), 4),
        "n_movers": len(movers),
    },
    "nodes": [],
    "edges": [],
}

for n in ordered_nodes:
    x, y = pos[n]
    network_export["nodes"].append({
        "id": n,
        "name": name_of.get(n, n),
        "era": era_of.get(n, ""),
        "subfields": subfields_of.get(n, "") if pd.notna(subfields_of.get(n, "")) else "",
        "description": (desc_of.get(n, "") or "")[:220],
        "x": round(float(x), 5),
        "y": round(float(y), 5),
        "degree": int(Wg.degree(n)),
        "comm_uw": int(label_uw_sorted[n]),
        "comm_w": int(label_w_sorted[n]),
        "moved": n in mover_set,
    })

for a, b, data in Wg.edges(data=True):
    network_export["edges"].append({"source": a, "target": b, "weight": int(data["weight"])})

with open(DATA_OUT / "week4_network.json", "w") as f:
    json.dump(network_export, f)
print("wrote", DATA_OUT / "week4_network.json")

with open(DATA_OUT / "week4_movers.json", "w") as f:
    json.dump(mover_info, f, indent=2)
print("wrote", DATA_OUT / "week4_movers.json")

# --- summary json for quick reference while writing the post ---
summary = {
    "n_nodes_total": len(nodes),
    "n_nodes_giant": len(ordered_nodes),
    "n_components": len(components),
    "n_communities_unweighted": len(comms_uw_sorted),
    "n_communities_weighted": len(comms_w_sorted),
    "nmi": round(float(nmi), 4),
    "n_movers": len(movers),
    "pct_movers": round(100 * len(movers) / len(ordered_nodes), 2),
    "top_movers": mover_info[:15],
    "weighted_community_sizes": [len(c) for c in comms_w_sorted],
    "unweighted_community_sizes": [len(c) for c in comms_uw_sorted],
}
with open(REPO_ROOT / "analysis" / "week4" / "output_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("wrote analysis/week4/output_summary.json")

print("\ndone.")
