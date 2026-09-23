"""
Week 3b — The Web Explorer.

Interactive tool: select a character, see their largest clique highlighted.
Select a second character, see the shortest path between them, with every
path-node's own largest clique also highlighted (each in its own color).

Exports:
  - week3b_network.json: same layout as week1 (nodes/edges/positions),
    plus each node's largest clique (as a sorted tuple of member ids,
    used as a stable clique "key" for coloring) and clique size.
  - week3b_cliques.json: clique_key -> list of member names (for legend /
    lookup), keyed by a stable string so the same clique always gets the
    same color across selections.
"""
import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_OUT = REPO_ROOT / "docs" / "assets" / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)


def count_leading_comments(path):
    with open(path) as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return skip


nodes = pd.read_csv(REPO_ROOT / "week1_nodes.tsv", sep="\t",
                     skiprows=count_leading_comments(REPO_ROOT / "week1_nodes.tsv"))
edges = pd.read_csv(REPO_ROOT / "week1_edges.tsv", sep="\t",
                     skiprows=count_leading_comments(REPO_ROOT / "week1_edges.tsv"),
                     names=["source", "target"])

G = nx.DiGraph()
G.add_nodes_from(nodes["node_id"])
G.add_edges_from(zip(edges["source"], edges["target"]))
name_of = dict(zip(nodes["node_id"], nodes["name"]))

UG = G.to_undirected()

# --- reuse the exact week-1 layout so this tool visually matches week 1 ---
isolates = list(nx.isolates(G))
components = sorted(nx.connected_components(UG), key=len, reverse=True)
giant = components[0]
others = components[1:]
small_comps = [c for c in others if len(c) > 1]

pos = {}
giant_sub = G.subgraph(giant)
giant_pos = nx.spring_layout(giant_sub, seed=42, k=1.6 / np.sqrt(len(giant)), iterations=200)
gx = np.array([p[0] for p in giant_pos.values()])
gy = np.array([p[1] for p in giant_pos.values()])
scale = 1.0 / max(gx.std(), gy.std())
for k, (x, y) in giant_pos.items():
    pos[k] = (x * scale, y * scale + 0.3)

for i, comp in enumerate(small_comps):
    sub = G.subgraph(comp)
    sp = nx.spring_layout(sub, seed=1, k=0.8)
    cx, cy = -3.2, 1.0 - i * 1.6
    for k, (x, y) in sp.items():
        pos[k] = (x * 0.5 + cx, y * 0.5 + cy)

n_per_row = 9
for i, nid in enumerate(isolates):
    row, col = divmod(i, n_per_row)
    pos[nid] = (-3.2 + col * 0.85, -2.6 - row * 0.55)

in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())


def role_of(nid):
    if nid in isolates:
        return "isolate"
    if nid in giant:
        return "giant"
    return "island"


# --- cliques: find each node's single largest maximal clique ---
print("Finding maximal cliques...")
cliques = list(nx.find_cliques(UG))
print(f"{len(cliques)} maximal cliques found.")

# For each node, find the largest clique it belongs to. Ties broken by
# picking the clique whose sorted member-id tuple is lexicographically
# smallest, so the choice is stable/reproducible.
best_clique_for = {}  # node_id -> clique tuple (sorted node ids)
for c in cliques:
    key = tuple(sorted(c))
    for n in c:
        cur = best_clique_for.get(n)
        if cur is None or len(key) > len(cur) or (len(key) == len(cur) and key < cur):
            best_clique_for[n] = key

# assign a stable color-index per distinct clique (by order of first
# appearance sorted by size desc, so bigger/more "important" cliques get
# lower indices - purely for nicer default coloring in a categorical scale)
distinct_cliques = sorted(set(best_clique_for.values()), key=lambda c: (-len(c), c))
clique_index = {c: i for i, c in enumerate(distinct_cliques)}

print(f"{len(distinct_cliques)} distinct 'largest cliques' assigned across {len(best_clique_for)} nodes.")

# --- export network with clique annotations ---
network_data = {
    "nodes": [
        {
            "id": nid,
            "name": name_of[nid],
            "x": round(pos[nid][0], 4),
            "y": round(pos[nid][1], 4),
            "in": in_deg[nid],
            "out": out_deg[nid],
            "role": role_of(nid),
            "clique": clique_index[best_clique_for[nid]],
            "cliqueSize": len(best_clique_for[nid]),
        }
        for nid in G.nodes()
    ],
    "edges": [{"source": u, "target": v} for u, v in G.edges()],
}
with open(DATA_OUT / "week3b_network.json", "w") as f:
    json.dump(network_data, f, separators=(",", ":"))

# --- export clique membership lookup (index -> member names) ---
clique_lookup = {
    str(idx): [name_of[n] for n in members]
    for members, idx in clique_index.items()
}
with open(DATA_OUT / "week3b_cliques.json", "w") as f:
    json.dump(clique_lookup, f, separators=(",", ":"))

print("Done. Wrote week3b_network.json and week3b_cliques.json")

# quick sanity print
print("\nSpider-Man's largest clique:", clique_lookup[str(network_data['nodes'][
    [n['id'] for n in network_data['nodes']].index('Spider-Man')
]['clique'])])
