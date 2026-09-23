"""
Week 3 — Six degrees of Spider-Man.

For every character in the week-1 snapshot, find the shortest path to
Spider-Man (undirected, since "linked to" and "linked from" both count
as a real-world connection for this game). Report:
  - the distribution of distances
  - the longest chains, and who's in them
  - characters with NO path at all (different components / isolates)
  - export a distance+path lookup table for the interactive tool
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = REPO_ROOT
FIG_OUT = REPO_ROOT / "docs" / "assets" / "img" / "week3"
DATA_OUT = REPO_ROOT / "docs" / "assets" / "data"
SUMMARY_OUT = REPO_ROOT / "analysis" / "week3" / "output"
FIG_OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)
SUMMARY_OUT.mkdir(parents=True, exist_ok=True)
OUT = FIG_OUT  # figures go straight into the site

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def count_leading_comments(path):
    with open(path) as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return skip


nodes = pd.read_csv(BASE / "week1_nodes.tsv", sep="\t", skiprows=count_leading_comments(BASE / "week1_nodes.tsv"))
edges = pd.read_csv(BASE / "week1_edges.tsv", sep="\t",
                     skiprows=count_leading_comments(BASE / "week1_edges.tsv"),
                     names=["source", "target"])

G = nx.DiGraph()
G.add_nodes_from(nodes["node_id"])
G.add_edges_from(zip(edges["source"], edges["target"]))
name_of = dict(zip(nodes["node_id"], nodes["name"]))
id_of_name = {v: k for k, v in name_of.items()}

UG = G.to_undirected()

SPIDEY = "Spider-Man"
assert SPIDEY in UG.nodes

# BFS shortest path lengths from Spider-Man over the undirected graph
lengths = nx.single_source_shortest_path_length(UG, SPIDEY)
paths = nx.single_source_shortest_path(UG, SPIDEY)

n = UG.number_of_nodes()
reachable = set(lengths.keys())
unreachable = set(UG.nodes) - reachable

print(f"Total nodes: {n}")
print(f"Reachable from Spider-Man: {len(reachable)} ({len(reachable)/n*100:.1f}%)")
print(f"Unreachable: {len(unreachable)}")

dist_counts = pd.Series(list(lengths.values())).value_counts().sort_index()
print("\nDistance distribution (hops from Spider-Man):")
for d, c in dist_counts.items():
    print(f"  {d} hops: {c} characters")

max_d = dist_counts.index.max()
farthest = [nid for nid, d in lengths.items() if d == max_d]
print(f"\nFarthest distance: {max_d} hops. Characters at that distance:")
for nid in farthest:
    path = paths[nid]
    path_names = [name_of[p] for p in path]
    print(f"  {name_of[nid]}: {' -> '.join(path_names)}")

# --- degree distribution plot ---
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(dist_counts.index.astype(str), dist_counts.values, color="#ff3b3b", edgecolor="#3a0d10")
ax.set_xlabel("hops from Spider-Man")
ax.set_ylabel("number of characters")
ax.set_title("Six degrees of Spider-Man: distance distribution")
for i, (d, c) in enumerate(dist_counts.items()):
    ax.text(i, c + 1, str(c), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "distance_hist.png", bbox_inches="tight")
plt.close(fig)

# --- export distance-annotated network for the interactive chart ---
# Reuses the same node/edge set as week 1's network export, plus each
# node's hop-distance from Spider-Man, so the front end can recolor /
# filter the already-validated week-1 layout by distance instead of by
# component. Falls back to computing its own layout if week1's export
# hasn't been generated yet in this environment.
week1_network_path = DATA_OUT / "week1_network.json"
if week1_network_path.exists():
    net = json.load(open(week1_network_path))
    for node in net["nodes"]:
        d = lengths.get(node["id"])
        node["dist"] = d if d is not None else -1  # -1 = unreachable
    with open(DATA_OUT / "week3_network.json", "w") as f:
        json.dump(net, f, separators=(",", ":"))
    print("\nWrote distance-annotated network to", DATA_OUT / "week3_network.json")
else:
    print("\nSkipped week3_network.json — run week1_analysis.py first to generate the base layout.")

dist_hist_data = {
    "distances": [int(d) for d in dist_counts.index],
    "counts": [int(c) for c in dist_counts.values],
}
with open(DATA_OUT / "week3_distance_hist.json", "w") as f:
    json.dump(dist_hist_data, f, separators=(",", ":"))

# --- export lookup table for the interactive tool ---
lookup = {}
for nid, d in lengths.items():
    path = paths[nid]
    lookup[nid] = {
        "name": name_of[nid],
        "distance": d,
        "path": [name_of[p] for p in path],
        "path_ids": path,
    }
for nid in unreachable:
    lookup[nid] = {
        "name": name_of[nid],
        "distance": None,
        "path": None,
        "path_ids": None,
    }

with open(SUMMARY_OUT / "spidey_distances.json", "w") as f:
    json.dump(lookup, f, indent=1)

# compact version for the client-side "six degrees" tool embedded in the post:
# keyed by name, includes the wiki url, minified.
compact = {}
for nid, entry in lookup.items():
    compact[entry["name"]] = {
        "d": entry["distance"],
        "p": entry["path"],
        "url": None,  # filled in below from the node roster
    }
url_of = dict(zip(nodes["name"], nodes["url"]))
for name, entry in compact.items():
    entry["url"] = url_of.get(name, "")
with open(DATA_OUT / "spidey_distances.json", "w") as f:
    json.dump(compact, f, separators=(",", ":"))

# unreachable characters (isolates + other components)
print(f"\nUnreachable characters ({len(unreachable)}):")
for nid in sorted(unreachable, key=lambda x: name_of[x]):
    print(f"  {name_of[nid]}")

# summary stats
summary = {
    "n_total": n,
    "n_reachable": len(reachable),
    "n_unreachable": len(unreachable),
    "max_distance": int(max_d),
    "farthest_characters": [
        {"name": name_of[nid], "path": [name_of[p] for p in paths[nid]]}
        for nid in farthest
    ],
    "distance_distribution": {str(d): int(c) for d, c in dist_counts.items()},
    "unreachable_names": sorted([name_of[nid] for nid in unreachable]),
    "mean_distance": float(np.mean([d for d in lengths.values() if d is not None])),
}
with open(SUMMARY_OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone.")
