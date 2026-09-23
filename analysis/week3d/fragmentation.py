"""
Week 3 extension — fragmentation / robustness.

Remove characters one at a time, in different orders, and track how the
giant component shrinks. Orders compared:
  - by degree (highest total degree removed first) -- "attack"
  - by betweenness centrality (highest first) -- "attack"
  - random order, averaged over several runs -- "random failure" baseline

Also records exactly what happens to giant-component size at the moment
Spider-Man himself is removed, for both attack orders.
"""
import pandas as pd
import numpy as np
import networkx as nx
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_OUT = REPO_ROOT / "docs" / "assets" / "data"
SUMMARY_OUT = REPO_ROOT / "analysis" / "week3d" / "output"
DATA_OUT.mkdir(parents=True, exist_ok=True)
SUMMARY_OUT.mkdir(parents=True, exist_ok=True)


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
UG_full = G.to_undirected()

n_total = UG_full.number_of_nodes()
print(f"n = {n_total}")

# --- centrality rankings (computed once, on the full graph) ---
print("Computing betweenness centrality (this can take a moment)...")
betweenness = nx.betweenness_centrality(UG_full)
degree = dict(UG_full.degree())

top_betweenness = sorted(betweenness.items(), key=lambda kv: -kv[1])[:10]
top_degree = sorted(degree.items(), key=lambda kv: -kv[1])[:10]
print("\nTop 10 by betweenness centrality:")
for nid, v in top_betweenness:
    print(f"  {name_of[nid]:35s} {v:.4f}")
print("\nTop 10 by degree:")
for nid, v in top_degree:
    print(f"  {name_of[nid]:35s} {v}")

spiderman_betweenness_rank = [nid for nid, _ in sorted(betweenness.items(), key=lambda kv: -kv[1])].index("Spider-Man") + 1
spiderman_degree_rank = [nid for nid, _ in sorted(degree.items(), key=lambda kv: -kv[1])].index("Spider-Man") + 1
print(f"\nSpider-Man rank by betweenness: {spiderman_betweenness_rank} / {n_total}")
print(f"Spider-Man rank by degree: {spiderman_degree_rank} / {n_total}")


def giant_component_sizes(order):
    """Remove nodes in `order`, return list of giant-component sizes after
    each removal (index i = size after removing the first i+1 nodes)."""
    H = UG_full.copy()
    sizes = []
    for nid in order:
        if H.has_node(nid):
            H.remove_node(nid)
        if H.number_of_nodes() == 0:
            sizes.append(0)
            continue
        largest = max(nx.connected_components(H), key=len)
        sizes.append(len(largest))
    return sizes


order_degree = [nid for nid, _ in sorted(degree.items(), key=lambda kv: -kv[1])]
order_betweenness = [nid for nid, _ in sorted(betweenness.items(), key=lambda kv: -kv[1])]

print("\nSimulating targeted removal by degree...")
sizes_degree = giant_component_sizes(order_degree)
print("Simulating targeted removal by betweenness...")
sizes_betweenness = giant_component_sizes(order_betweenness)

print("Simulating random removal (averaged over 30 runs)...")
random.seed(42)
all_nodes = list(UG_full.nodes())
n_runs = 30
random_runs = []
for run in range(n_runs):
    order = all_nodes[:]
    random.shuffle(order)
    random_runs.append(giant_component_sizes(order))
sizes_random_avg = np.mean(random_runs, axis=0).tolist()

# find the step at which Spider-Man is removed in each attack order
spidey_step_degree = order_degree.index("Spider-Man")
spidey_step_betweenness = order_betweenness.index("Spider-Man")

print(f"\nSpider-Man removed at step {spidey_step_degree + 1} in degree-order attack")
print(f"  giant component size just before: {n_total if spidey_step_degree == 0 else sizes_degree[spidey_step_degree - 1]}")
print(f"  giant component size just after:  {sizes_degree[spidey_step_degree]}")

print(f"\nSpider-Man removed at step {spidey_step_betweenness + 1} in betweenness-order attack")
print(f"  giant component size just before: {n_total if spidey_step_betweenness == 0 else sizes_betweenness[spidey_step_betweenness - 1]}")
print(f"  giant component size just after:  {sizes_betweenness[spidey_step_betweenness]}")

# --- export for the interactive chart ---
export = {
    "n_total": n_total,
    "steps": list(range(1, n_total + 1)),
    "degree_order": sizes_degree,
    "betweenness_order": sizes_betweenness,
    "random_avg": sizes_random_avg,
    "spiderman_step_degree": spidey_step_degree + 1,
    "spiderman_step_betweenness": spidey_step_betweenness + 1,
    "spiderman_rank_degree": spiderman_degree_rank,
    "spiderman_rank_betweenness": spiderman_betweenness_rank,
    "top_betweenness": [[name_of[nid], round(v, 4)] for nid, v in top_betweenness],
    "top_degree": [[name_of[nid], v] for nid, v in top_degree],
    "removal_order_degree_names": [name_of[nid] for nid in order_degree[:20]],
    "removal_order_betweenness_names": [name_of[nid] for nid in order_betweenness[:20]],
}
with open(DATA_OUT / "week3d_fragmentation.json", "w") as f:
    json.dump(export, f, separators=(",", ":"))

# --- full removal order (node ids), for the step-by-step animation ---
# The animation frontend already has node positions from week3b_network.json
# (keyed by node id); this file just tells it, for each attack order, which
# node id is removed at each step, so it can compute "who's still alive at
# step N" without re-running the simulation client-side.
animation_export = {
    "n_total": n_total,
    "order_degree": order_degree,
    "order_betweenness": order_betweenness,
    "sizes_degree": sizes_degree,
    "sizes_betweenness": sizes_betweenness,
    "sizes_random_avg": sizes_random_avg,
}
with open(DATA_OUT / "week3d_removal_orders.json", "w") as f:
    json.dump(animation_export, f, separators=(",", ":"))

summary = {k: v for k, v in export.items() if k not in ("steps", "degree_order", "betweenness_order", "random_avg")}
with open(SUMMARY_OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Wrote week3d_fragmentation.json")
