"""
Week 4, part 2 — link communities and the community leaders.

Two additions on top of analyze.py's weighted-vs-unweighted Louvain result:

1. COMMUNITY LEADERS: for each Louvain community (both the unweighted and
   weighted partition), find its "leader" — the member with the highest
   betweenness centrality COMPUTED WITHIN THAT COMMUNITY'S OWN SUBGRAPH
   (not global betweenness). This asks "who is the broker inside this
   circle," not "who is famous across the whole network."

2. LINK COMMUNITIES: Louvain assigns each philosopher to exactly ONE
   community. Real intellectual traditions overlap — Aristotle is plausibly
   both "a student of Plato" and "the root of Aristotelianism/scholastic
   philosophy," two different circles. Link communities (Ahn, Bagrow &
   Lehmann, 2010) cluster EDGES instead of NODES: two edges that share a
   node are merged into the same link community if they connect to similar
   neighborhoods. A node then belongs to every link community touching any
   of its edges — so it can straddle several at once, which is exactly the
   "which tradition is X really in" question.

   Implemented by hand (not in networkx): for every pair of edges sharing a
   node, compute a Jaccard-style similarity of their endpoints' inclusive
   neighborhoods, single-linkage cluster the edges on that similarity, and
   cut the resulting dendrogram at the threshold that maximizes partition
   density (the link-community analogue of modularity).

Outputs:
  - console stats (leaders, Aristotle's link-community memberships)
  - docs/assets/data/week4_leaders.json     (per-community leader + their
    link-community memberships)
  - docs/assets/data/week4_linkcomms.json   (per-node list of link-community
    ids they belong to, for the explorer)
"""
import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_OUT = REPO_ROOT / "docs" / "assets" / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)

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


# --- rebuild the same graphs + Louvain partitions as analyze.py ---
# (kept self-contained/re-derived rather than importing analyze.py, since
# that script runs its full pipeline — incl. figure generation — on import)
nodes_path = REPO_ROOT / "week4_philosophers_nodes.tsv"
edges_path = REPO_ROOT / "week4_philosophers_edges.tsv"

nodes = pd.read_csv(nodes_path, sep="\t", skiprows=count_leading_comments(nodes_path))
edges = pd.read_csv(edges_path, sep="\t", skiprows=count_leading_comments(edges_path) + 1,
                     names=["source", "target", "weight"])
edges["weight"] = edges["weight"].astype(int)

name_of = dict(zip(nodes["node_id"], nodes["name"]))
era_of = dict(zip(nodes["node_id"], nodes["era"]))

UW = nx.Graph()
W = nx.Graph()
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

giant_nodes = max(nx.connected_components(UW), key=len)
UWg = UW.subgraph(giant_nodes).copy()
Wg = W.subgraph(giant_nodes).copy()

comms_uw = sorted(nx.community.louvain_communities(UWg, weight=None, seed=RNG_SEED), key=len, reverse=True)
comms_w = sorted(nx.community.louvain_communities(Wg, weight="weight", seed=RNG_SEED), key=len, reverse=True)

label_uw = {n: i for i, c in enumerate(comms_uw) for n in c}
label_w = {n: i for i, c in enumerate(comms_w) for n in c}

print(f"unweighted: {len(comms_uw)} communities, weighted: {len(comms_w)} communities")

# =====================================================================
# PART 1 — community leaders by WITHIN-COMMUNITY betweenness centrality
# =====================================================================
def community_leaders(G, comms, weight_key):
    """For each community, the member with highest betweenness centrality
    computed on that community's own induced subgraph (a local, not global,
    notion of 'broker')."""
    leaders = []
    for i, c in enumerate(comms):
        if len(c) < 3:
            # betweenness is meaningless on 1-2 node subgraphs
            leader = next(iter(c))
            leaders.append({"community": i, "size": len(c), "leader_id": leader,
                             "leader_name": name_of.get(leader, leader), "betweenness": 0.0})
            continue
        sub = G.subgraph(c)
        bc = nx.betweenness_centrality(sub, weight=weight_key, normalized=True)
        leader_id, score = max(bc.items(), key=lambda kv: kv[1])
        leaders.append({
            "community": i, "size": len(c),
            "leader_id": leader_id, "leader_name": name_of.get(leader_id, leader_id),
            "betweenness": round(float(score), 4),
        })
    return leaders

print("\ncomputing within-community betweenness (unweighted partition)...")
leaders_uw = community_leaders(UWg, comms_uw, weight_key=None)
print("computing within-community betweenness (weighted partition)...")
# betweenness with `weight` treats it as a DISTANCE (lower = closer), but our
# weight is a STRENGTH (higher = closer) — invert it per edge for this call
Wg_dist = Wg.copy()
for a, b, data in Wg_dist.edges(data=True):
    data["distance"] = 1.0 / data["weight"]
leaders_w = community_leaders(Wg_dist, comms_w, weight_key="distance")

print("\nLeaders, unweighted partition:")
for L in leaders_uw:
    print(f"  community {L['community']:2d} (n={L['size']:4d}): {L['leader_name']:30s} betweenness={L['betweenness']:.3f}")
print("\nLeaders, weighted partition:")
for L in leaders_w:
    print(f"  community {L['community']:2d} (n={L['size']:4d}): {L['leader_name']:30s} betweenness={L['betweenness']:.3f}")

# =====================================================================
# PART 2 — link communities (Ahn, Bagrow & Lehmann 2010), hand-rolled
# =====================================================================
print("\nbuilding link communities...")

G = Wg  # run on the weighted giant component
edge_list = list(G.edges())
edge_index = {frozenset(e): i for i, e in enumerate(edge_list)}
n_edges = len(edge_list)

# inclusive neighborhood: a node's neighbors plus itself
neighbors_incl = {n: set(G.neighbors(n)) | {n} for n in G.nodes()}


def edge_similarity(e1, e2, shared_node):
    """Jaccard similarity of the inclusive neighborhoods of the two OTHER
    endpoints of edges e1=(shared,a), e2=(shared,b). Weighted variant from
    the original paper: weighted Jaccard using tie strength as the vector
    magnitude on each neighbor. Falls back cleanly to plain Jaccard when
    weights are absent."""
    a = e1[0] if e1[1] == shared_node else e1[1]
    b = e2[0] if e2[1] == shared_node else e2[1]
    na, nb = neighbors_incl[a], neighbors_incl[b]
    inter = na & nb
    union = na | nb
    if not union:
        return 0.0

    def w(x, y):
        if x == y:
            # self-similarity term: use the average weight of x/shared_node's ties
            return 1.0
        if G.has_edge(x, y):
            return G[x][y].get("weight", 1)
        return 0.0

    num = sum(min(w(a, x), w(b, x)) for x in inter)
    den = sum(max(w(a, x), w(b, x)) for x in union)
    return num / den if den > 0 else 0.0


# only compare edge pairs that share a node (the only pairs that can ever
# be merged early in single-linkage clustering; this is what keeps the
# O(sum deg^2)-not-O(E^2) algorithm tractable)
pair_best = {}  # (edge_i, edge_j) -> similarity, i<j
for node in G.nodes():
    incident = [(node, nb) for nb in G.neighbors(node)]
    incident_idx = [edge_index[frozenset(e)] for e in incident]
    m = len(incident)
    for x in range(m):
        for y in range(x + 1, m):
            i, j = incident_idx[x], incident_idx[y]
            key = (min(i, j), max(i, j))
            sim = edge_similarity(incident[x], incident[y], node)
            if key not in pair_best or sim > pair_best[key]:
                pair_best[key] = sim

print(f"  {len(pair_best)} edge-pairs compared (sharing a node)")

# single-linkage hierarchical clustering needs a full condensed distance
# matrix over edges that appear in at least one pair; edges with NO shared-
# node partner (shouldn't happen in a connected graph, but just in case)
# get their own singleton cluster at the end.
involved_edges = sorted({i for pair in pair_best for i in pair})
edge_pos = {e: i for i, e in enumerate(involved_edges)}
m = len(involved_edges)

# distance = 1 - similarity; unseen pairs (no shared node) get distance 1
dist = np.ones((m, m))
np.fill_diagonal(dist, 0.0)
for (i, j), sim in pair_best.items():
    pi, pj = edge_pos[i], edge_pos[j]
    d = 1.0 - sim
    dist[pi, pj] = d
    dist[pj, pi] = d

condensed = squareform(dist, checks=False)
Z = linkage(condensed, method="single")

# --- partition density: link-community analogue of modularity ---
# For a cluster of m_c edges spanning n_c nodes: D_c = (m_c - (n_c-1)) / (n_c(n_c-1)/2 - (n_c-1))
# (0 for a tree, 1 for a clique). Overall partition density is the edge-count-weighted average.
def partition_density(cluster_labels):
    clusters = defaultdict(list)
    for idx, lab in enumerate(cluster_labels):
        clusters[lab].append(involved_edges[idx])
    total = 0.0
    M = n_edges
    for lab, eidxs in clusters.items():
        if len(eidxs) < 2:
            continue
        node_set = set()
        for ei in eidxs:
            a, b = edge_list[ei]
            node_set.add(a); node_set.add(b)
        m_c = len(eidxs)
        n_c = len(node_set)
        denom = (n_c * (n_c - 1) / 2) - (n_c - 1)
        if denom <= 0:
            continue
        d_c = (m_c - (n_c - 1)) / denom
        total += m_c * d_c
    return total / M if M else 0.0


print("  scanning cut thresholds for max partition density...")
best_density = -1.0
best_labels = None
# scan a reasonable number of distinct merge heights from the dendrogram
heights = np.unique(Z[:, 2])
sample_heights = heights[:: max(1, len(heights) // 200)]  # cap ~200 evaluations
for h in sample_heights:
    labels = fcluster(Z, t=h, criterion="distance")
    d = partition_density(labels)
    if d > best_density:
        best_density = d
        best_labels = labels

print(f"  best partition density D = {best_density:.4f}")

# map cluster label -> list of edge indices -> set of node ids per link community
cluster_edges = defaultdict(list)
for idx, lab in enumerate(best_labels):
    cluster_edges[lab].append(involved_edges[idx])

# any edges never involved in a shared-node pair (isolated dyads) become
# their own singleton link community
next_label = max(cluster_edges.keys(), default=0) + 1
for ei in range(n_edges):
    if ei not in involved_edges:
        cluster_edges[next_label] = [ei]
        next_label += 1

# renumber link communities by descending size (edge count), keep only
# clusters with >=3 edges as "real" communities (2-edge slivers are noise)
sized = sorted(cluster_edges.items(), key=lambda kv: -len(kv[1]))
link_comms = []  # list of {edges: [...], nodes: set(...)}
for lab, eidxs in sized:
    node_set = set()
    for ei in eidxs:
        a, b = edge_list[ei]
        node_set.add(a); node_set.add(b)
    link_comms.append({"edge_idxs": eidxs, "nodes": node_set})

n_real = sum(1 for lc in link_comms if len(lc["edge_idxs"]) >= 3)
print(f"  {len(link_comms)} link communities total, {n_real} with >=3 edges")
print(f"  largest 10 sizes (by edge count): {[len(lc['edge_idxs']) for lc in link_comms[:10]]}")

# node -> set of link-community indices it belongs to (via any incident edge)
node_linkcomms = defaultdict(set)
for lc_idx, lc in enumerate(link_comms):
    for n in lc["nodes"]:
        node_linkcomms[n].add(lc_idx)

overlap_counts = Counter(len(v) for v in node_linkcomms.values())
print(f"\n  node overlap: {sum(1 for v in node_linkcomms.values() if len(v) > 1)} / {len(node_linkcomms)} "
      f"philosophers belong to >1 link community")
print(f"  distribution of #link-communities per node: {dict(sorted(overlap_counts.items()))}")

# --- Aristotle spotlight ---
aristotle_id = next((n for n in G.nodes() if name_of.get(n) == "Aristotle"), None)
if aristotle_id:
    lcs = sorted(node_linkcomms.get(aristotle_id, []), key=lambda i: -len(link_comms[i]["edge_idxs"]))
    print(f"\nAristotle belongs to {len(lcs)} link communities:")
    for lc_idx in lcs[:6]:
        lc = link_comms[lc_idx]
        other_names = sorted({name_of.get(n, n) for n in lc["nodes"] if n != aristotle_id})
        print(f"  link-community {lc_idx} ({len(lc['edge_idxs'])} edges, {len(lc['nodes'])} people): "
              f"{', '.join(other_names[:8])}{', ...' if len(other_names) > 8 else ''}")

# =====================================================================
# EXPORT
# =====================================================================
leaders_export = {
    "unweighted": leaders_uw,
    "weighted": leaders_w,
}
with open(DATA_OUT / "week4_leaders.json", "w") as f:
    json.dump(leaders_export, f, indent=2)
print("\nwrote", DATA_OUT / "week4_leaders.json")

# per-node link-community membership + per-link-community member list
# (only export "real" communities, >=3 edges, to keep the file readable)
real_link_comms = [lc for lc in link_comms if len(lc["edge_idxs"]) >= 3]
real_index_map = {}  # old index in `link_comms` -> new index in `real_link_comms`
ri = 0
for i, lc in enumerate(link_comms):
    if len(lc["edge_idxs"]) >= 3:
        real_index_map[i] = ri
        ri += 1

linkcomms_export = {
    "n_link_communities": len(real_link_comms),
    "communities": [
        {
            "id": i,
            "n_edges": len(lc["edge_idxs"]),
            "n_nodes": len(lc["nodes"]),
            "members": sorted(name_of.get(n, n) for n in lc["nodes"]),
        }
        for i, lc in enumerate(real_link_comms)
    ],
    "node_memberships": {
        name_of.get(n, n): sorted(real_index_map[i] for i in idxs if i in real_index_map)
        for n, idxs in node_linkcomms.items()
        if any(i in real_index_map for i in idxs)
    },
}
with open(DATA_OUT / "week4_linkcomms.json", "w") as f:
    json.dump(linkcomms_export, f)
print("wrote", DATA_OUT / "week4_linkcomms.json")

# --- combined leader-spotlight cards ---
# One card per weighted-partition Louvain community: its leader (highest
# within-community betweenness), plus that leader's link-community
# memberships EXCLUDING the giant catch-all cluster (>=50 edges — that one
# is just "the mainstream," not a distinctive tradition). Capped at the 6
# largest remaining memberships per leader so the card stays readable.
GIANT_EDGE_THRESHOLD = 50
leader_cards = []
for L in leaders_w:
    if L["size"] < 10:
        continue  # skip trivial communities (e.g. a lone 2-node pair)
    leader_name = L["leader_name"]
    memberships = linkcomms_export["node_memberships"].get(leader_name, [])
    distinctive = [i for i in memberships if linkcomms_export["communities"][i]["n_edges"] < GIANT_EDGE_THRESHOLD]
    distinctive.sort(key=lambda i: -linkcomms_export["communities"][i]["n_edges"])
    top = []
    for i in distinctive[:6]:
        c = linkcomms_export["communities"][i]
        companions = [m for m in c["members"] if m != leader_name]
        top.append({"n_edges": c["n_edges"], "companions": companions})
    leader_cards.append({
        "community": L["community"],
        "size": L["size"],
        "leader_name": leader_name,
        "betweenness": L["betweenness"],
        "n_total_link_communities": len(memberships),
        "n_distinctive_link_communities": len(distinctive),
        "top_link_communities": top,
    })

with open(DATA_OUT / "week4_leader_cards.json", "w") as f:
    json.dump(leader_cards, f, indent=2)
print("wrote", DATA_OUT / "week4_leader_cards.json")

# --- playback stages: per weighted-community leader + their best clique,
# as node ids (not names), for the in-browser staged reveal ---
playback = []
for L in leaders_w:
    if L["size"] < 10:
        continue
    leader_id = L["leader_id"]
    lc_idxs = [i for i in node_linkcomms.get(leader_id, set())
               if len(link_comms[i]["edge_idxs"]) >= 3 and len(link_comms[i]["edge_idxs"]) < GIANT_EDGE_THRESHOLD]
    lc_idxs.sort(key=lambda i: -len(link_comms[i]["edge_idxs"]))
    best_clique_ids = sorted(link_comms[lc_idxs[0]]["nodes"]) if lc_idxs else []
    playback.append({
        "community": L["community"],
        "size": L["size"],
        "leader_id": leader_id,
        "leader_name": L["leader_name"],
        "clique_ids": best_clique_ids,
    })

with open(DATA_OUT / "week4_playback.json", "w") as f:
    json.dump(playback, f, indent=2)
print("wrote", DATA_OUT / "week4_playback.json")

print("\ndone.")
