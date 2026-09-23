"""
Draw Spider-Man plus the "far shore": the bridge characters (hop 1) that
lead to every hop-3 character, laid out as a small radial diagram.
"""
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_DIR = REPO_ROOT / "analysis" / "week3" / "output"
OUT = REPO_ROOT / "docs" / "assets" / "img" / "week3"
OUT.mkdir(parents=True, exist_ok=True)
summary = json.load(open(SUMMARY_DIR / "summary.json"))

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

G = nx.DiGraph()
G.add_node("Spider-Man")
for c in summary["farthest_characters"]:
    path = c["path"]  # Spider-Man, hop1, hop2, hop3(=far char)
    for a, b in zip(path[:-1], path[1:]):
        G.add_edge(a, b)

fig, ax = plt.subplots(figsize=(11, 9))
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", root="Spider-Man") if False else None
pos = nx.spring_layout(G, seed=11, k=1.1, iterations=200)
# force Spider-Man to center
cx, cy = pos["Spider-Man"]
pos = {k: (x - cx, y - cy) for k, (x, y) in pos.items()}

hop1 = {c["path"][1] for c in summary["farthest_characters"]}
hop2 = {c["path"][2] for c in summary["farthest_characters"]}
hop3 = {c["path"][3] for c in summary["farthest_characters"]}

node_colors = []
node_sizes = []
for node in G.nodes():
    if node == "Spider-Man":
        node_colors.append("#ff3b3b"); node_sizes.append(900)
    elif node in hop1:
        node_colors.append("#e0b45c"); node_sizes.append(400)
    elif node in hop2:
        node_colors.append("#7fa8d9"); node_sizes.append(250)
    else:
        node_colors.append("#d9b8ac"); node_sizes.append(180)

nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#3a0d10", width=1.0, alpha=0.5,
                        arrows=True, arrowsize=10, connectionstyle="arc3,rad=0.08")
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                        linewidths=1, edgecolors="#3a0d10")

for node in G.nodes():
    x, y = pos[node]
    fs = 10 if node == "Spider-Man" else (8.5 if node in hop1 else 7)
    fw = "bold" if node in ({"Spider-Man"} | hop1) else "normal"
    ax.annotate(node, (x, y), fontsize=fs, fontweight=fw,
                xytext=(0, 9), textcoords="offset points", ha="center")

ax.set_title(
    "The far shore: every 3-hop chain from Spider-Man\n"
    "red = Spider-Man · gold = bridge (hop 1) · blue = hop 2 · tan = farthest (hop 3)",
    fontsize=12,
)
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "far_shore.png", bbox_inches="tight")
plt.close(fig)
print("done")
