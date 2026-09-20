"""
Crawl Category:Turing Award laureates from live Wikipedia and build the
same kind of directed link network as the course's Marvel dataset:
one node per laureate, edge A -> B when A's article links to B's.

Uses action=parse + wikitext to find [[Page name]] links, mirroring how
the Marvel network's edges were harvested (per the assignment notes).
"""
import requests
import re
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "analysis" / "bonus_turing" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "social-graph-interactions/1.0 "
        "(student project, DTU 02805; "
        "https://github.com/TSimona08/social-graph-interactions)"
    )
}
API = "https://en.wikipedia.org/w/api.php"
CATEGORY = "Category:Turing Award laureates"


def get_category_members(category):
    members = []
    cont = {}
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": 500,
            "cmnamespace": 0,  # articles only, no subcats/files
            "format": "json",
            **cont,
        }
        r = requests.get(API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        members.extend(m["title"] for m in data["query"]["categorymembers"])
        if "continue" in data:
            cont = data["continue"]
            time.sleep(0.2)
        else:
            break
    return members


def get_wikitext(title):
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        return None
    return data["parse"]["wikitext"]["*"]


LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def extract_links(wikitext):
    if not wikitext:
        return set()
    links = set()
    for m in LINK_RE.finditer(wikitext):
        target = m.group(1).strip()
        if not target or ":" in target.split("/")[0] and target.split(":")[0].lower() in (
            "file", "image", "category", "template", "wikipedia", "help", "portal", "special"
        ):
            continue
        # normalize: Wikipedia titles are case-sensitive after first char is not,
        # first letter capitalized, spaces for underscores
        target = target[0].upper() + target[1:] if target else target
        target = target.replace("_", " ")
        links.add(target)
    return links


if __name__ == "__main__":
    print("Fetching category members...")
    members = get_category_members(CATEGORY)
    print(f"{len(members)} laureates found.")

    member_set = set(members)
    node_id = {name: name for name in members}  # keep names as-is for node ids

    edges = []
    wikitext_cache = {}
    for i, name in enumerate(members):
        wt = get_wikitext(name)
        wikitext_cache[name] = wt
        links = extract_links(wt)
        # keep only edges to other members of this category
        targets = links & member_set
        targets.discard(name)
        for t in targets:
            edges.append((name, t))
        print(f"[{i+1}/{len(members)}] {name}: {len(targets)} in-category links")
        time.sleep(0.15)

    out = {
        "category": CATEGORY,
        "n_nodes": len(members),
        "n_edges": len(edges),
        "nodes": members,
        "edges": edges,
    }
    with open(OUT_DIR / "turing_network.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone. {len(members)} nodes, {len(edges)} edges.")
