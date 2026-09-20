"""
Stretch goal: re-derive a corner of the dataset via the Wikipedia API and
compare against the frozen week-1 snapshot.

We check: does Spider-Man's live "what links here" count (restricted to
articles that are plausible Marvel-superhero category members) roughly
match the snapshot's in-degree of 106?

Wikipedia API note: unauthenticated requests need a descriptive User-Agent
or you risk a 403.
"""
import requests
import time

HEADERS = {
    "User-Agent": (
        "social-graph-interactions/1.0 "
        "(student project, DTU 02805; "
        "https://github.com/TSimona08/social-graph-interactions)"
    )
}
API = "https://en.wikipedia.org/w/api.php"

def get_category_members(category, limit=500):
    members = set()
    cont = {}
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": limit,
            "format": "json",
            **cont,
        }
        r = requests.get(API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        members.update(m["title"] for m in data["query"]["categorymembers"])
        if "continue" in data:
            cont = data["continue"]
            time.sleep(0.2)
        else:
            break
    return members

def get_linkshere_count(title, restrict_to=None):
    """Count pages that link to `title`, optionally restricted to a set of titles."""
    count = 0
    cont = {}
    matched = []
    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "linkshere",
            "lhlimit": 500,
            "lhnamespace": 0,
            "format": "json",
            **cont,
        }
        r = requests.get(API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        pages = data["query"]["pages"]
        for _, page in pages.items():
            for lh in page.get("linkshere", []):
                count += 1
                if restrict_to is not None and lh["title"] in restrict_to:
                    matched.append(lh["title"])
        if "continue" in data:
            cont = data["continue"]
            time.sleep(0.2)
        else:
            break
    return count, matched

if __name__ == "__main__":
    cat = "Category:Marvel Comics superheroes"
    print("Fetching live category membership for:", cat)
    live_members = get_category_members(cat)
    print(f"Live category currently has {len(live_members)} members "
          f"(snapshot had 303 as of 2026-08-26).")

    print("\nFetching live 'what links here' for Spider-Man...")
    total_links, matched = get_linkshere_count("Spider-Man", restrict_to=live_members)
    print(f"Total live pages (ns=0) linking to Spider-Man: {total_links}")
    print(f"Of those, currently in Category:Marvel Comics superheroes: {len(matched)}")
    print("\n(Snapshot in-degree for Spider-Man, restricted to the 303-node roster: 106)")
