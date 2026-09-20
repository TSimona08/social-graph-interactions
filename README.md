# The Marvel Graph Diaries

A weekly interactive data-story series for DTU course **02805 Social Graphs and Interactions**,
mining the network of the 303 characters in Wikipedia's *Category:Marvel Comics superheroes*.

**Live site:** https://TSimona08.github.io/social-graph-interactions/

## Repo layout

```
docs/                     # published GitHub Pages site
  index.html              # home page / post list
  about.html
  posts/
    week1.html            # week 1 post
  assets/
    css/style.css
    img/week1/            # figures embedded in the week 1 post

analysis/                 # reproducible analysis code behind each post
  week1_analysis.py        # loads week1_edges.tsv/week1_nodes.tsv, produces
                            # all week-1 figures + analysis/output/week1_summary.json
  week1_wiki_check.py       # stretch goal: re-derives category size + Spider-Man
                            # in-degree live via the Wikipedia API, for comparison
                            # against the frozen snapshot

week1_edges.tsv            # frozen week-1 snapshot (course data page)
week1_nodes.tsv            # frozen week-1 node roster (303 characters, incl. isolates)
```

## Reproducing week 1

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy networkx matplotlib requests

python3 analysis/week1_analysis.py      # -> docs/assets/img/week1/*.png + analysis/output/week1_summary.json
python3 analysis/week1_wiki_check.py    # -> live Wikipedia API comparison (needs internet)
```

## Publishing / updating the site

The site is plain static HTML/CSS in `docs/` served via GitHub Pages (Settings → Pages →
Deploy from branch → `main` → folder `/docs`). No build step — edit the HTML, commit, push.

## Data

Week-1 network snapshot from the course's shared playground data page (frozen 2026-08-26):
303 nodes, 1784 directed edges, 17 isolated nodes. See `docs/posts/week1.html` for the write-up.
