#!/usr/bin/env python3
"""Build a DE/EN(/TR) medical glossary from Wikidata.

    python3 scripts/build_glossary_wikidata.py -o data/term_banks/wikidata_med.csv

Wikidata is the only source in the shortlist that is CC0, needs no registration
and is available the same day. Items are anchored on a MeSH descriptor ID
(P486) or a UMLS CUI (P2892), so every row carries a stable external concept id
and can later be joined against RadLex, SNOMED or ICD.

Aliases (``skos:altLabel``) are the reason to prefer this over a bare ontology
dump: they carry the colloquial and abbreviated variants that appear in dictated
reports and that formal terminologies leave out.

The public endpoint enforces a query timeout, so the fetch is paged and each
page is retried. Partial results are written rather than lost.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dataclasses import replace

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medmt_eval.glossary import Glossary, GlossaryEntry  # noqa: E402

ENDPOINT = "https://query.wikidata.org/sparql"
# A descriptive agent is required by the WDQS usage policy; an anonymous or
# library-default agent gets throttled or blocked.
USER_AGENT = "medmt-eval/0.1 (medical MT evaluation research; contact via repo)"

# Two passes. Combining GROUP_CONCAT over two OPTIONAL altLabel joins with an
# ORDER BY in one query makes the public endpoint time out (504) even at
# LIMIT 1500 — the sort plus the alias cross-product is too expensive. Labels
# alone return 2000 rows in ~7 s, and aliases come back cheaply when the items
# are already known and supplied as a VALUES block.
_LABELS_QUERY = """
SELECT ?item ?cid ?en ?de ?tr WHERE {
  ?item wdt:%(prop)s ?cid .
  ?item rdfs:label ?en . FILTER(LANG(?en) = "en")
  ?item rdfs:label ?de . FILTER(LANG(?de) = "de")
  OPTIONAL { ?item rdfs:label ?tr . FILTER(LANG(?tr) = "tr") }
}
LIMIT %(limit)d OFFSET %(offset)d
"""

_TREE_QUERY = """
SELECT ?item (GROUP_CONCAT(DISTINCT ?t; separator="|") AS ?tree) WHERE {
  VALUES ?item { %(items)s }
  OPTIONAL { ?item wdt:P672 ?t }
}
GROUP BY ?item
"""

_ALIAS_QUERY = """
SELECT ?item
       (GROUP_CONCAT(DISTINCT ?aen; separator="|") AS ?aliases_en)
       (GROUP_CONCAT(DISTINCT ?ade; separator="|") AS ?aliases_de)
WHERE {
  VALUES ?item { %(items)s }
  OPTIONAL { ?item skos:altLabel ?aen . FILTER(LANG(?aen) = "en") }
  OPTIONAL { ?item skos:altLabel ?ade . FILTER(LANG(?ade) = "de") }
}
GROUP BY ?item
"""

# P486 = MeSH descriptor ID, P2892 = UMLS CUI. Both give a joinable concept id.
PROPERTIES = {"mesh": "P486", "umls": "P2892"}


def _run(query: str, timeout: int = 180) -> list[dict]:
    for attempt in range(4):
        response = requests.get(
            ENDPOINT,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()["results"]["bindings"]
        # 429/5xx on a shared public endpoint is load, not a bug.
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
    return []


def fetch_labels(prop: str, limit: int, offset: int) -> list[dict]:
    return _run(_LABELS_QUERY % {"prop": prop, "limit": limit, "offset": offset})


def fetch_aliases(qids: list[str], batch: int = 400) -> dict[str, tuple[str, str]]:
    """Aliases for known items, in VALUES batches."""
    out: dict[str, tuple[str, str]] = {}
    for start in range(0, len(qids), batch):
        chunk = qids[start:start + batch]
        items = " ".join(f"wd:{q}" for q in chunk)
        for row in _run(_ALIAS_QUERY % {"items": items}):
            qid = row["item"]["value"].rsplit("/", 1)[-1]
            out[qid] = (
                row.get("aliases_en", {}).get("value", ""),
                row.get("aliases_de", {}).get("value", ""),
            )
        print(f"  aliases: {min(start + batch, len(qids))}/{len(qids)}", flush=True)
        time.sleep(0.5)
    return out


def fetch_trees(qids: list[str], batch: int = 400) -> dict[str, str]:
    """MeSH tree codes for known items — the domain filter's input."""
    out: dict[str, str] = {}
    for start in range(0, len(qids), batch):
        chunk = qids[start:start + batch]
        items = " ".join(f"wd:{q}" for q in chunk)
        for row in _run(_TREE_QUERY % {"items": items}):
            qid = row["item"]["value"].rsplit("/", 1)[-1]
            out[qid] = row.get("tree", {}).get("value", "")
        print(f"  trees: {min(start + batch, len(qids))}/{len(qids)}", flush=True)
        time.sleep(0.4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("data/term_banks/wikidata_med.csv"))
    parser.add_argument("--page-size", type=int, default=1500)
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--no-aliases", action="store_true",
                        help="Skip the alias pass (faster, loses colloquial variants).")
    parser.add_argument("--properties", nargs="+", default=["mesh", "umls"],
                        choices=sorted(PROPERTIES))
    args = parser.parse_args()

    by_concept: dict[str, GlossaryEntry] = {}
    for key in args.properties:
        prop = PROPERTIES[key]
        offset = 0
        while offset < args.max_rows:
            rows = fetch_labels(prop, args.page_size, offset)
            if not rows:
                break
            for row in rows:
                qid = row["item"]["value"].rsplit("/", 1)[-1]
                # Key on the Wikidata item: one concept can carry both a MeSH id
                # and a CUI and would otherwise be emitted twice. This also makes
                # OFFSET paging safe — the endpoint gives no stable order without
                # an ORDER BY (too slow here), so pages may overlap slightly.
                if qid in by_concept:
                    continue
                by_concept[qid] = GlossaryEntry(
                    concept_id=f"{key.upper()}:{row['cid']['value']}|{qid}",
                    source=f"wikidata-{key}",
                    en=row["en"]["value"],
                    de=row["de"]["value"],
                    tr=row.get("tr", {}).get("value", ""),
                )
            print(f"  {key}: +{len(rows):5d} rows (offset {offset}) "
                  f"-> {len(by_concept)} concepts", flush=True)
            if len(rows) < args.page_size:
                break
            offset += args.page_size
            time.sleep(1.0)

    if not args.no_aliases and by_concept:
        alias_map = fetch_aliases(sorted(by_concept))
        for qid, (aen, ade) in alias_map.items():
            entry = by_concept[qid]
            by_concept[qid] = replace(
                entry,
                aliases_en=tuple(a for a in aen.split("|") if a),
                aliases_de=tuple(a for a in ade.split("|") if a),
            )

    if by_concept:
        tree_map = fetch_trees(sorted(by_concept))
        for qid, tree in tree_map.items():
            by_concept[qid] = replace(
                by_concept[qid], tree=tuple(t for t in tree.split("|") if t)
            )

    glossary = Glossary(by_concept.values())
    glossary.to_csv(args.output)
    with_tr = sum(1 for e in glossary if e.tr)
    with_alias = sum(1 for e in glossary if e.aliases_de or e.aliases_en)
    print(f"\nwrote {args.output}")
    print(f"  {len(glossary)} concepts · {with_tr} with Turkish · {with_alias} with aliases")
    print(f"  sources: {glossary.sources()}")
    radiology = glossary.in_branches("ACE")
    print(f"  in MeSH branches A/C/E (anatomy, disease, diagnostics): {len(radiology)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
