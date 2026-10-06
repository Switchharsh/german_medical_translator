"""Align a (source, hypothesis, reference) report triple into short parallel segments.

Why this exists
---------------
Learned metrics have hard input limits. xCOMET concatenates source, hypothesis and
reference into ONE sequence of at most 512 tokens, and on PARROT 47% of reports exceed
that once all three are counted (median report ~164 tokens per side, p90 441, max 1104).
Scoring whole reports silently truncates the reference mid-text. Segment-level scoring
needs the three texts cut into aligned pieces, and the translations are not guaranteed
to have the same number of sentences as the source.

Method
------
1. Split each text into units (lines, then sentences).
2. Align source<->reference and source<->hypothesis separately with Gale & Church
   (1993) length-based dynamic programming, which allows 1-1, 1-0, 0-1, 2-1, 1-2, 2-2
   beads.
3. Keep only source boundaries that BOTH alignments agree on. The segments between them
   are the unit of scoring, so a disagreement coarsens the segment rather than pairing
   the wrong sentences.
4. If a segment is still over the token budget, split it at the middle source unit and
   cut the other two texts at the proportional character position. This is approximate;
   such segments are flagged ``approx=True`` so the share can be reported.

Gale-Church needs no model or network and is deterministic.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from medmt_eval.data.chunking import split_sentences

# Gale & Church 1993 defaults: mean length ratio and variance per character.
_C, _S2 = 1.0, 6.8
_BEADS = {(1, 1): 0.89, (1, 0): 0.0099, (0, 1): 0.0099, (2, 1): 0.089, (1, 2): 0.089, (2, 2): 0.011}


def units(text: str) -> list[str]:
    """Lines first (reports are full of header lines), then sentences within a line."""
    out: list[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in split_sentences(line)] if len(line) > 40 else [line]
        out.extend(p for p in parts if p)
    return out


def _cost(l1: int, l2: int) -> float:
    if l1 == 0 and l2 == 0:
        return 0.0
    delta = (l2 - l1 * _C) / math.sqrt(max(l1, 1) * _S2)
    prob = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(delta) / math.sqrt(2.0))))
    return -math.log(max(prob, 1e-30))


def align(src: Sequence[str], tgt: Sequence[str]) -> list[tuple[int, int, int, int]]:
    """Gale-Church beads as (src_start, src_end, tgt_start, tgt_end), half-open."""
    n, m = len(src), len(tgt)
    cs = [0]; ct = [0]
    for s in src: cs.append(cs[-1] + len(s))
    for t in tgt: ct.append(ct[-1] + len(t))
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if D[i][j] == INF:
                continue
            for (di, dj), prior in _BEADS.items():
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                c = D[i][j] + _cost(cs[ni] - cs[i], ct[nj] - ct[j]) - math.log(prior)
                if c < D[ni][nj]:
                    D[ni][nj] = c
                    back[ni][nj] = (i, j)
    beads = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = back[i][j]
        beads.append((pi, i, pj, j))
        i, j = pi, pj
    return beads[::-1]


@dataclass
class Segment:
    src: str
    hyp: str
    ref: str
    approx: bool = False


def _boundaries(beads, n_src):
    return {b[1] for b in beads} | {0, n_src}


def _gather(beads, tgt_units, segs):
    """Collect each segment's target text from the beads that fall inside it."""
    out = [[] for _ in segs]
    s = 0
    for (i0, i1, j0, j1) in beads:
        while s < len(segs) - 1 and i1 > segs[s][1]:
            s += 1
        out[s].extend(tgt_units[j0:j1])
    return [" ".join(x) for x in out]


def _split_oversize(seg: Segment, src_units: list[str], length_fn, budget: int, depth=0) -> list[Segment]:
    total = length_fn(seg.src) + length_fn(seg.hyp) + length_fn(seg.ref)
    if total <= budget or len(src_units) < 2 or depth > 6:
        return [seg]
    k = len(src_units) // 2
    left_src, right_src = " ".join(src_units[:k]), " ".join(src_units[k:])
    frac = len(left_src) / max(1, len(left_src) + len(right_src))

    def cut(text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        pos = int(len(text) * frac)
        # snap to the nearest whitespace so words are not split
        left = text.rfind(" ", 0, pos + 1)
        right = text.find(" ", pos)
        cand = [c for c in (left, right) if c > 0]
        pos = min(cand, key=lambda c: abs(c - pos)) if cand else pos
        return text[:pos].strip(), text[pos:].strip()

    hl, hr = cut(seg.hyp); rl, rr = cut(seg.ref)
    a = Segment(left_src, hl, rl, True); b = Segment(right_src, hr, rr, True)
    return (_split_oversize(a, src_units[:k], length_fn, budget, depth + 1)
            + _split_oversize(b, src_units[k:], length_fn, budget, depth + 1))


def align_triple(src: str, hyp: str, ref: str, *, length_fn: Callable[[str], int] = len,
                 budget: int = 440) -> list[Segment]:
    su, hu, ru = units(src), units(hyp), units(ref)
    if not su or not hu or not ru:
        return [Segment(src.strip(), hyp.strip(), ref.strip(), True)]
    b_ref, b_hyp = align(su, ru), align(su, hu)
    common = sorted(_boundaries(b_ref, len(su)) & _boundaries(b_hyp, len(su)))
    segs = list(zip(common[:-1], common[1:]))
    s_txt = [" ".join(su[a:b]) for a, b in segs]
    r_txt, h_txt = _gather(b_ref, ru, segs), _gather(b_hyp, hu, segs)
    out: list[Segment] = []
    for (a, b), s, h, r in zip(segs, s_txt, h_txt, r_txt):
        out.extend(_split_oversize(Segment(s, h, r), su[a:b], length_fn, budget))
    return out


def flatten(rows, *, length_fn=len, budget: int = 440):
    """Segment every row. Returns (segments_by_row, unique_triples).

    ``segments_by_row[i]`` is the list of triple keys for row i; identical triples are
    scored once. Round-trip cycles converge to fixed points, so most later cycles repeat
    earlier segments verbatim.
    """
    unique: dict[tuple[str, str, str], int] = {}
    by_row: list[list[int]] = []
    approx_by_row: list[float] = []
    for r in rows:
        segs = align_triple(r["src"], r["hyp"], r["ref"], length_fn=length_fn, budget=budget)
        ids = []
        for s in segs:
            key = (s.src, s.hyp, s.ref)
            ids.append(unique.setdefault(key, len(unique)))
        by_row.append(ids)
        approx_by_row.append(sum(s.approx for s in segs) / max(1, len(segs)))
    triples = [None] * len(unique)
    for k, i in unique.items():
        triples[i] = k
    return by_row, triples, approx_by_row
