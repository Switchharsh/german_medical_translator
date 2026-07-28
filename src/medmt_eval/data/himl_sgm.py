"""Parse WMT HimL SGML test files into the pipeline's normalized Segment schema.

The WMT Biomedical shared-task SGML files use a simple structure:
    <srcset ...> / <tstset ...>
      <doc docid="..." ...>
        <seg id="N">text</seg>

This is *not* full SGML — it is well-formed XML with a handful of XML entities
(`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`).  A lightweight regex-based
extractor is sufficient and avoids pulling in a heavy SGML parser.
"""

from __future__ import annotations

import html
import re
import tarfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from medmt_eval.schema import Segment

# Segment-ID pattern inside <seg id="N"> ... </seg>.
_SEG_RE = re.compile(r'<seg\s+id="(\d+)">(.*?)</seg>', re.DOTALL)

# Subset → (file_prefix, domain-label).
# The 2015 archive uses {subset}.all.{lang}.sgm (also has .testing and .tuning,
# but .all is the canonical test set).
# The 2017 archive uses {subset}_output.{lang}.sgm.
_HIML2015_SUBSETS: dict[str, tuple[str, str]] = {
    "cochrane.all": "himl2015-cochrane",
    "nhs24.all": "himl2015-nhs24",
    "himl.testing": "himl2015-himl",
}
_HIML2017_SUBSETS: dict[str, tuple[str, str]] = {
    "cochrane_output": "himl2017-cochrane",
    "nhs_output": "himl2017-nhs",
}


def _parse_segments(text: str) -> dict[str, str]:
    """Return {seg_id: unescaped_text} from raw SGML content."""
    segments: dict[str, str] = {}
    for match in _SEG_RE.finditer(text):
        seg_id = match.group(1)
        raw_text = match.group(2)
        # Collapse SGML line-wrap whitespace inside <seg> content.
        cleaned = " ".join(raw_text.split())
        # Unescape standard XML entities (&amp; &lt; &gt; &quot; &apos;).
        unescaped = html.unescape(cleaned)
        segments[seg_id] = unescaped
    return segments


def _read_file_in_tar(tar: tarfile.TarFile, name: str) -> str:
    """Read a named member from an open tar archive as UTF-8 text."""
    member = tar.getmember(name)
    with tar.extractfile(member) as fh:
        return fh.read().decode("utf-8")


def _collect_sgm_names(tar: tarfile.TarFile, prefix: str, suffix: str) -> list[str]:
    """Return tar member names matching ``{prefix}.{suffix}.sgm``."""
    return [
        info.name
        for info in tar.getmembers()
        if info.name.endswith(f"{prefix}.{suffix}.sgm")
    ]


def _align_pair(
    src_segments: dict[str, str],
    tgt_segments: dict[str, str],
    *,
    domain: str,
    src_lang: str,
    tgt_lang: str,
    doc_id: str,
    id_prefix: str,
) -> list[Segment]:
    """Align by segment ID and emit Segment records."""
    src_ids = set(src_segments)
    tgt_ids = set(tgt_segments)
    if src_ids != tgt_ids:
        missing_in_tgt = src_ids - tgt_ids
        missing_in_src = tgt_ids - src_ids
        parts: list[str] = []
        if missing_in_tgt:
            parts.append(f"IDs in source but missing from target: {sorted(missing_in_tgt)}")
        if missing_in_src:
            parts.append(f"IDs in target but missing from source: {sorted(missing_in_src)}")
        raise ValueError(
            f"Segment-ID mismatch in {domain} ({doc_id}): {'; '.join(parts)}"
        )
    segments: list[Segment] = []
    for seg_id in sorted(src_segments, key=int):
        segments.append(
            Segment(
                id=f"{id_prefix}-{seg_id}",
                domain=domain,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=src_segments[seg_id],
                ref_text=tgt_segments[seg_id],
                doc_id=doc_id,
            )
        )
    return segments


def load_himl_from_tar(
    tar_path: str | Path,
    *,
    year: int = 2015,
    src_lang: str = "en",
    tgt_lang: str = "de",
    subsets: Mapping[str, str] | None = None,
) -> list[Segment]:
    """Load EN↔DE segments from a HimL test-set tarball.

    Parameters
    ----------
    tar_path:
        Path to ``himl-test-2015.tgz`` or ``himl-test-2017.tgz``.
    year:
        ``2015`` or ``2017`` — selects the expected subset/filename mapping.
    src_lang, tgt_lang:
        The language pair to extract.  Defaults to EN→DE.
    subsets:
        Override the default subset mapping.  Each key is the filename prefix
        (e.g. ``cochrane.all``); the value is the domain label.
    """
    if subsets is None:
        subsets = _HIML2015_SUBSETS if year == 2015 else _HIML2017_SUBSETS

    all_segments: list[Segment] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for file_prefix, domain in subsets.items():
            src_names = _collect_sgm_names(tar, file_prefix, src_lang)
            tgt_names = _collect_sgm_names(tar, file_prefix, tgt_lang)
            if not src_names:
                raise FileNotFoundError(
                    f"No {src_lang} SGML file found for subset {file_prefix!r} in {tar_path}"
                )
            if not tgt_names:
                raise FileNotFoundError(
                    f"No {tgt_lang} SGML file found for subset {file_prefix!r} in {tar_path}"
                )
            # There should be exactly one file per (subset, lang).
            src_name = src_names[0]
            tgt_name = tgt_names[0]
            src_segs = _parse_segments(_read_file_in_tar(tar, src_name))
            tgt_segs = _parse_segments(_read_file_in_tar(tar, tgt_name))
            all_segments.extend(
                _align_pair(
                    src_segs,
                    tgt_segs,
                    domain=domain,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    doc_id=file_prefix,
                    id_prefix=domain,
                )
            )
    return all_segments
