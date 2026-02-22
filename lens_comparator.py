"""
Lens Comparator — runs the same question through every active lens
and produces a structured diff report showing what each lens
uniquely sees vs what all lenses agree on.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ── Triple fingerprint for dedup / overlap detection ─────────
def triple_key(t: dict) -> str:
    """Normalised key for comparing triples across lenses."""
    return (
        t.get("subject", "").lower().strip(),
        t.get("relation", "").lower().strip(),
        t.get("object", "").lower().strip(),
    )


def entity_key(name: str) -> str:
    return name.lower().strip()


# ── Per-lens result container ─────────────────────────────────
def run_lens_query(
    question: str,
    lens_name: str,
    embed_model,
    llm,
    top_k: int,
    query_fn,           # run_graph_rag_query from app.py
) -> dict:
    """
    Run a single lens query and return a structured result.
    Returns timing, triples, entities, answer.
    """
    t0 = time.time()
    try:
        result = query_fn(
            question    = question,
            embed_model = embed_model,
            llm         = llm,
            top_k       = top_k,
            lens_filter = lens_name,
        )
        result["lens_name"]    = lens_name
        result["query_time_s"] = round(time.time() - t0, 2)
        result["error"]        = None
        return result
    except Exception as e:
        logger.error(f"Lens query failed for {lens_name}: {e}")
        return {
            "lens_name":    lens_name,
            "answer":       "",
            "triples":      [],
            "entities_found": [],
            "source_chunks":  [],
            "timing":       {},
            "query_time_s": round(time.time() - t0, 2),
            "error":        str(e),
        }


def run_unfiltered_query(
    question: str,
    embed_model,
    llm,
    top_k: int,
    query_fn,
) -> dict:
    """Run the same question with NO lens filter (baseline)."""
    t0 = time.time()
    try:
        result = query_fn(
            question    = question,
            embed_model = embed_model,
            llm         = llm,
            top_k       = top_k,
            lens_filter = None,
        )
        result["lens_name"]    = "no_lens"
        result["query_time_s"] = round(time.time() - t0, 2)
        result["error"]        = None
        return result
    except Exception as e:
        return {
            "lens_name":    "no_lens",
            "answer":       "",
            "triples":      [],
            "entities_found": [],
            "source_chunks":  [],
            "timing":       {},
            "query_time_s": round(time.time() - t0, 2),
            "error":        str(e),
        }


# ── Comparison report builder ─────────────────────────────────
def build_comparison_report(
    baseline: dict,
    lens_results: List[dict],
    active_lenses: List[str],
) -> dict:
    """
    Analyse all results and produce a structured comparison report.

    Returns:
      - overlap:        triples ALL lenses agree on
      - unique_per_lens: triples only THIS lens found
      - entity_coverage: which entities each lens surfaces
      - relation_coverage: which relation types each lens uses
      - confidence_profile: avg/max/min conf per lens
      - answer_lengths:  how verbose each lens answer is
      - lens_focus:      what topics each lens emphasises
      - blind_spots:     what each lens MISSES vs baseline
    """
    report = {}

    # ── 1. Triple overlap analysis ────────────────────────────
    # Map: triple_key → set of lenses that found it
    triple_to_lenses: Dict[tuple, set] = defaultdict(set)

    baseline_keys = set()
    for t in baseline.get("triples", []):
        baseline_keys.add(triple_key(t))

    lens_triple_sets: Dict[str, set] = {}
    for lr in lens_results:
        lname = lr["lens_name"]
        keys  = set(triple_key(t) for t in lr.get("triples", []))
        lens_triple_sets[lname] = keys
        for k in keys:
            triple_to_lenses[k].add(lname)

    # Triples found by ALL active lenses (consensus)
    all_lens_names = set(lr["lens_name"] for lr in lens_results)
    consensus_keys = {
        k for k, lenses in triple_to_lenses.items()
        if len(lenses) == len(lens_results) and len(lens_results) > 1
    }

    # Triples unique to each lens
    unique_per_lens: Dict[str, List[tuple]] = {}
    for lr in lens_results:
        lname = lr["lens_name"]
        keys  = lens_triple_sets.get(lname, set())
        unique_per_lens[lname] = [
            k for k in keys
            if len(triple_to_lenses[k]) == 1
        ]

    # Triples in baseline but NOT in any lens (lost by filtering)
    lens_all_keys = set().union(*lens_triple_sets.values()) if lens_triple_sets else set()
    baseline_only = baseline_keys - lens_all_keys

    # Triples in lenses but NOT in baseline (gained by lens focus)
    lens_gained   = lens_all_keys - baseline_keys

    report["triple_overlap"] = {
        "consensus_count":   len(consensus_keys),
        "baseline_count":    len(baseline_keys),
        "lens_union_count":  len(lens_all_keys),
        "baseline_only":     len(baseline_only),
        "lens_gained":       len(lens_gained),
        "unique_per_lens":   {k: len(v) for k, v in unique_per_lens.items()},
    }

    # ── 2. Entity coverage per lens ───────────────────────────
    entity_coverage: Dict[str, List[str]] = {}
    all_entities_baseline = set(
        entity_key(e["name"]) for e in baseline.get("entities_found", [])
    )

    for lr in lens_results:
        lname = lr["lens_name"]
        ents  = [entity_key(e["name"]) for e in lr.get("entities_found", [])]
        entity_coverage[lname] = ents

    # Entities unique to each lens vs baseline
    unique_entities_per_lens: Dict[str, List[str]] = {}
    for lname, ents in entity_coverage.items():
        unique_entities_per_lens[lname] = [
            e for e in ents if e not in all_entities_baseline
        ]

    report["entity_coverage"] = {
        "baseline_entities":      list(all_entities_baseline),
        "per_lens":               entity_coverage,
        "unique_vs_baseline":     unique_entities_per_lens,
    }

    # ── 3. Relation type profile per lens ────────────────────
    relation_profile: Dict[str, Dict[str, int]] = {}
    for lr in lens_results:
        lname   = lr["lens_name"]
        rel_cnt: Dict[str, int] = defaultdict(int)
        for t in lr.get("triples", []):
            rel_cnt[t.get("relation", "?")] += 1
        relation_profile[lname] = dict(rel_cnt)

    baseline_rel_cnt: Dict[str, int] = defaultdict(int)
    for t in baseline.get("triples", []):
        baseline_rel_cnt[t.get("relation", "?")] += 1
    relation_profile["no_lens"] = dict(baseline_rel_cnt)

    report["relation_profile"] = relation_profile

    # ── 4. Confidence profile per lens ───────────────────────
    conf_profile: Dict[str, dict] = {}
    for lr in [baseline] + lens_results:
        lname   = lr["lens_name"]
        confs   = [t.get("confidence", 0) for t in lr.get("triples", [])]
        if confs:
            conf_profile[lname] = {
                "avg":   round(sum(confs) / len(confs), 3),
                "max":   round(max(confs), 3),
                "min":   round(min(confs), 3),
                "count": len(confs),
            }
        else:
            conf_profile[lname] = {
                "avg": 0, "max": 0, "min": 0, "count": 0
            }

    report["confidence_profile"] = conf_profile

    # ── 5. Answer length + token estimate ────────────────────
    answer_stats: Dict[str, dict] = {}
    for lr in [baseline] + lens_results:
        lname  = lr["lens_name"]
        answer = lr.get("answer", "")
        words  = len(answer.split())
        answer_stats[lname] = {
            "chars":     len(answer),
            "words":     words,
            "sentences": answer.count(".") + answer.count("!") + answer.count("?"),
        }

    report["answer_stats"] = answer_stats

    # ── 6. Topic focus (top nouns from answer) ────────────────
    # Simple keyword frequency — no NLTK needed
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "it", "its", "this", "that", "these", "those", "they", "them",
        "their", "there", "then", "than", "as", "or", "and", "but",
        "not", "no", "so", "if", "i", "we", "you", "he", "she",
        "based", "only", "context", "provided", "above", "which",
        "about", "also", "more", "other", "any", "all", "each",
    }

    topic_focus: Dict[str, List[Tuple[str, int]]] = {}
    for lr in [baseline] + lens_results:
        lname  = lr["lens_name"]
        answer = lr.get("answer", "").lower()
        words  = [
            w.strip(".,;:!?\"'()[]")
            for w in answer.split()
            if len(w) > 3
        ]
        freq: Dict[str, int] = defaultdict(int)
        for w in words:
            if w not in STOPWORDS:
                freq[w] += 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
        topic_focus[lname] = top

    report["topic_focus"] = topic_focus

    # ── 7. Blind spots: what lens MISSES vs baseline ──────────
    blind_spots: Dict[str, List[str]] = {}
    for lr in lens_results:
        lname     = lr["lens_name"]
        lens_keys = lens_triple_sets.get(lname, set())
        missed    = baseline_keys - lens_keys
        blind_spots[lname] = list(missed)[:10]   # cap at 10

    report["blind_spots"] = {
        lname: len(v) for lname, v in blind_spots.items()
    }

    # ── 8. Timing comparison ──────────────────────────────────
    timing_cmp: Dict[str, float] = {
        lr["lens_name"]: lr.get("query_time_s", 0)
        for lr in [baseline] + lens_results
    }
    report["timing"] = timing_cmp

    # ── Store raw for rendering ───────────────────────────────
    report["_baseline"]     = baseline
    report["_lens_results"] = lens_results
    report["_unique_triples"] = unique_per_lens
    report["_baseline_only_keys"] = baseline_only

    return report