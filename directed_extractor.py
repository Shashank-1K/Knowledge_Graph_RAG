"""
Directed Knowledge Graph Extractor.
Fixed: relaxed validation, better JSON repair, fuzzy type matching.
"""

import json
import logging
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama

from schema import (
    StakeholderLens, ExtractedTriple,
    EntityType, RelationType,
    get_entity_type_safe, get_relation_type_safe,
    STAKEHOLDER_LENSES,
)

logger = logging.getLogger(__name__)


# ============================================================
# ENTITY CANONICALIZATION
# ============================================================

class EntityCanonicalizer:
    """Ensures same real-world entity always gets same name in graph."""

    def __init__(self):
        self._alias_map: Dict[str, str] = {}
        self._canonical_aliases: Dict[str, List[str]] = defaultdict(list)

    def register_alias(self, alias: str, canonical: str):
        normalized = alias.lower().strip()
        self._alias_map[normalized] = canonical
        if alias not in self._canonical_aliases[canonical]:
            self._canonical_aliases[canonical].append(alias)

    def canonicalize(self, name: str) -> str:
        normalized = name.lower().strip()
        return self._alias_map.get(normalized, name.strip())

    def auto_detect_aliases(self, triples: List[ExtractedTriple]) -> None:
        all_names = set()
        for t in triples:
            all_names.add(t.subject.strip())
            all_names.add(t.object.strip())

        sorted_names = sorted(all_names, key=len)
        for i, short_name in enumerate(sorted_names):
            for long_name in sorted_names[i + 1:]:
                if (
                    short_name.lower() in long_name.lower()
                    and len(long_name) > len(short_name) + 2
                    and len(short_name) > 2
                ):
                    self.register_alias(long_name, short_name)

    def apply_to_triples(
        self, triples: List[ExtractedTriple]
    ) -> List[ExtractedTriple]:
        for triple in triples:
            triple.subject = self.canonicalize(triple.subject)
            triple.object = self.canonicalize(triple.object)
        return triples


# ============================================================
# TRIPLE DEDUPLICATOR
# ============================================================

class TripleDeduplicator:
    """Merges duplicate triples extracted by multiple lenses."""

    def deduplicate(
        self, triples: List[ExtractedTriple]
    ) -> List[ExtractedTriple]:
        triple_map: Dict[Tuple, ExtractedTriple] = {}
        lens_tracking: Dict[Tuple, List[str]] = defaultdict(list)

        for triple in triples:
            key = (
                triple.subject.lower(),
                triple.relation,
                triple.object.lower(),
            )
            lens_tracking[key].append(triple.lens)

            if key not in triple_map:
                triple_map[key] = triple
            else:
                existing = triple_map[key]
                if triple.confidence > existing.confidence:
                    triple_map[key] = triple
                if triple.evidence not in existing.evidence:
                    existing.evidence = f"{existing.evidence} | {triple.evidence}"

        result = []
        for key, triple in triple_map.items():
            lenses = lens_tracking[key]
            if len(lenses) > 1:
                triple.lens = f"multi:{','.join(set(lenses))}"
            result.append(triple)

        logger.info(
            f"Deduplication: {len(triples)} → {len(result)} triples "
            f"({len(triples) - len(result)} duplicates removed)"
        )
        return result


# ============================================================
# JSON REPAIR UTILITY
# ============================================================

def repair_json_array(text: str) -> str:
    """
    Attempt to repair truncated or malformed JSON arrays.
    Handles the most common LLM output issues.
    """
    text = text.strip()

    # Remove trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # If array is not closed, close it
    open_brackets = text.count('[') - text.count(']')
    open_braces = text.count('{') - text.count('}')

    # Close unclosed objects first, then array
    if open_braces > 0:
        text += '}' * open_braces
    if open_brackets > 0:
        text += ']' * open_brackets

    return text


# ============================================================
# LENS-BASED EXTRACTOR
# ============================================================

class LensExtractor:
    """Applies a single stakeholder lens to a text chunk."""

    # All valid entity types as a set for fast lookup
    ALL_ENTITY_VALUES = {et.value for et in EntityType}
    ALL_RELATION_VALUES = {rt.value for rt in RelationType}

    def __init__(self, llm: Ollama, lens: StakeholderLens):
        self.llm = llm
        self.lens = lens
        # Build lowercase lookup maps for fuzzy matching
        self._entity_lower = {et.value.lower(): et for et in EntityType}
        self._relation_lower = {rt.value.lower(): rt for rt in RelationType}
        # Allowed sets for this lens
        self._allowed_entities = set(self.lens.entity_types)
        self._allowed_relations = set(self.lens.relationship_types)

    def _build_prompt(self, text: str) -> str:
        entity_types_str = ", ".join(et.value for et in self.lens.entity_types)
        relation_types_str = ", ".join(rt.value for rt in self.lens.relationship_types)
        return self.lens.extraction_prompt_template.format(
            text=text,
            entity_types=entity_types_str,
            relationship_types=relation_types_str,
        )

    def _parse_response(self, response_text: str) -> List[dict]:
        """Extract and repair JSON from LLM response."""
        text = response_text.strip()

        # Strip markdown fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Find JSON array boundaries
        start = text.find('[')
        end = text.rfind(']')

        if start == -1:
            logger.warning(
                f"No JSON array found in lens '{self.lens.name}' response"
            )
            return []

        if end == -1 or end < start:
            # Array was cut off — take from [ to end and repair
            candidate = text[start:]
        else:
            candidate = text[start: end + 1]

        # Try parse as-is first
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Try repair
        repaired = repair_json_array(candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parse error for lens '{self.lens.name}' "
                f"even after repair: {e}"
            )
            logger.debug(f"Raw (first 600): {response_text[:600]}")
            return []

    def _fuzzy_entity_type(self, value: str) -> Optional[EntityType]:
        """
        Match entity type string with tolerance for case/spacing differences.
        Tries: exact → case-insensitive → substring.
        """
        if not value:
            return None
        # Exact
        et = get_entity_type_safe(value)
        if et:
            return et
        # Case-insensitive
        et = self._entity_lower.get(value.lower().strip())
        if et:
            return et
        # Substring (e.g. "hr" → "HR_People")
        val_lower = value.lower().strip()
        for key, et in self._entity_lower.items():
            if val_lower in key or key in val_lower:
                return et
        return None

    def _fuzzy_relation_type(self, value: str) -> Optional[RelationType]:
        """Match relation type with tolerance."""
        if not value:
            return None
        rt = get_relation_type_safe(value)
        if rt:
            return rt
        rt = self._relation_lower.get(value.lower().strip())
        if rt:
            return rt
        val_lower = value.lower().replace(" ", "_").replace("-", "_")
        for key, rt in self._relation_lower.items():
            if val_lower == key:
                return rt
        return None

    def _validate_triple(
        self, raw: dict, chunk_id: str, doc_id: str
    ) -> Optional[ExtractedTriple]:
        """
        Validate and convert raw dict to ExtractedTriple.

        Key change from original: we NO LONGER reject triples whose
        entity/relation types fall outside the lens definition.
        The lens prompt already guides the LLM — if it returns a valid
        schema type, we accept it.  This avoids the 1/9, 0/5 discard rates.
        """
        required = ["subject", "subject_type", "relation", "object", "object_type"]
        for field in required:
            if field not in raw or not raw[field]:
                return None

        # Fuzzy-match types
        subject_type = self._fuzzy_entity_type(raw["subject_type"])
        object_type = self._fuzzy_entity_type(raw["object_type"])
        relation = self._fuzzy_relation_type(raw["relation"])

        if not subject_type:
            logger.debug(f"Unrecognised subject_type: '{raw['subject_type']}'")
            return None
        if not object_type:
            logger.debug(f"Unrecognised object_type: '{raw['object_type']}'")
            return None
        if not relation:
            logger.debug(f"Unrecognised relation: '{raw['relation']}'")
            return None

        confidence = float(raw.get("confidence", 0.7))
        if confidence < 0.6:
            return None

        # Skip self-loops
        subj = raw["subject"].strip()
        obj = raw["object"].strip()
        if not subj or not obj:
            return None
        if subj.lower() == obj.lower():
            return None

        return ExtractedTriple(
            subject=subj,
            subject_type=subject_type,
            relation=relation,
            object=obj,
            object_type=object_type,
            confidence=confidence,
            evidence=raw.get("evidence", "")[:500],
            lens=self.lens.name,
            chunk_id=chunk_id,
            doc_id=doc_id,
        )

    def extract(
        self, text: str, chunk_id: str, doc_id: str
    ) -> List[ExtractedTriple]:
        """Run extraction for this lens on a text chunk."""
        prompt = self._build_prompt(text)

        try:
            start = time.time()
            response = self.llm.complete(prompt)
            elapsed = time.time() - start
            logger.info(
                f"Lens '{self.lens.name}' | chunk {chunk_id[:8]} | "
                f"{elapsed:.1f}s | response {len(str(response))} chars"
            )
        except Exception as e:
            logger.error(
                f"LLM call failed for lens '{self.lens.name}': {e}"
            )
            return []

        raw_triples = self._parse_response(str(response))
        logger.info(
            f"Lens '{self.lens.name}' | chunk {chunk_id[:8]} | "
            f"Parsed {len(raw_triples)} raw triples"
        )

        validated = []
        for raw in raw_triples:
            triple = self._validate_triple(raw, chunk_id, doc_id)
            if triple:
                validated.append(triple)

        logger.info(
            f"Lens '{self.lens.name}' | chunk {chunk_id[:8]} | "
            f"Validated {len(validated)}/{len(raw_triples)} triples"
        )
        return validated


# ============================================================
# DIRECTED GRAPH EXTRACTOR
# ============================================================

class DirectedGraphExtractor:
    """
    Orchestrates multi-lens extraction over all document chunks.
    """

    def __init__(
        self,
        llm: Ollama,
        active_lenses: Optional[List[str]] = None,
        min_confidence: float = 0.6,
    ):
        self.llm = llm
        self.min_confidence = min_confidence
        self.canonicalizer = EntityCanonicalizer()
        self.deduplicator = TripleDeduplicator()

        lenses_to_use = active_lenses or list(STAKEHOLDER_LENSES.keys())
        self.lens_extractors = {
            name: LensExtractor(llm, STAKEHOLDER_LENSES[name])
            for name in lenses_to_use
            if name in STAKEHOLDER_LENSES
        }
        logger.info(
            f"DirectedGraphExtractor initialized with lenses: "
            f"{list(self.lens_extractors.keys())}"
        )

    def extract_from_chunk(
        self,
        text: str,
        chunk_id: str,
        doc_id: str,
        lens_names: Optional[List[str]] = None,
    ) -> List[ExtractedTriple]:
        all_triples: List[ExtractedTriple] = []

        extractors = (
            {k: v for k, v in self.lens_extractors.items() if k in lens_names}
            if lens_names else self.lens_extractors
        )

        for lens_name, extractor in extractors.items():
            logger.info(
                f"Extracting with lens '{lens_name}' "
                f"from chunk {chunk_id[:8]}..."
            )
            triples = extractor.extract(text, chunk_id, doc_id)
            all_triples.extend(triples)

        return self.deduplicator.deduplicate(all_triples)

    def extract_from_nodes(
        self,
        nodes: List[TextNode],
        progress_callback=None,
    ) -> List[ExtractedTriple]:
        all_triples: List[ExtractedTriple] = []
        total = len(nodes)

        logger.info(
            f"Starting directed extraction: {total} chunks × "
            f"{len(self.lens_extractors)} lenses"
        )

        for i, node in enumerate(nodes):
            chunk_id = node.node_id
            doc_id = node.metadata.get("file_name", "unknown")
            text = node.text

            if progress_callback:
                progress_callback(
                    i, total,
                    f"Chunk {i+1}/{total}: applying "
                    f"{len(self.lens_extractors)} lenses..."
                )

            logger.info(
                f"Processing chunk {i+1}/{total} | "
                f"id={chunk_id[:8]} | {len(text)} chars"
            )

            chunk_triples = self.extract_from_chunk(text, chunk_id, doc_id)
            all_triples.extend(chunk_triples)

            logger.info(
                f"Chunk {i+1}/{total}: extracted {len(chunk_triples)} triples "
                f"| running total: {len(all_triples)}"
            )

        logger.info("Running entity canonicalization across all chunks...")
        self.canonicalizer.auto_detect_aliases(all_triples)
        all_triples = self.canonicalizer.apply_to_triples(all_triples)

        logger.info("Running global deduplication...")
        final_triples = self.deduplicator.deduplicate(all_triples)

        logger.info(
            f"Extraction complete: {len(final_triples)} final triples from "
            f"{total} chunks using {len(self.lens_extractors)} lenses"
        )
        return final_triples

    def get_extraction_summary(
        self, triples: List[ExtractedTriple]
    ) -> Dict:
        lens_counts: Dict[str, int] = defaultdict(int)
        entity_type_counts: Dict[str, int] = defaultdict(int)
        relation_counts: Dict[str, int] = defaultdict(int)
        unique_entities: set = set()

        for t in triples:
            primary_lens = (
                t.lens.split(":")[1].split(",")[0]
                if "multi:" in t.lens else t.lens
            )
            lens_counts[primary_lens] += 1
            entity_type_counts[t.subject_type.value] += 1
            entity_type_counts[t.object_type.value] += 1
            relation_counts[t.relation.value] += 1
            unique_entities.add(t.subject.lower())
            unique_entities.add(t.object.lower())

        return {
            "total_triples": len(triples),
            "unique_entities": len(unique_entities),
            "by_lens": dict(lens_counts),
            "by_entity_type": dict(entity_type_counts),
            "by_relation_type": dict(relation_counts),
            "avg_confidence": (
                sum(t.confidence for t in triples) / len(triples)
                if triples else 0.0
            ),
        }