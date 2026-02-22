"""
Organic Relation Discovery.

This module finds relationships that CANNOT be found within a single chunk —
they emerge from connecting entities across different chunks/documents.

Examples of organic relations:
- Entity A appears in chunk 1 (with property X)
  Entity A appears in chunk 5 (with property Y)
  → Merge: A has both X and Y
  
- Entity A is mentioned with Entity B in chunk 2
  Entity B is mentioned with Entity C in chunk 7
  → Inferred: A is transitively connected to C

- Same metric mentioned across different time periods
  → Temporal progression relation

This is what separates Graph RAG from flat RAG.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from schema import (
    ExtractedTriple, EntityType, RelationType,
    get_entity_type_safe, get_relation_type_safe,
)

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EntityProfile:
    """
    Aggregated profile of an entity across all chunks.
    Built by scanning all triples mentioning this entity.
    """
    name: str
    entity_type: EntityType
    # All chunks where this entity appears
    source_chunks: Set[str] = field(default_factory=set)
    # All outgoing relations: (relation, target_entity)
    outgoing: List[Tuple[RelationType, str]] = field(default_factory=list)
    # All incoming relations: (source_entity, relation)
    incoming: List[Tuple[str, RelationType]] = field(default_factory=list)
    # All lenses that found this entity
    lenses: Set[str] = field(default_factory=set)
    # Confidence scores seen
    confidence_scores: List[float] = field(default_factory=list)

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0

    @property
    def chunk_count(self) -> int:
        return len(self.source_chunks)

    @property
    def is_hub(self) -> bool:
        """Entity is a hub if it appears in many chunks or has many connections."""
        return self.chunk_count >= 3 or (len(self.outgoing) + len(self.incoming)) >= 5


@dataclass
class OrganicRelation(ExtractedTriple):
    """An organically discovered cross-chunk relation."""
    discovery_method: str = "organic"
    supporting_chunks: List[str] = field(default_factory=list)


# ============================================================
# ENTITY GRAPH (IN-MEMORY)
# ============================================================

class EntityGraph:
    """
    In-memory graph of all entities and their profiles.
    Built from extracted triples before writing to Memgraph.
    """

    def __init__(self):
        # entity_name_lower -> EntityProfile
        self._profiles: Dict[str, EntityProfile] = {}

    def build_from_triples(self, triples: List[ExtractedTriple]) -> None:
        """Build entity profiles from all extracted triples."""
        self._profiles = {}

        for triple in triples:
            subj_key = triple.subject.lower()
            obj_key = triple.object.lower()

            # Ensure subject profile exists
            if subj_key not in self._profiles:
                self._profiles[subj_key] = EntityProfile(
                    name=triple.subject,
                    entity_type=triple.subject_type,
                )

            # Ensure object profile exists
            if obj_key not in self._profiles:
                self._profiles[obj_key] = EntityProfile(
                    name=triple.object,
                    entity_type=triple.object_type,
                )

            # Update subject profile
            subj_profile = self._profiles[subj_key]
            subj_profile.source_chunks.add(triple.chunk_id)
            subj_profile.outgoing.append((triple.relation, triple.object))
            subj_profile.lenses.add(triple.lens)
            subj_profile.confidence_scores.append(triple.confidence)

            # Update object profile
            obj_profile = self._profiles[obj_key]
            obj_profile.source_chunks.add(triple.chunk_id)
            obj_profile.incoming.append((triple.subject, triple.relation))
            obj_profile.lenses.add(triple.lens)
            obj_profile.confidence_scores.append(triple.confidence)

        logger.info(f"EntityGraph built: {len(self._profiles)} unique entities")

    def get_profile(self, entity_name: str) -> Optional[EntityProfile]:
        return self._profiles.get(entity_name.lower())

    def get_all_profiles(self) -> Dict[str, EntityProfile]:
        return self._profiles

    def get_hub_entities(self) -> List[EntityProfile]:
        return [p for p in self._profiles.values() if p.is_hub]

    def get_entities_by_type(self, entity_type: EntityType) -> List[EntityProfile]:
        return [
            p for p in self._profiles.values()
            if p.entity_type == entity_type
        ]

    def get_shared_entities(
        self, chunk_id_1: str, chunk_id_2: str
    ) -> List[EntityProfile]:
        """Find entities that appear in BOTH chunks."""
        return [
            p for p in self._profiles.values()
            if chunk_id_1 in p.source_chunks and chunk_id_2 in p.source_chunks
        ]


# ============================================================
# ORGANIC RELATION DISCOVERERS
# ============================================================

class CooccurrenceRelationDiscoverer:
    """
    Discovers relations between entities that co-occur across chunks
    but were never explicitly linked within a single chunk.

    Logic: If entity A and entity B both appear in 3+ chunks together,
    they likely have a meaningful relationship even if not explicitly stated.
    """

    def __init__(self, min_cooccurrence: int = 2):
        self.min_cooccurrence = min_cooccurrence

    def discover(
        self,
        entity_graph: EntityGraph,
        existing_triples: List[ExtractedTriple],
        source_doc: str = "organic",
    ) -> List[OrganicRelation]:
        """Find new relations from co-occurrence patterns."""
        # Build existing relation set to avoid duplicates
        existing_relations = {
            (t.subject.lower(), t.relation, t.object.lower())
            for t in existing_triples
        }

        # Build chunk → entities mapping
        chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        for profile in entity_graph.get_all_profiles().values():
            for chunk_id in profile.source_chunks:
                chunk_to_entities[chunk_id].add(profile.name.lower())

        # Count co-occurrences between entity pairs
        cooccurrence: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for chunk_id, entities in chunk_to_entities.items():
            entity_list = sorted(entities)
            for i, e1 in enumerate(entity_list):
                for e2 in entity_list[i+1:]:
                    pair = (min(e1, e2), max(e1, e2))  # Canonical pair order
                    cooccurrence[pair].add(chunk_id)

        # Generate organic relations for high co-occurrence pairs
        new_relations: List[OrganicRelation] = []

        for (e1, e2), shared_chunks in cooccurrence.items():
            if len(shared_chunks) < self.min_cooccurrence:
                continue

            p1 = entity_graph.get_profile(e1)
            p2 = entity_graph.get_profile(e2)
            if not p1 or not p2:
                continue

            # Check if relation already exists
            if (e1, RelationType.PART_OF, e2) in existing_relations:
                continue

            # Determine best relation type based on entity types
            relation = self._infer_relation(p1, p2)
            if relation is None:
                continue

            # Skip if this exact triple already exists
            if (e1, relation, e2) in existing_relations:
                continue

            relation_obj = OrganicRelation(
                subject=p1.name,
                subject_type=p1.entity_type,
                relation=relation,
                object=p2.name,
                object_type=p2.entity_type,
                confidence=min(0.6 + 0.1 * len(shared_chunks), 0.85),
                evidence=f"Co-occurs in {len(shared_chunks)} chunks: {list(shared_chunks)[:3]}",
                lens="organic_cooccurrence",
                chunk_id="cross_chunk",
                doc_id=source_doc,
                discovery_method="cooccurrence",
                supporting_chunks=list(shared_chunks),
            )
            new_relations.append(relation_obj)

        logger.info(f"CooccurrenceDiscoverer: found {len(new_relations)} organic relations")
        return new_relations

    def _infer_relation(
        self, p1: EntityProfile, p2: EntityProfile
    ) -> Optional[RelationType]:
        """Infer the most likely relation based on entity types."""
        # Entity type pair → likely relation
        type_pair_map = {
            (EntityType.PERSON, EntityType.ORGANIZATION): RelationType.WORKS_AT,
            (EntityType.ORGANIZATION, EntityType.LOCATION): RelationType.LOCATED_IN,
            (EntityType.PRODUCT, EntityType.ORGANIZATION): RelationType.PART_OF,
            (EntityType.SERVICE, EntityType.ORGANIZATION): RelationType.PART_OF,
            (EntityType.TECHNOLOGY, EntityType.PRODUCT): RelationType.SUPPORTS,
            (EntityType.TECHNOLOGY, EntityType.PROCESS): RelationType.ENABLES,
            (EntityType.ROLE, EntityType.DEPARTMENT): RelationType.BELONGS_TO,
            (EntityType.PERSON, EntityType.ROLE): RelationType.RESPONSIBLE_FOR,
            (EntityType.ORGANIZATION, EntityType.MARKET): RelationType.OPERATES_IN,
            (EntityType.ORGANIZATION, EntityType.INDUSTRY): RelationType.OPERATES_IN,
            (EntityType.DEPARTMENT, EntityType.ORGANIZATION): RelationType.PART_OF,
            (EntityType.TEAM, EntityType.DEPARTMENT): RelationType.PART_OF,
            (EntityType.PROJECT, EntityType.ORGANIZATION): RelationType.PART_OF,
            (EntityType.CAPABILITY, EntityType.ORGANIZATION): RelationType.PART_OF,
            (EntityType.PROCESS, EntityType.ORGANIZATION): RelationType.PART_OF,
        }

        pair = (p1.entity_type, p2.entity_type)
        reverse_pair = (p2.entity_type, p1.entity_type)

        return type_pair_map.get(pair) or type_pair_map.get(reverse_pair)


class HubEntityRelationDiscoverer:
    """
    Discovers indirect relations through hub entities.

    If A → HUB → B exists but A → B does not,
    creates a weaker indirect relation A → B with lower confidence.

    This captures multi-hop knowledge that would otherwise
    require multiple RAG queries.
    """

    def __init__(self, max_hops: int = 2):
        self.max_hops = max_hops

    def discover(
        self,
        entity_graph: EntityGraph,
        existing_triples: List[ExtractedTriple],
        source_doc: str = "organic",
    ) -> List[OrganicRelation]:
        """Find transitive relations through hub entities."""
        existing_relations = {
            (t.subject.lower(), t.object.lower())
            for t in existing_triples
        }

        hub_entities = entity_graph.get_hub_entities()
        logger.info(f"HubEntityDiscoverer: {len(hub_entities)} hub entities identified")

        new_relations: List[OrganicRelation] = []

        for hub in hub_entities:
            # Get all entities directly connected to hub
            hub_neighbors = set()
            for (relation, target) in hub.outgoing:
                hub_neighbors.add(target.lower())
            for (source, relation) in hub.incoming:
                hub_neighbors.add(source.lower())

            if len(hub_neighbors) < 2:
                continue

            # For each pair of hub neighbors, check if they should be connected
            neighbor_list = list(hub_neighbors)
            for i, n1 in enumerate(neighbor_list):
                for n2 in neighbor_list[i+1:]:
                    # Skip if already connected
                    if (n1, n2) in existing_relations or (n2, n1) in existing_relations:
                        continue

                    p1 = entity_graph.get_profile(n1)
                    p2 = entity_graph.get_profile(n2)
                    if not p1 or not p2:
                        continue

                    # Both connected to same hub → likely related
                    new_rel = OrganicRelation(
                        subject=p1.name,
                        subject_type=p1.entity_type,
                        relation=RelationType.COLLABORATES_WITH,
                        object=p2.name,
                        object_type=p2.entity_type,
                        confidence=0.5,  # Lower confidence for inferred
                        evidence=f"Both connected to hub entity: {hub.name}",
                        lens="organic_hub",
                        chunk_id="cross_chunk",
                        doc_id=source_doc,
                        discovery_method="hub_inference",
                        supporting_chunks=list(hub.source_chunks),
                    )
                    new_relations.append(new_rel)

        # Keep only top-confidence hub relations (avoid explosion)
        new_relations.sort(key=lambda x: x.confidence, reverse=True)
        max_hub_relations = 50
        result = new_relations[:max_hub_relations]

        logger.info(f"HubEntityDiscoverer: {len(result)} hub-inferred relations (capped at {max_hub_relations})")
        return result


class TemporalRelationDiscoverer:
    """
    Discovers temporal progression relations between entities
    that appear with different time contexts across chunks.

    Example: "Company had 100 employees in 2020" and
             "Company has 500 employees in 2024"
    → Creates: (Company, 2020_state) -PRECEDED_BY-> (Company, 2024_state)
    """

    TEMPORAL_KEYWORDS = [
        "2020", "2021", "2022", "2023", "2024", "2025",
        "q1", "q2", "q3", "q4",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "year", "month", "quarter",
        "began", "started", "founded", "established",
        "current", "previous", "future", "planned",
    ]

    def discover(
        self,
        entity_graph: EntityGraph,
        existing_triples: List[ExtractedTriple],
        source_doc: str = "organic",
    ) -> List[OrganicRelation]:
        """Find temporal triples involving Timeframe entities."""
        timeframe_entities = entity_graph.get_entities_by_type(EntityType.TIMEFRAME)

        if not timeframe_entities:
            logger.info("TemporalDiscoverer: no Timeframe entities found")
            return []

        new_relations: List[OrganicRelation] = []

        # Find pairs of timeframe entities → infer ordering
        timeframe_entities.sort(key=lambda p: p.name)

        for i, tf1 in enumerate(timeframe_entities):
            for tf2 in timeframe_entities[i+1:]:
                # Try to determine temporal order
                if self._comes_before(tf1.name, tf2.name):
                    rel = OrganicRelation(
                        subject=tf1.name,
                        subject_type=EntityType.TIMEFRAME,
                        relation=RelationType.PRECEDED_BY,
                        object=tf2.name,
                        object_type=EntityType.TIMEFRAME,
                        confidence=0.75,
                        evidence=f"Temporal ordering inferred from names",
                        lens="organic_temporal",
                        chunk_id="cross_chunk",
                        doc_id=source_doc,
                        discovery_method="temporal_inference",
                        supporting_chunks=[],
                    )
                    new_relations.append(rel)

        logger.info(f"TemporalDiscoverer: {len(new_relations)} temporal relations")
        return new_relations

    def _comes_before(self, name1: str, name2: str) -> bool:
        """Simple heuristic: extract year numbers and compare."""
        import re
        years1 = re.findall(r'\b(20\d{2}|19\d{2})\b', name1)
        years2 = re.findall(r'\b(20\d{2}|19\d{2})\b', name2)
        if years1 and years2:
            return int(years1[0]) < int(years2[0])
        return False


# ============================================================
# ORGANIC RELATION PIPELINE
# ============================================================

class OrganicRelationPipeline:
    """
    Runs all organic relation discoverers on the extracted triple set.
    """

    def __init__(
        self,
        min_cooccurrence: int = 2,
        enable_hub_discovery: bool = True,
        enable_temporal_discovery: bool = True,
        max_organic_relations: int = 200,
    ):
        self.discoverers = [
            CooccurrenceRelationDiscoverer(min_cooccurrence=min_cooccurrence),
        ]
        if enable_hub_discovery:
            self.discoverers.append(HubEntityRelationDiscoverer())
        if enable_temporal_discovery:
            self.discoverers.append(TemporalRelationDiscoverer())

        self.max_organic_relations = max_organic_relations

    def run(
        self,
        extracted_triples: List[ExtractedTriple],
        source_doc: str = "organic",
        progress_callback=None,
    ) -> Tuple[List[ExtractedTriple], List[OrganicRelation]]:
        """
        Run organic relation discovery.

        Returns:
            - Combined triples (original + organic)
            - Just the organic relations
        """
        # Build entity graph from all extracted triples
        entity_graph = EntityGraph()
        entity_graph.build_from_triples(extracted_triples)

        all_organic: List[OrganicRelation] = []

        for i, discoverer in enumerate(self.discoverers):
            name = type(discoverer).__name__
            if progress_callback:
                progress_callback(i, len(self.discoverers), f"Running {name}...")

            logger.info(f"Running organic discoverer: {name}")
            organic = discoverer.discover(entity_graph, extracted_triples, source_doc)
            all_organic.extend(organic)

        # Sort by confidence and cap
        all_organic.sort(key=lambda x: x.confidence, reverse=True)
        all_organic = all_organic[:self.max_organic_relations]

        # Combine with original triples
        combined = extracted_triples + all_organic

        logger.info(
            f"Organic pipeline complete: "
            f"{len(extracted_triples)} original + "
            f"{len(all_organic)} organic = "
            f"{len(combined)} total triples"
        )

        return combined, all_organic

    def get_organic_summary(
        self, organic_relations: List[OrganicRelation]
    ) -> Dict:
        """Summary of organically discovered relations."""
        by_method = defaultdict(int)
        for rel in organic_relations:
            by_method[rel.discovery_method] += 1

        return {
            "total_organic": len(organic_relations),
            "by_method": dict(by_method),
            "avg_confidence": (
                sum(r.confidence for r in organic_relations) / len(organic_relations)
                if organic_relations else 0.0
            ),
        }