"""
Schema definitions for directed graph generation.

Instead of letting LLM freely extract anything, we define:
1. Stakeholder lenses - different 'views' of the same data
2. Entity types per lens
3. Allowed relationship types per lens
4. Extraction templates per lens

This ensures the graph is structured, consistent, and queryable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


# ============================================================
# CORE ENUMS
# ============================================================

class EntityType(str, Enum):
    """All possible entity types across all lenses."""
    # Organization entities
    ORGANIZATION = "Organization"
    DEPARTMENT = "Department"
    TEAM = "Team"
    PERSON = "Person"
    ROLE = "Role"

    # Business entities
    PRODUCT = "Product"
    SERVICE = "Service"
    PROJECT = "Project"
    CLIENT = "Client"
    MARKET = "Market"
    INDUSTRY = "Industry"

    # Technical entities
    TECHNOLOGY = "Technology"
    TOOL = "Tool"
    PLATFORM = "Platform"
    INFRASTRUCTURE = "Infrastructure"
    PROCESS = "Process"

    # Operational entities
    LOCATION = "Location"
    EVENT = "Event"
    METRIC = "Metric"
    TIMEFRAME = "Timeframe"
    POLICY = "Policy"

    # Knowledge entities
    CONCEPT = "Concept"
    CAPABILITY = "Capability"
    OBJECTIVE = "Objective"
    CHALLENGE = "Challenge"
    OUTCOME = "Outcome"


class RelationType(str, Enum):
    """All possible relationship types across all lenses."""
    # Structural
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    BELONGS_TO = "BELONGS_TO"
    REPORTS_TO = "REPORTS_TO"
    MANAGES = "MANAGES"
    LOCATED_IN = "LOCATED_IN"

    # Operational
    USES = "USES"
    PROVIDES = "PROVIDES"
    DEPENDS_ON = "DEPENDS_ON"
    INTEGRATES_WITH = "INTEGRATES_WITH"
    SUPPORTS = "SUPPORTS"
    ENABLES = "ENABLES"

    # Business
    SERVES = "SERVES"
    PARTNERS_WITH = "PARTNERS_WITH"
    COMPETES_WITH = "COMPETES_WITH"
    OPERATES_IN = "OPERATES_IN"
    TARGETS = "TARGETS"
    GENERATES = "GENERATES"

    # People
    WORKS_AT = "WORKS_AT"
    LEADS = "LEADS"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    HAS_SKILL = "HAS_SKILL"
    RESPONSIBLE_FOR = "RESPONSIBLE_FOR"

    # Temporal
    PRECEDED_BY = "PRECEDED_BY"
    FOLLOWED_BY = "FOLLOWED_BY"
    DURING = "DURING"
    STARTED_AT = "STARTED_AT"

    # Causal
    CAUSES = "CAUSES"
    RESULTS_IN = "RESULTS_IN"
    INFLUENCES = "INFLUENCES"
    REQUIRES = "REQUIRES"
    ACHIEVES = "ACHIEVES"


# ============================================================
# STAKEHOLDER LENS DEFINITIONS
# ============================================================

@dataclass
class StakeholderLens:
    """
    Defines how a specific stakeholder 'sees' the data.

    Each stakeholder extracts different entities and relationships
    from the SAME document chunk — giving us richer, multi-perspective graphs.
    """
    name: str
    description: str
    entity_types: List[EntityType]
    relationship_types: List[RelationType]
    extraction_prompt_template: str
    priority: int = 1  # Higher = more important lens


# ============================================================
# DEFINE ALL STAKEHOLDER LENSES
# ============================================================

STAKEHOLDER_LENSES: Dict[str, StakeholderLens] = {

    "executive": StakeholderLens(
        name="Executive",
        description="C-suite view: strategy, markets, outcomes, high-level structure",
        entity_types=[
            EntityType.ORGANIZATION,
            EntityType.MARKET,
            EntityType.INDUSTRY,
            EntityType.OBJECTIVE,
            EntityType.OUTCOME,
            EntityType.CLIENT,
            EntityType.CHALLENGE,
            EntityType.METRIC,
            EntityType.TIMEFRAME,
        ],
        relationship_types=[
            RelationType.OPERATES_IN,
            RelationType.SERVES,
            RelationType.TARGETS,
            RelationType.ACHIEVES,
            RelationType.GENERATES,
            RelationType.PARTNERS_WITH,
            RelationType.RESULTS_IN,
            RelationType.INFLUENCES,
        ],
        extraction_prompt_template="""
You are extracting a knowledge graph from the perspective of a C-suite Executive.
Focus ONLY on: strategic goals, market presence, business outcomes, client relationships,
financial metrics, organizational objectives, and competitive positioning.

TEXT:
{text}

Extract entities and relationships. Use ONLY these entity types:
{entity_types}

Use ONLY these relationship types:
{relationship_types}

Output as JSON list of triples:
[
  {{"subject": "EntityName", "subject_type": "EntityType", 
    "relation": "RELATION_TYPE", 
    "object": "EntityName", "object_type": "EntityType",
    "confidence": 0.0-1.0,
    "evidence": "exact quote from text supporting this"}}
]

Rules:
- Only extract what is EXPLICITLY stated or strongly implied
- Confidence < 0.6: skip it
- Use canonical names (e.g., "GYTWorkz" not "the company")
- Output ONLY valid JSON, no explanation
""",
        priority=3,
    ),

    "technical": StakeholderLens(
        name="Technical",
        description="Engineering view: tools, technologies, infrastructure, integrations",
        entity_types=[
            EntityType.TECHNOLOGY,
            EntityType.TOOL,
            EntityType.PLATFORM,
            EntityType.INFRASTRUCTURE,
            EntityType.PROCESS,
            EntityType.PRODUCT,
            EntityType.SERVICE,
            EntityType.CAPABILITY,
        ],
        relationship_types=[
            RelationType.USES,
            RelationType.INTEGRATES_WITH,
            RelationType.DEPENDS_ON,
            RelationType.SUPPORTS,
            RelationType.ENABLES,
            RelationType.PROVIDES,
            RelationType.REQUIRES,
        ],
        extraction_prompt_template="""
You are extracting a knowledge graph from the perspective of a Technical Engineer/Architect.
Focus ONLY on: technologies, tools, platforms, technical processes, integrations,
capabilities, infrastructure components, and technical dependencies.

TEXT:
{text}

Extract entities and relationships. Use ONLY these entity types:
{entity_types}

Use ONLY these relationship types:
{relationship_types}

Output as JSON list of triples:
[
  {{"subject": "EntityName", "subject_type": "EntityType",
    "relation": "RELATION_TYPE",
    "object": "EntityName", "object_type": "EntityType",
    "confidence": 0.0-1.0,
    "evidence": "exact quote from text supporting this"}}
]

Rules:
- Only extract technical facts, not business opinions
- Confidence < 0.6: skip it
- Normalize tool names (e.g., "PostgreSQL" not "postgres")
- Output ONLY valid JSON, no explanation
""",
        priority=3,
    ),

    "hr_people": StakeholderLens(
        name="HR_People",
        description="HR view: people, roles, departments, skills, org structure",
        entity_types=[
            EntityType.PERSON,
            EntityType.ROLE,
            EntityType.DEPARTMENT,
            EntityType.TEAM,
            EntityType.ORGANIZATION,
            EntityType.CAPABILITY,
            EntityType.LOCATION,
        ],
        relationship_types=[
            RelationType.WORKS_AT,
            RelationType.LEADS,
            RelationType.MANAGES,
            RelationType.REPORTS_TO,
            RelationType.BELONGS_TO,
            RelationType.RESPONSIBLE_FOR,
            RelationType.COLLABORATES_WITH,
            RelationType.HAS_SKILL,
            RelationType.LOCATED_IN,
        ],
        extraction_prompt_template="""
You are extracting a knowledge graph from the perspective of an HR Manager.
Focus ONLY on: people, roles, teams, departments, reporting structures,
skills, responsibilities, headcount, and organizational hierarchy.

TEXT:
{text}

Extract entities and relationships. Use ONLY these entity types:
{entity_types}

Use ONLY these relationship types:
{relationship_types}

Output as JSON list of triples:
[
  {{"subject": "EntityName", "subject_type": "EntityType",
    "relation": "RELATION_TYPE",
    "object": "EntityName", "object_type": "EntityType",
    "confidence": 0.0-1.0,
    "evidence": "exact quote from text supporting this"}}
]

Rules:
- Use job titles as Role entities (e.g., "Software Engineer", "CEO")
- Use full names for Person entities when available
- Confidence < 0.6: skip it
- Output ONLY valid JSON, no explanation
""",
        priority=2,
    ),

    "operations": StakeholderLens(
        name="Operations",
        description="Ops view: processes, locations, projects, timelines, metrics",
        entity_types=[
            EntityType.PROCESS,
            EntityType.PROJECT,
            EntityType.LOCATION,
            EntityType.METRIC,
            EntityType.TIMEFRAME,
            EntityType.POLICY,
            EntityType.EVENT,
            EntityType.OUTCOME,
        ],
        relationship_types=[
            RelationType.LOCATED_IN,
            RelationType.DURING,
            RelationType.STARTED_AT,
            RelationType.PRECEDED_BY,
            RelationType.FOLLOWED_BY,
            RelationType.RESULTS_IN,
            RelationType.REQUIRES,
            RelationType.SUPPORTS,
            RelationType.PART_OF,
        ],
        extraction_prompt_template="""
You are extracting a knowledge graph from the perspective of an Operations Manager.
Focus ONLY on: processes, projects, locations, timelines, operational metrics,
policies, events, and operational outcomes.

TEXT:
{text}

Extract entities and relationships. Use ONLY these entity types:
{entity_types}

Use ONLY these relationship types:
{relationship_types}

Output as JSON list of triples:
[
  {{"subject": "EntityName", "subject_type": "EntityType",
    "relation": "RELATION_TYPE",
    "object": "EntityName", "object_type": "EntityType",
    "confidence": 0.0-1.0,
    "evidence": "exact quote from text supporting this"}}
]

Rules:
- Quantify metrics when possible (e.g., "500 employees" not "many employees")
- Use specific location names
- Confidence < 0.6: skip it
- Output ONLY valid JSON, no explanation
""",
        priority=2,
    ),

    "client_market": StakeholderLens(
        name="Client_Market",
        description="Sales/BD view: clients, markets, services, value propositions",
        entity_types=[
            EntityType.CLIENT,
            EntityType.MARKET,
            EntityType.INDUSTRY,
            EntityType.SERVICE,
            EntityType.PRODUCT,
            EntityType.CAPABILITY,
            EntityType.OUTCOME,
        ],
        relationship_types=[
            RelationType.SERVES,
            RelationType.OPERATES_IN,
            RelationType.PROVIDES,
            RelationType.TARGETS,
            RelationType.PARTNERS_WITH,
            RelationType.ACHIEVES,
            RelationType.ENABLES,
        ],
        extraction_prompt_template="""
You are extracting a knowledge graph from the perspective of a Sales/Business Development Manager.
Focus ONLY on: clients, target markets, industries served, services/products offered,
partnerships, and client outcomes/value delivered.

TEXT:
{text}

Extract entities and relationships. Use ONLY these entity types:
{entity_types}

Use ONLY these relationship types:
{relationship_types}

Output as JSON list of triples:
[
  {{"subject": "EntityName", "subject_type": "EntityType",
    "relation": "RELATION_TYPE",
    "object": "EntityName", "object_type": "EntityType",
    "confidence": 0.0-1.0,
    "evidence": "exact quote from text supporting this"}}
]

Rules:
- Identify specific client names when mentioned
- Classify industries specifically (e.g., "Healthcare" not "business")
- Confidence < 0.6: skip it
- Output ONLY valid JSON, no explanation
""",
        priority=2,
    ),
}


# ============================================================
# SCHEMA VALIDATION
# ============================================================

@dataclass
class ExtractedTriple:
    """A single validated knowledge graph triple."""
    subject: str
    subject_type: EntityType
    relation: RelationType
    object: str
    object_type: EntityType
    confidence: float
    evidence: str
    lens: str  # Which stakeholder lens extracted this
    chunk_id: str  # Source chunk ID for traceability
    doc_id: str  # Source document ID

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "subject_type": self.subject_type.value,
            "relation": self.relation.value,
            "object": self.object,
            "object_type": self.object_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "lens": self.lens,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
        }


def get_active_lenses(lens_names: Optional[List[str]] = None) -> Dict[str, StakeholderLens]:
    """Get subset of lenses by name, or all if none specified."""
    if lens_names is None:
        return STAKEHOLDER_LENSES
    return {
        name: lens
        for name, lens in STAKEHOLDER_LENSES.items()
        if name in lens_names
    }


def get_entity_type_safe(value: str) -> Optional[EntityType]:
    """Safely parse an EntityType from string."""
    try:
        return EntityType(value)
    except ValueError:
        # Try case-insensitive match
        for et in EntityType:
            if et.value.lower() == value.lower():
                return et
        return None


def get_relation_type_safe(value: str) -> Optional[RelationType]:
    """Safely parse a RelationType from string."""
    try:
        return RelationType(value)
    except ValueError:
        for rt in RelationType:
            if rt.value.lower() == value.lower():
                return rt
        return None