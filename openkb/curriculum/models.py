from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CurriculumEdge:
    """Relationship from the current concept to another concept.

    concept:
        Slug of the related concept.

    required:
        Whether the learner must know this concept before learning
        the current concept.

    dependency:
        Relationship type. Examples:
        - prerequisite
        - conceptual
        - application
        - related
    """

    concept: str
    required: bool = True
    dependency: str = "prerequisite"


@dataclass(slots=True)
class LearningObjective:
    objective: str
    level: str = ""


@dataclass(slots=True)
class Misconception:
    misconception: str
    correction: str = ""


@dataclass(slots=True)
class Curriculum:

    difficulty: str = ""

    estimated_hours: float = 0.0

    prerequisites: list[CurriculumEdge] = field(default_factory=list)

    learning_objectives: list[LearningObjective] = field(default_factory=list)

    misconceptions: list[Misconception] = field(default_factory=list)

    next_concepts: list[CurriculumEdge] = field(default_factory=list)


@dataclass(slots=True)
class ConceptNode:

    slug: str

    title: str

    description: str

    curriculum: Curriculum

    source_file: str