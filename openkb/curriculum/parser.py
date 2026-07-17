from __future__ import annotations

from pathlib import Path

from openkb import frontmatter

from .models import (
    ConceptNode,
    Curriculum,
    CurriculumEdge,
    LearningObjective,
    Misconception,
)

from pathlib import Path

def _normalize_slug(value: str) -> str:
    value = value.strip()

    # Remove wiki links if present
    value = value.strip("[]")

    # Keep only the filename
    value = Path(value).stem

    return value

def _parse_edges(values):

    if not isinstance(values, list):
        return []

    edges = []

    for value in values:

        if isinstance(value, str):

            edges.append(CurriculumEdge(concept=_normalize_slug(value)))

    return edges

def _parse_learning_objectives(values):

    if not isinstance(values, list):
        return []

    objectives = []

    for value in values:

        if isinstance(value, str):

            objectives.append(
                LearningObjective(
                    objective=value
                )
            )

    return objectives

def _parse_misconceptions(values):

    if not isinstance(values, list):
        return []

    misconceptions = []

    for value in values:

        if isinstance(value, str):

            misconceptions.append(
                Misconception(
                    misconception=value
                )
            )

    return misconceptions

def _parse_curriculum(data):

    if not isinstance(data, dict):

        return Curriculum()

    return Curriculum(

        difficulty=data.get("difficulty", ""),

        estimated_hours=float(
            data.get("estimated_hours", 0)
        ),

        prerequisites=_parse_edges(
            data.get("prerequisites")
        ),

        learning_objectives=_parse_learning_objectives(
            data.get("learning_objectives")
        ),

        misconceptions=_parse_misconceptions(
            data.get("misconceptions")
        ),

        next_concepts=_parse_edges(
            data.get("next_concepts")
        ),
    )

def parse_concept(path: Path) -> ConceptNode:

    text = path.read_text(encoding="utf-8")

    metadata = frontmatter.parse(text)

    return ConceptNode(
        slug=path.stem,
        title = metadata.get(
            "title",
            path.stem.replace("-", " ").title(),
        ),
        description=metadata.get("description", ""),

        curriculum=_parse_curriculum(
            metadata.get("curriculum")
        ),

        source_file=str(path),
    )

def load_all_concepts(wiki_dir: Path):

    concepts = {}

    concepts_dir = wiki_dir / "concepts"

    if not concepts_dir.exists():
        return concepts
    
    for path in sorted(concepts_dir.glob("*.md")):

        concept = parse_concept(path)

        concepts[concept.slug] = concept

    return concepts
