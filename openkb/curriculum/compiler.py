from pathlib import Path

from .parser import parse_concept
from pathlib import Path

from .builder import build_curriculum_graph
from .parser import load_all_concepts
from .validator import validate_graph


def compile_curriculum_graph(wiki_dir: Path):

    concepts = load_all_concepts(wiki_dir)

    graph = build_curriculum_graph(concepts)

    validation = validate_graph(graph)

    if validation.errors:
        print("\nCurriculum validation errors:")
        for error in validation.errors:
            print(f"  ERROR: {error}")

    if validation.warnings:
        print("\nCurriculum validation warnings:")
        for warning in validation.warnings:
            print(f"  WARNING: {warning}")

    return graph