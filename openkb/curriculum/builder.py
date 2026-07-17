from .models import ConceptNode
from .graph import CurriculumGraph

def build_curriculum_graph(
    concepts: dict[str, ConceptNode],
) -> CurriculumGraph:

    graph = CurriculumGraph()

    graph.nodes = concepts

    for slug in concepts:

        graph.children.setdefault(slug, set())

        graph.parents.setdefault(slug, set())

    for slug, concept in concepts.items():

        for edge in concept.curriculum.prerequisites:

            prerequisite = edge.concept

            if prerequisite not in concepts:
                print(
                    f"WARNING: Missing prerequisite '{prerequisite}' "
                    f"for concept '{slug}'"
                )
                continue

            graph.children[prerequisite].add(slug)

            graph.parents[slug].add(prerequisite)

    return graph