from __future__ import annotations

from .graph import CurriculumGraph
from .models import ConceptNode

class CurriculumPlanner:

    def __init__(self, graph: CurriculumGraph):
        self.graph = graph

    def available_concepts(
        self,
        completed: set[str],
    ) -> list[str]:

        available = []

        for slug in self.graph.nodes:

            if slug in completed:
                continue

            prerequisites = self.graph.parents[slug]

            if prerequisites.issubset(completed):
                available.append(slug)

        return sorted(available)
    
    
    def remaining_prerequisites(
        self,
        target: str,
        completed: set[str],
    ) -> list[ConceptNode]:

        return [
            concept
            for concept in self.learning_path(target)
            if concept.slug not in completed
        ]
    
    def learning_path(
        self,
        target: str,
    ) -> list[ConceptNode]:

        required = self.graph.ancestors(target)
        required.add(target)

        ordered = (
            self.graph
            .induced_subgraph(required)
            .learning_order()
        )

        return [
            self.graph.nodes[slug]
            for slug in ordered
        ]
    
    def learning_path_slugs(
        self,
        target: str,
    ) -> list[str]:
        return [
            concept.slug
            for concept in self.learning_path(target)
        ]