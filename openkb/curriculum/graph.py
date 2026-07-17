from __future__ import annotations

from dataclasses import dataclass, field

from .models import ConceptNode
from collections import deque

@dataclass(slots=True)
class CurriculumGraph:

    nodes: dict[str, ConceptNode] = field(default_factory=dict)

    children: dict[str, set[str]] = field(default_factory=dict)

    parents: dict[str, set[str]] = field(default_factory=dict)

    def prerequisites(self, slug):

        return self.parents.get(slug, set())

    def dependents(self, slug):

        return self.children.get(slug, set())

    def has(self, slug):

        return slug in self.nodes
    
    def learning_order(self):

        indegree = {
            node: len(self.parents[node])
            for node in self.nodes
        }

        queue = deque(
            node
            for node, degree in indegree.items()
            if degree == 0
        )

        order = []

        while queue:

            node = queue.popleft()

            order.append(node)

            for child in self.children[node]:

                indegree[child] -= 1

                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError(
                "Curriculum graph contains a cycle."
            )

        return order
    
    def parents_of(self, slug: str) -> set[str]:
        return self.parents.get(slug, set())
    
    def children_of(self, slug: str) -> set[str]:
        return self.children.get(slug, set())
    
    def ancestors(self, slug: str) -> set[str]:
        """
        Return every prerequisite (direct and indirect)
        of the given concept.
        """

        visited = set()
        stack = list(self.parents_of(slug))

        while stack:
            node = stack.pop()

            if node in visited:
                continue

            visited.add(node)

            stack.extend(self.parents_of(node) - visited)

        return visited
    
    def _reachable(
        self,
        start: str,
        neighbor_fn,
    ) -> set[str]:

        visited = set()
        stack = list(neighbor_fn(start))

        while stack:
            node = stack.pop()

            if node in visited:
                continue

            visited.add(node)

            stack.extend(neighbor_fn(node) - visited)

        return visited
    
    def ancestors(self, slug: str) -> set[str]:
        return self._reachable(slug, self.parents_of)
    
    def descendants(self, slug: str) -> set[str]:
        return self._reachable(slug, self.children_of)
    
    def has_path(self, source: str, target: str) -> bool:
        return target in self.descendants(source)
    
    def induced_subgraph(
        self,
        nodes: set[str],
    ) -> "CurriculumGraph":

        subgraph = CurriculumGraph()

        subgraph.nodes = {
            slug: self.nodes[slug]
            for slug in nodes
        }

        for slug in nodes:

            subgraph.parents[slug] = (
                self.parents_of(slug) & nodes
            )

            subgraph.children[slug] = (
                self.children_of(slug) & nodes
            )

        return subgraph
    
    def to_dict(self) -> dict:
        return {
            "nodes": {
                slug: {
                    "difficulty": node.curriculum.difficulty,
                    "estimated_hours": node.curriculum.estimated_hours,
                    "prerequisites": sorted(self.parents_of(slug)),
                    "children": sorted(self.children_of(slug)),
                }
                for slug, node in self.nodes.items()
            }
        }
    
    def to_mermaid(self) -> str:
        """Return a Mermaid flowchart representation."""

        lines = [
            "flowchart TD"
        ]

        # Emit all nodes
        for slug, node in sorted(self.nodes.items()):
            title = getattr(node, "title", slug)
            title = title.replace('"', '\\"')
            lines.append(f'    {slug}["{title}"]')

        lines.append("")

        # Emit edges
        seen = set()

        for parent, children in self.children.items():
            for child in children:
                edge = (parent, child)
                if edge in seen:
                    continue
                seen.add(edge)

                lines.append(f"    {parent} --> {child}")

        return "\n".join(lines)
    
    def save_mermaid(self, path):
        with open(path, "w") as f:
            f.write(self.to_mermaid())



