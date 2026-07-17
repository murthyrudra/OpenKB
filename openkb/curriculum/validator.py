from __future__ import annotations

from dataclasses import dataclass, field

from .graph import CurriculumGraph


@dataclass(slots=True)
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0
    
def validate_graph(graph: CurriculumGraph) -> ValidationResult:
    result = ValidationResult()

    for slug, concept in graph.nodes.items():

        seen = set()

        for edge in concept.curriculum.prerequisites:

            # Missing prerequisite
            if edge.concept not in graph.nodes:
                result.errors.append(
                    f"{slug}: missing prerequisite '{edge.concept}'"
                )

            # Self dependency
            if edge.concept == slug:
                result.errors.append(
                    f"{slug}: cannot depend on itself"
                )

            # Duplicate prerequisite
            if edge.concept in seen:
                result.warnings.append(
                    f"{slug}: duplicate prerequisite '{edge.concept}'"
                )

            seen.add(edge.concept)

    return result