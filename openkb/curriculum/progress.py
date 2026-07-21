from __future__ import annotations

import json
from pathlib import Path

from openkb.locks import atomic_write_json


class LearnerProgress:
    """Tracks which concepts a learner has completed, persisted per KB."""

    def __init__(self, path: Path) -> None:
        self._path = path
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._completed: set[str] = set(data.get("completed", []))
        else:
            self._completed = set()

    @property
    def completed(self) -> set[str]:
        return set(self._completed)

    def is_completed(self, slug: str) -> bool:
        return slug in self._completed

    def mark_complete(self, slug: str) -> bool:
        """Mark slug completed. Returns False if it was already marked."""
        if slug in self._completed:
            return False
        self._completed.add(slug)
        self._persist()
        return True

    def mark_incomplete(self, slug: str) -> bool:
        """Unmark slug as completed. Returns False if it wasn't marked."""
        if slug not in self._completed:
            return False
        self._completed.discard(slug)
        self._persist()
        return True

    def _persist(self) -> None:
        atomic_write_json(self._path, {"completed": sorted(self._completed)})


def progress_path(kb_dir: Path) -> Path:
    return kb_dir / ".openkb" / "progress.json"


def load_progress(kb_dir: Path) -> LearnerProgress:
    return LearnerProgress(progress_path(kb_dir))
