"""Shared helpers for resolving project-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def find_project_root(start_path: Path | None = None, markers: Iterable[str] | None = None) -> Path:
    """Return the project root by searching parent directories.

    The search stops when any of the ``markers`` exists in the directory.
    Defaults to common markers used in this repository.
    """

    search_markers = set(markers or {".git", "1_data", "3_model", "5_docs"})
    current = (start_path or Path(__file__)).resolve()
    for parent in [current] + list(current.parents):
        if any((parent / m).exists() for m in search_markers):
            return parent
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다. 검색 마커를 확인하세요.")


PROJECT_ROOT = find_project_root()

__all__ = ["find_project_root", "PROJECT_ROOT"]
