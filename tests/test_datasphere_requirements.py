"""Ensure Datasphere requirements files parse (no comments or pip flags as lines)."""

from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILES = [
    REPO_ROOT / "jobs" / "requirements-datasphere-cpu.txt",
    REPO_ROOT / "jobs" / "requirements-datasphere-gpu.txt",
]


@pytest.mark.parametrize("requirements_path", REQUIREMENTS_FILES, ids=lambda p: p.name)
def test_requirements_lines_are_packaging_parseable(requirements_path: Path) -> None:
    for line_no, raw_line in enumerate(requirements_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        assert not line.startswith("#"), f"{requirements_path.name}:{line_no}: comments not supported"
        assert not line.startswith("-"), f"{requirements_path.name}:{line_no}: use job YAML pip config for flags"
        try:
            Requirement(line)
        except InvalidRequirement as exc:
            pytest.fail(f"{requirements_path.name}:{line_no}: {line!r} — {exc}")
