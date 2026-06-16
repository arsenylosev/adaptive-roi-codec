"""Ensure stage-1 extract CLI stays free of PyTorch imports."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_FRAMES = REPO_ROOT / "adaptive_roi_codec" / "cli" / "extract_frames.py"
TORCH_FREE_MODULES = {
    "adaptive_roi_codec.utils.frame_io",
    "adaptive_roi_codec.utils.video_index",
    "adaptive_roi_codec.utils.config",
    "adaptive_roi_codec.utils.env",
}


def _module_imports(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_extract_frames_cli_does_not_import_torch() -> None:
    imports = _module_imports(EXTRACT_FRAMES)
    assert "torch" not in imports
    assert "adaptive_roi_codec.utils.kvasir_loader" not in imports


def test_stage1_helper_modules_do_not_import_torch() -> None:
    for dotted in TORCH_FREE_MODULES:
        rel = dotted.replace(".", "/") + ".py"
        imports = _module_imports(REPO_ROOT / rel)
        assert "torch" not in imports, dotted
