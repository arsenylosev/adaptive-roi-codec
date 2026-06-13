"""Tests for Datasphere job config rendering."""

from pathlib import Path

import yaml

from adaptive_roi_codec.jobs.launch import render_template, write_generated_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "jobs" / "configs" / "job_train.yaml.template"


def test_rendered_job_config_is_valid_yaml_with_required_sections(tmp_path: Path) -> None:
    context = {
        "JOB_NAME": "test-train",
        "JOB_DESC": "unit test job",
        "S3_CONNECTOR_ID": "connector-test",
        "S3_DATA_PREFIX": "kvasir-capsule",
        "S3_CHECKPOINT_SUBDIR": "checkpoints",
        "WORKING_STORAGE_GB": "150",
    }
    rendered = render_template(TEMPLATE, context)
    config = yaml.safe_load(rendered)

    assert config["name"] == "test-train"
    assert config["s3-mounts"] == ["connector-test"]
    assert "g1.1" in config["cloud-instance-types"]
    assert "python -m adaptive_roi_codec.train" in config["cmd"]


def test_write_generated_config_persists_file(tmp_path: Path, monkeypatch) -> None:
    from adaptive_roi_codec.jobs import launch as launch_module

    monkeypatch.setattr(launch_module, "GENERATED_DIR", tmp_path)
    content = "name: smoke\ndesc: test\n"
    path = write_generated_config(content, "smoke.yaml")

    assert path.exists()
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["name"] == "smoke"
