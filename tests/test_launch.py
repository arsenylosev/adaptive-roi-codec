"""Tests for Datasphere job config rendering."""

import json
from pathlib import Path

import pytest
import yaml

from adaptive_roi_codec.jobs.launch import (
    EXTRACT_TEMPLATE,
    format_train_job_desc,
    format_train_job_name,
    materialize_train_params,
    params_input_path,
    render_template,
    resolve_datasphere_cli,
    write_generated_config,
    yaml_quote,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "jobs" / "configs" / "job_train.yaml.template"


def test_rendered_job_config_is_valid_yaml_with_required_sections(tmp_path: Path) -> None:
    context = {
        "JOB_NAME": yaml_quote("test-train"),
        "JOB_DESC": yaml_quote("unit test job"),
        "PARAMS_INPUT": "jobs/inputs/train_smoke.json",
        "TRAIN_BATCH_SIZE": "12",
        "S3_CONNECTOR_ID": "connector-test",
        "S3_DATA_PREFIX": "kvasir-capsule",
        "S3_CHECKPOINT_SUBDIR": "checkpoints",
        "FRAMES_S3_DIR": "/job/s3/connector-test/kvasir-capsule/processed/frames",
        "WORKING_STORAGE_GB": "150",
    }
    rendered = render_template(TEMPLATE, context)
    config = yaml.safe_load(rendered)

    assert config["name"] == "test-train"
    assert config["s3-mounts"] == ["connector-test"]
    assert "g1.1" in config["cloud-instance-types"]
    assert config["cmd"] == (
        "python -m adaptive_roi_codec.train --config ${CONFIG} --params ${PARAMS}"
    )
    assert "\n" not in config["cmd"]
    assert config["inputs"][1] == {"jobs/inputs/train_smoke.json": "PARAMS"}
    env_vars = {}
    for item in config["env"]["vars"]:
        env_vars.update(item)
    assert str(env_vars["TRAIN_BATCH_SIZE"]) == "12"
    assert env_vars["TQDM_DISABLE"] == "1"
    assert env_vars["PYTHONUNBUFFERED"] == "1"


def test_materialize_train_params_writes_batch_size_override(tmp_path: Path, monkeypatch) -> None:
    from adaptive_roi_codec.jobs import launch as launch_module

    params_path = tmp_path / "base.json"
    params_path.write_text(
        json.dumps({"training": {"epochs": 1}, "data": {"source": "preprocessed"}}),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    monkeypatch.setattr(launch_module, "GENERATED_DIR", generated_dir)
    monkeypatch.setattr(launch_module, "GENERATED_TRAIN_PARAMS", generated_dir / "train_params.json")

    rendered = materialize_train_params(params_path, 16)
    payload = json.loads(rendered.read_text(encoding="utf-8"))

    assert rendered.name == "train_params.json"
    assert payload["training"]["batch_size"] == 16


def test_train_job_name_and_desc_include_batch_size() -> None:
    assert format_train_job_name("vae-capsule-train", 12) == "vae-capsule-train-bs12"
    assert format_train_job_desc("GPU training", 12) == "GPU training (batch_size=12)"
    assert format_train_job_name("vae-capsule-train", None) == "vae-capsule-train"


def test_params_input_path_must_be_inside_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the repository"):
        params_input_path(outside)


def test_params_input_path_returns_posix_relative_path() -> None:
    params = REPO_ROOT / "jobs" / "inputs" / "train_smoke.json"
    assert params_input_path(params) == "jobs/inputs/train_smoke.json"


def test_extract_job_description_with_colon_is_valid_yaml() -> None:
    context = {
        "JOB_NAME": yaml_quote("kvasir-extract-frames"),
        "JOB_DESC": yaml_quote("CPU stage-1: decode Kvasir MP4s to .pt tensors on S3"),
        "PARAMS_INPUT": "jobs/inputs/extract_input.json",
        "S3_CONNECTOR_ID": "connector-test",
        "S3_DATA_PREFIX": "kvasir-capsule",
        "FRAMES_S3_DIR": "/job/s3/connector-test/kvasir-capsule/processed/frames",
        "WORKING_STORAGE_GB": "150",
    }
    config = yaml.safe_load(render_template(EXTRACT_TEMPLATE, context))
    assert config["desc"] == "CPU stage-1: decode Kvasir MP4s to .pt tensors on S3"
    assert config["cloud-instance-types"][0] == "c1.8"


def test_resolve_datasphere_cli_from_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    cli = fake_bin / "datasphere"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setattr("adaptive_roi_codec.jobs.launch.sys.executable", str(fake_bin / "python"))
    monkeypatch.setattr("adaptive_roi_codec.jobs.launch.shutil.which", lambda _name: None)

    assert resolve_datasphere_cli() == str(cli)


def test_resolve_datasphere_cli_missing_exits_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "adaptive_roi_codec.jobs.launch.sys.executable",
        "/nonexistent/python",
    )
    monkeypatch.setattr("adaptive_roi_codec.jobs.launch.shutil.which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="uv sync --extra cloud"):
        resolve_datasphere_cli()


def test_write_generated_config_persists_file(tmp_path: Path, monkeypatch) -> None:
    from adaptive_roi_codec.jobs import launch as launch_module

    monkeypatch.setattr(launch_module, "GENERATED_DIR", tmp_path)
    content = "name: smoke\ndesc: test\n"
    path = write_generated_config(content, "smoke.yaml")

    assert path.exists()
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["name"] == "smoke"
