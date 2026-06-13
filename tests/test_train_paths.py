"""Tests for training path resolution."""

from adaptive_roi_codec.train import _s3_mount_available


def test_s3_mount_unavailable_when_connector_id_empty() -> None:
    assert _s3_mount_available("") is False


def test_s3_mount_unavailable_on_local_machine() -> None:
    assert _s3_mount_available("fake-connector-id") is False
