from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fets27_challenge import cli
from fets27_challenge.cli import non_negative_int


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("2", 2), ("16", 16)])
def test_non_negative_int_accepts_worker_counts(value, expected):
    assert non_negative_int(value) == expected


def test_non_negative_int_rejects_negative_worker_count():
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative integer"):
        non_negative_int("-1")


def test_run_local_forwards_worker_count(monkeypatch):
    captured = {}

    def fake_run_challenge(**kwargs):
        captured.update(kwargs)
        return Path("summary.json"), Path("summary.csv"), []

    monkeypatch.setattr(cli, "run_challenge", fake_run_challenge)

    cli.main(
        [
            "run-local",
            "--data-root",
            "data",
            "--workspace",
            "workspace",
            "--output-dir",
            "outputs",
            "--data-loader-workers",
            "0",
        ]
    )

    assert captured["data_loader_workers"] == 0
