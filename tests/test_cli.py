"""Smoke-test the CLI end to end through ``deltatrace.cli.main``."""

from __future__ import annotations

import pandas as pd

from conftest import build_versions

from deltatrace.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    _, versions, expected = build_versions()
    root = str(tmp_path / "store")
    f1 = tmp_path / "v1.csv"
    f2 = tmp_path / "v2.csv"
    versions[0][0].to_csv(f1, index=False)
    versions[1][0].to_csv(f2, index=False)

    assert main(["init", root, "--primary-key", "id"]) == 0
    assert main(["commit", root, str(f1), "-m", "load"]) == 0
    assert main(["commit", root, str(f2), "-m", "more"]) == 0
    assert main(["log", root]) == 0

    out = tmp_path / "out.csv"
    assert main(["checkout", root, "--version", "2", "--out", str(out)]) == 0
    got = pd.read_csv(out)
    assert sorted(got["id"].tolist()) == sorted(expected[2]["id"].tolist())

    assert main(["diff", root, "1", "2"]) == 0
    assert main(["snapshot", root, "--version", "2"]) == 0

    captured = capsys.readouterr().out
    assert "v2" in captured
