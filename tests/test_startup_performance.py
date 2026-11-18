"""Verify hard startup budgets and per-invocation profiling without audio data."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark_startup import LIMITS, compare

pytestmark = pytest.mark.harness
ROOT = Path(__file__).resolve().parents[1]


def report(**changes):
    return {**{name: limit / 2 for name, limit in LIMITS.items()},
            "slowest_operations": [], **changes}


@pytest.mark.parametrize("metric", LIMITS)
def test_each_limit_fails_at_the_boundary(metric):
    assert not compare(report(**{metric: LIMITS[metric]}), report())
    assert compare(report(**{metric: LIMITS[metric] - .01}), report())


def test_comparison_prints_real_before_after_and_signed_delta(capsys, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert compare(report(warm_median_ms=750), report(warm_median_ms=1250))
    console = capsys.readouterr().out
    assert all(value in console for value in ("1250.0", "750.0", "-500.0", "3000.0"))
    assert "| warm_median_ms | 1250.0 | 750.0 | -500.0 | 3000 |" in summary.read_text()


def test_baseline_measurement_does_not_emit_a_ci_error(capsys):
    assert not compare(report(operation_max_ms=700), None, enforce=False)
    assert "::error::" not in capsys.readouterr().out


def test_profiler_keeps_worst_individual_call_and_includes_root(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("import time\ndef pause():\n    time.sleep(.04)\npause()\npause()\n")
    destination = tmp_path / "profiles"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts/startup_probe"),
           "KEYCLF_STARTUP_PROFILE_DIR": str(destination), "KEYCLF_STARTUP_PROFILE_SCRIPT": str(target)}
    subprocess.run([sys.executable, "-c",
                    "import runpy,threading,sys; t=threading.Thread(target=runpy.run_path,args=(sys.argv[1],)); t.start(); t.join()",
                    str(target)], env=env, check=True, timeout=10)
    profile, = destination.glob("*.json")
    operations = json.loads(profile.read_text())["operations"]
    pause, = (stats for name, stats in operations.items() if name.endswith("(pause)"))
    root, = (stats for name, stats in operations.items() if name == f"{target}:1(<module>)")
    assert pause["calls"] == 2
    assert pause["max_ms"] >= 35
    assert root["max_ms"] >= pause["max_ms"] + 35
    assert root["max_ms"] >= 75
