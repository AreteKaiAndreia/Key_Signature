"""Check that the existing fonts ship intact and need no external requests."""

from pathlib import Path
import re

import pytest

from keyclf.studio import studio_css

pytestmark = pytest.mark.harness


def test_original_font_definitions_skip_css_discovery_without_bulk_font_payload():
    css = studio_css()
    assert "@import" not in css
    assert "fonts.googleapis.com" not in css
    assert "font-family: 'Geist'" in css
    assert "font-family: 'IBM Plex Mono'" in css
    sources = re.findall(r"url\((https://fonts.gstatic.com/[^)]+)\)", css)
    directory = Path(__file__).resolve().parents[1] / "keyclf/fonts"
    expected = {line for line in (directory / "SOURCES.txt").read_text().splitlines()
                if line.startswith("https://fonts.gstatic.com/")}
    assert set(sources) == expected
    assert len(css.encode()) < 40000
    assert ".studio-result" in css and ".studio-benchmark" in css
