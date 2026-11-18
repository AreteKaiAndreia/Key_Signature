"""Check that the existing fonts ship intact and need no external requests."""

from base64 import b64decode
from pathlib import Path
import re

import pytest

from keyclf.studio import studio_css

pytestmark = pytest.mark.harness


def test_original_font_faces_are_inlined_without_changing_the_styles():
    css = studio_css()
    assert "@import" not in css
    assert "fonts.googleapis.com" not in css
    assert "font-family: 'Geist'" in css
    assert "font-family: 'IBM Plex Mono'" in css
    sources = re.findall(r"url\(data:font/woff2;base64,([A-Za-z0-9+/=]+)\)", css)
    directory = Path(__file__).resolve().parents[1] / "keyclf/fonts"
    assert {b64decode(source) for source in sources} == {font.read_bytes() for font in directory.glob("*.woff2")}
    assert ".studio-result" in css and ".studio-benchmark" in css
