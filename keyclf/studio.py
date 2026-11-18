"""Presentation shared by the local and hosted Streamlit tools."""

from base64 import b64encode
from functools import cache
from pathlib import Path

_STYLE_DIR = Path(__file__).resolve().parent


@cache
def _font_css() -> str:
    """Keep the original fonts available without third-party network requests."""
    directory = _STYLE_DIR / "fonts"
    stylesheet = (directory / "fonts.css").read_text(encoding="utf-8")
    for font in directory.glob("*.woff2"):
        encoded = b64encode(font.read_bytes()).decode("ascii")
        stylesheet = stylesheet.replace(f"url({font.name})", f"url(data:font/woff2;base64,{encoded})")
    return stylesheet


def studio_css() -> str:
    """Inline the current stylesheet and its tokens for this Streamlit run."""
    tokens = (_STYLE_DIR / "tokens.css").read_text(encoding="utf-8")
    stylesheet = (_STYLE_DIR / "console.css").read_text(encoding="utf-8")
    stylesheet = stylesheet.replace('@import url("./tokens.css");', tokens)
    stylesheet = stylesheet.replace('@import url("./fonts/fonts.css");', _font_css())
    return f"<style>{stylesheet}</style>"


STUDIO_HEADER = """
<header class="studio-header">
  <h1 class="studio-title">Key Signature Classifier</h1>
  <p class="studio-intro">Find the musical key of any track or short recording.</p>
</header>
"""
