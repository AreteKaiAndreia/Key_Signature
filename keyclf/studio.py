"""Presentation shared by the local and hosted Streamlit tools."""

from pathlib import Path

_STYLE_DIR = Path(__file__).resolve().parent
# Streamlit inserts the style in the page. Inline the local token file so
# browsers do not have to resolve its path through Streamlit's URL routing.
_STYLESHEET = (_STYLE_DIR / "console.css").read_text(encoding="utf-8").replace(
    '@import url("./tokens.css");', (_STYLE_DIR / "tokens.css").read_text(encoding="utf-8")
)
STUDIO_CSS = f"<style>{_STYLESHEET}</style>"

STUDIO_HEADER = """
<header class="studio-header">
  <h1 class="studio-title">Key Signature Classifier</h1>
  <p class="studio-intro">Find the musical key of any track or short recording.</p>
</header>
"""
