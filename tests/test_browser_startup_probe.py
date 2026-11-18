"""Check that a Cloud-style hidden guest frame cannot count as a painted widget."""

from pathlib import Path
from urllib.parse import quote

import pytest

pytestmark = pytest.mark.impl


def test_widget_paint_waits_for_the_guest_frame_to_be_shown():
    playwright = pytest.importorskip("playwright.sync_api")
    probe = Path(__file__).resolve().parents[1] / "scripts/startup_probe/paint.js"
    with playwright.sync_playwright() as engine:
        browser = engine.chromium.launch(channel="chrome")
        try:
            page = browser.new_page()
            page.add_init_script(path=str(probe))
            page.goto("data:text/html," + quote("""
              <iframe style="display:none" title="guest"></iframe>
              <script>
                const frame=document.querySelector('iframe');
                setTimeout(()=>{
                  frame.contentDocument.open();
                  frame.contentDocument.write('<input role="combobox" value="Neural Network"><div data-testid="stFileUploaderDropzone">Upload</div>');
                  frame.contentDocument.close();
                },50);
                setTimeout(()=>{
                  window.shownAt=Date.now();
                  window.prematurePaint=frame.contentWindow.__startupPaintAt;
                  frame.style.display='block';
                },350);
              </script>"""))
            page.wait_for_function("document.querySelector('iframe').contentWindow.__startupPaintAt", timeout=5000)
            result = page.evaluate("""() => ({
              premature:!!window.prematurePaint, shown:window.shownAt,
              paint:document.querySelector('iframe').contentWindow.__startupPaintAt
            })""")
            assert not result["premature"]
            assert 0 <= result["paint"] - result["shown"] < 1000
        finally:
            browser.close()
