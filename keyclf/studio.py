"""Presentation shared by the local and hosted Streamlit tools."""

STUDIO_CSS = """
<style>
:root {
  --studio-ink: #f4f5f1;
  --studio-muted: #aeb6b2;
  --studio-accent: #d7f99b;
  --studio-line: #303935;
}
.stMainBlockContainer { max-width: 45rem; padding-top: 2.75rem; padding-bottom: 2rem; }
[data-testid="stHeader"] { background: transparent; }
.studio-kicker {
  font-size: .69rem; font-weight: 750; letter-spacing: .15em;
  text-transform: uppercase; color: var(--studio-accent);
  display: flex; align-items: center; gap: .65rem;
}
.studio-kicker span {
  display: inline-block; width: .45rem; height: .45rem;
  border-radius: 50%; background: var(--studio-accent);
}
.studio-title {
  color: var(--studio-ink); margin: .4rem 0 .24rem; padding: 0;
  font-size: clamp(1.9rem, 5.4vw, 2.55rem); line-height: 1.12; letter-spacing: -.045em;
}
.studio-intro { color: var(--studio-muted); margin: 0 0 .25rem; font-size: .95rem; }
.studio-upload-title { color: var(--studio-ink); font-size: 1.06rem; font-weight: 700; margin: 0; }
.studio-upload-hint { color: var(--studio-muted); font-size: .82rem; margin: .2rem 0 0; }
.st-key-track_input [data-testid="stFileUploaderDropzone"] {
  background: #1b2520; border: 1px dashed #566d59; border-radius: .8rem;
  padding: .85rem; min-height: 4.2rem;
}
.st-key-track_input [data-testid="stFileUploaderDropzone"] button {
  color: #101710; background: var(--studio-accent);
  border: 1px solid var(--studio-accent); font-weight: 700;
}
.st-key-track_input [data-testid="stFileUploaderDropzone"] button:hover { background: #e6ffb9; }
.studio-result {
  padding: 1.05rem 1.3rem; border: 1px solid #435447; border-radius: .95rem;
  background: linear-gradient(112deg, #233127, #19201e);
  overflow-wrap: anywhere;
}
.studio-result p { margin: 0; }
.studio-result h2 {
  font-size: clamp(2.3rem, 9vw, 3.7rem); color: var(--studio-ink); line-height: 1.04;
  letter-spacing: -.055em; margin: .25rem 0 .4rem; padding: 0;
}
.studio-confidence { color: var(--studio-muted); font-size: .83rem; }
.studio-confidence strong { color: var(--studio-accent); }
.studio-section {
  color: var(--studio-muted); font-size: .72rem; letter-spacing: .11em;
  font-weight: 720; text-transform: uppercase; margin: .15rem 0 0;
}
@media (max-width: 600px) {
  .stMainBlockContainer { padding: 2.6rem 1rem 1.3rem; }
  .studio-result { padding: .95rem 1rem; }
  .st-key-track_input [data-testid="stFileUploaderDropzone"] { padding: .6rem; }
}
</style>
"""

STUDIO_HEADER = """
<header>
  <div class="studio-kicker"><span aria-hidden="true"></span> Audio analysis</div>
  <h1 class="studio-title">Key Signature Classifier</h1>
  <p class="studio-intro">Identify the musical key of a track or soundbite.</p>
</header>
"""
