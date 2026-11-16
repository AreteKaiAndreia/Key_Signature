"""Mount the browser microphone recorder and receive its finished clip."""

from __future__ import annotations

import base64
import binascii
from functools import lru_cache
from pathlib import Path

import streamlit as st

from .cloud_audio import MAX_BYTES, MAX_SECONDS
from .keys import PITCH_NAMES

RECORDING_NAMES = {"audio/webm": "recording.webm", "audio/mp4": "recording.mp4",
                   "audio/ogg": "recording.ogg"}
PITCH_BARS = "".join(
    f'<div class="studio-pitch" data-pitch="{pitch}" aria-label="{pitch}: 0.0%">'
    f'<div class="studio-track" aria-hidden="true"><div class="studio-bar" style="--pitch: 0%"></div></div>'
    f'<span>{pitch}</span></div>'
    for pitch in PITCH_NAMES
)
RECORDER_HTML = f"""
<section class="studio-mic" aria-label="Microphone recorder">
  <div class="studio-mic-controls">
    <button type="button" data-record>Record</button>
    <span data-elapsed role="timer" aria-label="Recording duration">00:00</span>
    <button type="button" data-clear hidden>Clear recording</button>
  </div>
  <audio aria-label="Recorded soundbite" controls preload="metadata" hidden></audio>
  <figure class="studio-chart studio-mic-chart" aria-label="Live chroma preview" hidden>
    <figcaption>Live chroma preview</figcaption>
    <div class="studio-pitches">{PITCH_BARS}</div>
  </figure>
  <p class="studio-mic-message" role="status" hidden></p>
</section>
"""


@lru_cache(maxsize=1)
def recorder_component(script: str):
    """Register the recorder using the same style tokens as the result chart."""
    return st.components.v2.component("live_microphone", html=RECORDER_HTML, js=script, isolate_styles=False)


def record_soundbite() -> tuple[bytes, str]:
    """Read a completed recording after its browser preview ends."""
    script = Path(__file__).with_suffix(".js").read_text(encoding="utf-8")
    state = recorder_component(script)(
        key="microphone_recording", data={"maxSeconds": MAX_SECONDS, "maxBytes": MAX_BYTES},
        default={"clip": None}, on_clip_change=lambda: None,
    )
    clip = state.clip
    if clip is None:
        return b"", ""
    try:
        filename = RECORDING_NAMES[clip["mime"]]
        encoded = clip["audio"]
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_BYTES + 2) // 3):
            raise ValueError("Invalid recording size.")
        content = base64.b64decode(encoded, validate=True)
        if 0 < len(content) <= MAX_BYTES:
            return content, filename
    except (KeyError, TypeError, ValueError, binascii.Error):
        pass
    st.error("The recording could not be read. Try recording a shorter soundbite.")
    return b"", ""
