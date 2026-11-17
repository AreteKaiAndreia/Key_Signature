"""Mount the browser microphone recorder and receive its finished clip."""

from __future__ import annotations

import base64
import re
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import streamlit as st

from .cloud_audio import MAX_BYTES, MAX_SECONDS, SAMPLE_RATE
from .keys import PITCH_NAMES

PREVIEW_SECONDS = 3
WAV_HEADER_BYTES = 44
MAX_FRAMES = SAMPLE_RATE * MAX_SECONDS
PREVIEW_PACKET_FRAMES = 2 * SAMPLE_RATE * PREVIEW_SECONDS
PREVIEW_BYTES = WAV_HEADER_BYTES + 2 * PREVIEW_PACKET_FRAMES


@dataclass(frozen=True)
class Recording:
    """One microphone take and an optional current preview."""

    content: bytes = b""
    active: bool = False
    take_id: str = ""
    preview: bytes = b""
    token: str = ""
    preview_offset: int = 0


@dataclass
class LiveAudio:
    """The acknowledged PCM prefix of the current microphone take."""

    pcm: bytearray = field(default_factory=bytearray)

    @property
    def frames(self) -> int:
        return len(self.pcm) // 2

    def append(self, offset: int, packet: bytes) -> None:
        """Append new PCM, ignoring gaps, duplicate frames, and invalid packets."""
        if not packet or not 0 <= offset <= self.frames:
            return
        try:
            with wave.open(BytesIO(packet), "rb") as reader:
                params = reader.getparams()
                if (params.nchannels, params.sampwidth, params.framerate) != (1, 2, SAMPLE_RATE):
                    return
                frames = params.nframes
                if frames > PREVIEW_PACKET_FRAMES or offset + frames > MAX_FRAMES:
                    return
                samples = reader.readframes(frames)
        except (wave.Error, EOFError, ValueError):
            return
        if len(samples) != 2 * frames:
            return
        start = 2 * offset
        overlap = min(len(samples), len(self.pcm) - start)
        if self.pcm[start : start + overlap] == samples[:overlap]:
            self.pcm.extend(samples[overlap:])

    def wav(self) -> bytes:
        """Build a WAV snapshot for the same scoring path as the finished take."""
        out = BytesIO()
        with wave.open(out, "wb") as writer:
            writer.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            writer.writeframes(self.pcm)
        return out.getvalue()


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


def decode_payload(encoded, limit: int) -> bytes:
    """Reject non-WAV or oversized messages before inference."""
    if not isinstance(encoded, str) or len(encoded) > 4 * ((limit + 2) // 3):
        raise ValueError("Invalid recording size.")
    content = base64.b64decode(encoded, validate=True)
    if (not WAV_HEADER_BYTES <= len(content) <= limit or content[:4] != b"RIFF" or content[8:12] != b"WAVE"):
        raise ValueError("Invalid recording format.")
    return content


def record_soundbite() -> Recording:
    """Read the current microphone status, transient preview, or finished WAV."""
    script = Path(__file__).with_name("live_microphone.js").read_text(encoding="utf-8")
    state = recorder_component(script)(
        key="microphone_take",
        data={"maxSeconds": MAX_SECONDS, "maxBytes": MAX_BYTES, "sampleRate": SAMPLE_RATE,
              "previewSeconds": PREVIEW_SECONDS, "previewPacketFrames": PREVIEW_PACKET_FRAMES},
        default={"take": None}, on_take_change=lambda: None, on_preview_change=lambda: None,
    )
    take = state.take
    if not isinstance(take, dict):
        return Recording()
    if take.get("active") is True:
        take_id = take.get("id", "")
        message = state.preview
        if isinstance(message, dict) and message.get("id") == take.get("id"):
            token = message.get("token")
            if isinstance(token, str) and re.fullmatch(r"[a-f0-9-]{36}/[1-9]\d{0,2}", token):
                offset = message.get("offset")
                if type(offset) is int and 0 <= offset < MAX_FRAMES:
                    try:
                        audio = decode_payload(message.get("audio"), PREVIEW_BYTES)
                        return Recording(active=True, take_id=take_id, preview=audio, token=token,
                                         preview_offset=offset)
                    except ValueError:
                        pass
                return Recording(active=True, take_id=take_id, token=token)
        return Recording(active=True, take_id=take_id)
    if take.get("audio") is not None:
        try:
            return Recording(content=decode_payload(take["audio"], MAX_BYTES))
        except ValueError:
            st.error("The recording could not be read. Try recording a shorter soundbite.")
    return Recording()
