"""Exercise the deployed decoder without an operating-system FFmpeg install."""

from io import BytesIO
from pathlib import Path
import subprocess
import wave

import numpy as np
import pytest

from keyclf.cloud_audio import decode_clip, ffmpeg_executable, SAMPLE_RATE

pytestmark = pytest.mark.impl


@pytest.mark.parametrize("extension", ["wav", "mp3", "mp4"])
def test_packaged_decoder_with_empty_system_path(tmp_path, monkeypatch, extension):
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    executable = ffmpeg_executable()
    assert Path(executable).is_absolute()
    assert "imageio_ffmpeg" in executable
    time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio = (8000 * np.sin(2 * np.pi * 440 * time)).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(SAMPLE_RATE)
        recording.writeframes(audio.tobytes())
    source = tmp_path / "source.wav"
    source.write_bytes(buffer.getvalue())
    target = source
    if extension != "wav":
        target = tmp_path / f"clip.{extension}"
        subprocess.run([executable, "-v", "error", "-i", str(source), str(target)],
                       check=True, capture_output=True, timeout=10)
    samples, clipped = decode_clip(target.read_bytes(), target.name)
    assert not clipped
    assert abs(len(samples) - SAMPLE_RATE) < SAMPLE_RATE // 10
    assert np.isfinite(samples).all()
    spectrum = np.abs(np.fft.rfft(samples))
    frequency = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)[spectrum.argmax()]
    assert frequency == pytest.approx(440, abs=2)
