"""Verify the hosted upload and recorder paths from an isolated CI browser."""

import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright

APP = "https://musickeyanalyzer.streamlit.app/"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", *args], check=True)


def verify_result(frame) -> None:
    result = frame.get_by_role("region", name="Detected key pair")
    expect(result.get_by_text("higher SVM ranking score in pair", exact=False)).to_be_visible(timeout=45_000)
    expect(result.locator(".studio-reading")).to_have_text(re.compile(r"C major.*A minor|A minor.*C major"))
    expect(result.locator(".studio-alternates tbody tr")).to_have_count(2)


def main() -> None:
    evidence = Path("reports/public-svm-smoke")
    evidence.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as directory, sync_playwright() as playwright:
        fixture = Path(directory) / "c-major.wav"
        tone = "0.14*(sin(2*PI*261.626*t)+sin(2*PI*329.628*t)+sin(2*PI*391.995*t))"
        ffmpeg("-f", "lavfi", "-i", f"aevalsrc={tone}:s=44100:d=8", str(fixture))
        mp3, mp4 = fixture.with_suffix(".mp3"), fixture.with_suffix(".mp4")
        ffmpeg("-i", str(fixture), str(mp3))
        ffmpeg("-f", "lavfi", "-i", "color=s=160x90", "-i", str(fixture), "-shortest", str(mp4))

        browser = playwright.chromium.launch(channel="chrome", args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-audio-capture={fixture}",
        ])
        page = browser.new_page(viewport={"width": 1280, "height": 840})
        try:
            page.goto(APP, wait_until="domcontentloaded", timeout=60_000)
            frame = page.frame_locator('iframe[title="streamlitApp"]')
            engine = frame.get_by_role("combobox", name="Prediction engine")
            expect(engine).to_have_value("Custom ML (open-trained)", timeout=60_000)
            print("Signed-out visitor starts with Custom ML.", flush=True)
            frame.get_by_role("button", name="Open", exact=True).click()
            frame.get_by_role("option", name="Linear SVM", exact=True).click()
            expect(engine).to_have_value("Linear SVM")

            analyze = frame.get_by_role("button", name="Analyze", exact=True)
            for clip in (mp3, mp4):
                frame.locator('input[type="file"]').set_input_files(clip)
                expect(analyze).to_be_enabled(timeout=15_000)
                analyze.click()
                verify_result(frame)
                page.screenshot(path=evidence / f"upload-{clip.suffix[1:]}.png")
                print(f"{clip.name}: C major / A minor plus two alternatives.", flush=True)
                frame.get_by_role("button", name=f"Remove {clip.name}").click()
                expect(analyze).to_be_disabled()

            frame.get_by_text("Record a soundbite", exact=True).click()
            recorder = frame.get_by_test_id("stAudioInput")
            recorder.get_by_role("button", name="Record", exact=True).click()
            expect(recorder.get_by_test_id("stAudioInputWaveformTimeCode")).to_have_text("00:05", timeout=15_000)
            recorder.get_by_role("button", name="Stop recording", exact=True).click()
            expect(analyze).to_be_enabled(timeout=20_000)
            analyze.click()
            verify_result(frame)
            page.screenshot(path=evidence / "microphone.png")
            print("Virtual microphone: captured and analyzed C major / A minor plus two alternatives.", flush=True)
        except Exception:
            page.screenshot(path=evidence / "failure.png")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
