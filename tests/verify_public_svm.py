"""Verify the hosted upload and recorder paths from an isolated CI browser."""

import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright

APP = "https://musickeyanalyzer.streamlit.app/"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", *args], check=True)


def verify_result(frame, engine: str, *, live: bool = False, known_pair: bool = True) -> None:
    title = "Live key prediction" if live else "Detected key pair"
    result = frame.get_by_role("region", name=title)
    expect(result).to_have_attribute("data-engine", engine, timeout=45_000)
    pair = result.locator(".studio-reading")
    expect(pair).to_be_visible(timeout=45_000)
    if known_pair:
        expect(pair).to_have_text(re.compile(r"C major.*A minor|A minor.*C major"))
    expect(result.locator(".studio-alternates tbody tr")).to_have_count(2)


def select_engine(frame, label: str) -> None:
    frame.get_by_role("button", name="Open", exact=True).click()
    frame.get_by_role("option", name=label, exact=True).click()
    expect(frame.get_by_role("combobox", name="Prediction engine")).to_have_value(label)


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
            expect(engine).to_have_value("Neural Network", timeout=60_000)
            print("Signed-out visitor starts with Neural Network.", flush=True)
            select_engine(frame, "Linear SVM")
            expect(frame.get_by_role("button", name="Analyze", exact=True)).to_have_count(0)
            for clip in (mp3, mp4):
                frame.locator('input[type="file"]').set_input_files(clip)
                verify_result(frame, "Linear SVM")
                page.screenshot(path=evidence / f"upload-{clip.suffix[1:]}.png")
                print(f"{clip.name}: C major / A minor plus two alternatives.", flush=True)
                frame.get_by_role("button", name=f"Remove {clip.name}").click()
                expect(frame.locator(".studio-reading")).to_have_count(0)

            frame.get_by_text("Record a soundbite", exact=True).click()
            recorder = frame.get_by_role("region", name="Microphone recorder")
            recorder.get_by_role("button", name="Record", exact=True).click()
            chart = recorder.get_by_role("figure", name="Live chroma preview")
            expect(chart).to_be_visible()
            for pitch in ("C", "E", "G"):
                expect(chart.locator(f'[data-pitch="{pitch}"]')).to_have_attribute(
                    "aria-label", re.compile(rf"^{pitch}: [1-9]\d\.\d%$"), timeout=5_000,
                )
            stop = recorder.get_by_role("button", name="Stop recording", exact=True)
            print("Live histogram: C, E, and G bars populated while recording.", flush=True)
            for label in ("Linear SVM", "Neural Network", "Random Forest", "Gradient Boosting", "Essentia (Krumhansl–Schmuckler)"):
                if label != "Linear SVM":
                    select_engine(frame, label)
                verify_result(frame, label, live=True, known_pair=label in ("Linear SVM", "Neural Network"))
                expect(stop).to_be_enabled()
                expect(chart).to_be_visible()
                print(f"{label}: three live pairs while the same recorder is running.", flush=True)
                if label == "Linear SVM":
                    page.screenshot(path=evidence / "recording-live-chroma.png")

            stop.click()
            verify_result(frame, "Essentia (Krumhansl–Schmuckler)")
            page.screenshot(path=evidence / "microphone.png")
            print("Stopped recording: final C major / A minor without clicking Analyze.", flush=True)
            recorder.get_by_role("button", name="Clear recording", exact=True).click()
            expect(frame.locator(".studio-reading")).to_have_count(0)
        except Exception:
            page.screenshot(path=evidence / "failure.png")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
