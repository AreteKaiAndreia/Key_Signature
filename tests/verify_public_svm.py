"""Verify the hosted upload and recorder paths from an isolated CI browser."""

import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright
from imageio_ffmpeg import get_ffmpeg_exe

APP = "https://musickeyanalyzer.streamlit.app/"
METHODS = {
    "Neural Network": "neural_network",
    "Linear SVM": "linear_svm",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Essentia (Krumhansl–Schmuckler)": "essentia",
    "Soft Voting": "soft_voting",
}
BENCHMARK = json.loads(Path("keyclf/method_benchmarks.json").read_text())


def ffmpeg(*args: str) -> None:
    subprocess.run([get_ffmpeg_exe(), "-nostdin", "-loglevel", "error", *args], check=True)


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


def verify_method_details(frame, label: str) -> None:
    explanation = frame.get_by_role("region", name=f"How {label} works")
    expect(explanation.locator(".studio-method-copy p")).to_have_count(2)
    method = METHODS[label]
    figure = explanation.get_by_role("figure")
    expect(figure).to_be_visible()
    expect(figure).to_have_attribute("data-method", method)
    expect(figure.locator("figcaption")).to_have_text(re.compile(r"\S"))
    metrics = BENCHMARK["methods"][method]
    table = explanation.get_by_role("table", name="Held-out key prediction results")
    for heading in ("Guesses", "Accuracy", "Recall", "Precision"):
        expect(table.get_by_role("columnheader", name=heading, exact=True)).to_have_count(1)
    expect(table.locator("tbody th")).to_have_text(["First exact key", "Top 3 exact keys", "Top 3 key pairs"])
    expected = []
    for name in ("first_exact", "top3_exact", "top3_pairs"):
        row = metrics[name]
        expected.extend(f"{value:.1%}" for value in (row["correct"] / BENCHMARK["clips"], row["recall"], row["precision"]))
    expect(table.locator("td")).to_have_text(expected)


def verify_metric_tooltips(frame) -> None:
    table = frame.get_by_role("table", name="Held-out key prediction results")
    guesses = table.get_by_role("columnheader", name="Guesses", exact=True)
    for label, denominator in (
        ("Accuracy", "all test clips"),
        ("Recall", "test clips labeled with it"),
        ("Precision", "all guesses for it"),
    ):
        header = table.get_by_role("columnheader", name=label, exact=True)
        icon = header.get_by_role("img", name=f"How {label.lower()} is calculated")
        bubble = header.locator('[role="tooltip"]')
        expect(bubble).to_be_hidden()
        icon.hover()
        expect(bubble).to_be_visible()
        expect(bubble).to_contain_text(denominator)
        guesses.hover()
        expect(bubble).to_be_hidden()
        icon.focus()
        expect(bubble).to_be_visible()
        icon.blur()
        expect(bubble).to_be_hidden()



def verify_bundled_fonts(frame) -> None:
    status = frame.locator(".studio-header").evaluate("""async () => {
      await document.fonts.ready;
      return {
        loaded:[...document.fonts].filter(f=>f.status==='loaded').map(f=>f.family.replaceAll('"','').replaceAll("'",'')),
        embedded:[...document.querySelectorAll('style')].some(s=>s.textContent.includes('data:font/woff2;base64,')),
        family:getComputedStyle(document.querySelector('.studio-title')).fontFamily
      };
    }""")
    assert status["embedded"] and "Geist" in status["family"], status
    assert {"Geist", "IBM Plex Mono"} <= set(status["loaded"]), status


def verify_live_readout_persists(frame) -> None:
    """Observe every update without allowing the last prediction to disappear."""
    live = frame.get_by_role("region", name="Live key prediction")
    stability = live.evaluate("""initial => new Promise(resolve => {
        const engine = initial.dataset.engine;
        const first = Number(initial.querySelector("[data-preview-duration]").dataset.previewDuration);
        const seen = new Set([first]);
        let gaps = 0;
        let animation;
        let completed = false;
        const observer = new MutationObserver(inspect);
        const timer = setTimeout(finish, 25000);

        function inspect() {
            const result = document.querySelector(".studio-result");
            const pair = result?.querySelector(".studio-reading");
            if (!pair || result.dataset.engine !== engine || result.querySelectorAll(".studio-alternates tr").length !== 2) {
                gaps++;
            } else {
                const css = getComputedStyle(pair);
                if (css.display === "none" || css.visibility === "hidden" || Number(css.opacity) === 0) gaps++;
                const duration = Number(result.querySelector("[data-preview-duration]")?.dataset.previewDuration);
                if (Number.isFinite(duration)) seen.add(duration);
            }
            if (seen.size >= 3) finish();
        }
        function sampleFrame() {
            inspect();
            if (!completed) animation = requestAnimationFrame(sampleFrame);
        }
        function finish() {
            if (completed) return;
            completed = true;
            observer.disconnect();
            cancelAnimationFrame(animation);
            clearTimeout(timer);
            resolve({gaps, durations:[...seen]});
        }
        observer.observe(document.body, {childList:true, subtree:true});
        animation = requestAnimationFrame(sampleFrame);
    })""")
    assert stability["gaps"] == 0, stability
    assert len(stability["durations"]) >= 3, stability
    print(f"Live result stayed visible through three audio snapshots: {stability['durations']}.", flush=True)


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

        microphone = fixture.with_name("changing-keys.wav")
        shifted = "0.14*(sin(2*PI*369.994*t)+sin(2*PI*466.164*t)+sin(2*PI*554.365*t))"
        progression = f"aevalsrc='if(lt(t,24),{tone},{shifted})':s=44100:d=48"
        ffmpeg("-f", "lavfi", "-i", progression, str(microphone))

        browser = playwright.chromium.launch(channel="chrome", args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-audio-capture={microphone}",
        ])
        page = browser.new_page(viewport={"width": 1280, "height": 840})
        try:
            app_url = os.environ.get("KEYCLF_TEST_APP_URL", APP)
            page.goto(app_url, wait_until="domcontentloaded", timeout=60_000)
            frame = page.frame_locator('iframe[title="streamlitApp"]') if app_url == APP else page
            engine = frame.get_by_role("combobox", name="Prediction engine")
            expect(engine).to_have_value("Neural Network", timeout=60_000)
            verify_bundled_fonts(frame)
            page.set_viewport_size({"width": 390, "height": 844})
            expect(frame.locator('[data-testid="stFileUploaderDropzone"]')).to_be_visible()
            expect(engine).to_be_visible()
            page.screenshot(path=evidence / "phone-upload.png")
            page.set_viewport_size({"width": 1280, "height": 840})
            print("The original Geist and IBM Plex Mono fonts load from bundled data; phone and desktop inputs remain available.", flush=True)
            expect(engine).to_have_attribute("inputmode", "none")
            engine.press("ArrowDown")
            expect(frame.get_by_role("option")).to_have_count(len(METHODS))
            engine.press("x")
            expect(frame.get_by_role("option")).to_have_count(len(METHODS))
            engine.press("Escape")
            expect(engine).to_have_value("Neural Network")
            print("The signed-out engine selector opens with the keyboard and does not accept text.", flush=True)
            verify_metric_tooltips(frame)
            frame.get_by_role("img", name="How precision is calculated").hover()
            page.screenshot(path=evidence / "metric-tooltip.png")
            print("All three benchmark definitions appear on hover and keyboard focus.", flush=True)
            expect(frame.get_by_role("button", name="Analyze", exact=True)).to_have_count(0)
            for clip in (mp3, mp4):
                frame.locator('input[type="file"]').set_input_files(clip)
                for label in METHODS if clip == mp3 else ("Linear SVM", "Soft Voting"):
                    select_engine(frame, label)
                    verify_method_details(frame, label)
                    verify_result(frame, label, known_pair=label in ("Linear SVM", "Soft Voting"))
                    if clip == mp3 and label == "Linear SVM":
                        frame.locator(".studio-method-overview").screenshot(path=evidence / "linear-svm-diagram.png")
                    print(f"{clip.name}, {label}: explanation, diagram, metrics, and three key pairs.", flush=True)
                page.screenshot(path=evidence / f"upload-{clip.suffix[1:]}.png")
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
            for label in METHODS:
                select_engine(frame, label)
                verify_method_details(frame, label)
                verify_result(frame, label, live=True, known_pair=label in ("Linear SVM", "Neural Network", "Soft Voting"))
                if label == "Neural Network":
                    verify_live_readout_persists(frame)
                    page.screenshot(path=evidence / "persistent-live-prediction.png")
                expect(stop).to_be_enabled()
                expect(chart).to_be_visible()
                print(f"{label}: three live pairs while the same recorder is running.", flush=True)
                if label == "Linear SVM":
                    page.screenshot(path=evidence / "recording-live-chroma.png")

            expect(chart.locator('[data-pitch="F#"]')).to_have_attribute(
                "aria-label", re.compile(r"^F#: [1-9]\d?\.\d%$"), timeout=45_000,
            )
            live = frame.get_by_role("region", name="Live key prediction")
            expect(live.locator("[data-preview-duration]")).to_have_attribute(
                "data-preview-duration", re.compile(r"^(?:[3-9]\d|1\d{2})\."), timeout=45_000,
            )
            verify_result(frame, "Soft Voting", live=True)
            print("After the chord changed to F-sharp major, the live result still ranked the accumulated C major / A minor first.", flush=True)

            stop.click()
            verify_result(frame, "Soft Voting")
            page.screenshot(path=evidence / "microphone.png")
            print("Stopped recording: final C major / A minor without clicking Analyze.", flush=True)
            recorder.get_by_role("button", name="Clear recording", exact=True).click()
            expect(frame.locator(".studio-reading")).to_have_count(0)
            recorder.get_by_role("button", name="Record", exact=True).click()
            expect(live.locator("[data-preview-duration]")).to_have_attribute(
                "data-preview-duration", re.compile(r"^[2-6]\."), timeout=45_000,
            )
            stop.click()
            verify_result(frame, "Soft Voting", known_pair=False)
            print("A new take starts the accumulated duration over.", flush=True)
        except Exception:
            page.screenshot(path=evidence / "failure.png")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
