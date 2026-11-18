"""Measure URL-to-widget rendering on fresh and warm Streamlit processes."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

PROBE = Path(__file__).resolve().parent / "startup_probe"
LIMITS = {"warm_median_ms": 3000, "cold_max_ms": 30000, "operation_max_ms": 500}


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


class AppProcess:
    """Launch Streamlit on demand for the first browser request."""

    def __init__(self, source: Path, profile: Path, log: Path):
        self.source, self.profile, self.port = source, profile, free_port()
        self.url = f"http://127.0.0.1:{self.port}/"
        self.process = None
        self.lock = threading.Lock()
        self.log = log.open("w")

    def start(self):
        with self.lock:
            if self.process is None:
                env = os.environ.copy()
                env.update(
                    KEYCLF_STARTUP_PROFILE_DIR=str(self.profile),
                    KEYCLF_STARTUP_PROFILE_SCRIPT=str(self.source / "streamlit_app.py"),
                    PYTHONPATH=os.pathsep.join((str(PROBE), str(self.source))),
                    STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
                    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                )
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                     "--server.address=127.0.0.1", f"--server.port={self.port}", "--server.headless=true"],
                    cwd=self.source, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                )
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Streamlit exited with {self.process.returncode}")
                try:
                    with urlopen(self.url + "_stcore/health", timeout=.3) as response:
                        if response.status == 200:
                            return
                except (URLError, TimeoutError):
                    time.sleep(.025)
            raise TimeoutError("Streamlit did not start within 40 seconds")

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


class StartupGateway(ThreadingHTTPServer):
    """Start the fresh server in response to the browser's first URL request."""

    def __init__(self, app: AppProcess):
        self.app = app
        super().__init__(("127.0.0.1", 0), StartupRequest)


class StartupRequest(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self.server.app.start()
            self.send_response(302)
            self.send_header("Location", self.server.app.url)
            self.end_headers()
        except Exception as error:
            self.send_error(503, str(error))

    def log_message(self, *_):
        pass


def browser_load(browser, url: str, *, hosted: bool = False) -> dict:
    from playwright.sync_api import expect

    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    frame = page.frame_locator('iframe[title="streamlitApp"]') if hosted else page
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script("""window.__startupTasks=[];
      new PerformanceObserver(list => window.__startupTasks.push(...list.getEntries().map(e=>({start:e.startTime,duration:e.duration}))))
        .observe({type:'longtask', buffered:true});
      function ready() {
        const engine=document.querySelector('input[role="combobox"]');
        const upload=document.querySelector('[data-testid="stFileUploaderDropzone"]');
        return engine?.value === 'Neural Network' && engine.getClientRects().length && upload?.getClientRects().length;
      }
      function watchPaint() {
        if (!ready()) { requestAnimationFrame(watchPaint); return; }
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (!ready()) { requestAnimationFrame(watchPaint); return; }
          window.__startupPaintAt=performance.timeOrigin+performance.now();
          window.__startupPaintEnd=performance.now();
        }));
      }
      requestAnimationFrame(watchPaint);
    """)
    start_epoch_ms = time.time_ns() / 1e6
    start = time.perf_counter()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        widget = frame.get_by_role("combobox", name="Prediction engine")
        expect(widget).to_have_value("Neural Network", timeout=45000)
        expect(frame.locator('[data-testid="stFileUploaderDropzone"]')).to_be_visible()
        timing = widget.evaluate("""async () => {
          const deadline=Date.now()+5000;
          while (!window.__startupPaintAt) {
            if (Date.now()>deadline) throw new Error('Widget paint marker missing after visible controls');
            await new Promise(resolve=>setTimeout(resolve,16));
          }
          return {
          paint_epoch_ms:window.__startupPaintAt,
          navigation:performance.getEntriesByType('navigation').map(e=>({duration:e.duration,domContentLoaded:e.domContentLoadedEventEnd})),
          longtask_max_ms:Math.max(0,...window.__startupTasks.filter(e=>e.start<window.__startupPaintEnd).map(e=>e.duration))
        }}""")
        observed = 1000 * (time.perf_counter() - start)
        paint_at = timing.pop("paint_epoch_ms")
        elapsed = paint_at - start_epoch_ms
        if not 0 <= elapsed <= observed + 50:
            raise RuntimeError(f"Browser/Python clocks disagree: paint={elapsed}, observed={observed}")
        timing["longtask_max_ms"] = max(timing["longtask_max_ms"], page.evaluate(
            "end => Math.max(0,...window.__startupTasks.filter(e=>performance.timeOrigin+e.start<end).map(e=>e.duration))", paint_at))
        # Validate the same page content without shortening the measured load path.
        expect(frame.get_by_role("heading", name="How Neural Network works")).to_be_visible()
        expect(frame.locator(".studio-method-diagram")).to_be_visible()
        expect(frame.get_by_role("table", name="Held-out key prediction results")).to_be_visible()
        widget.press("ArrowDown")
        expect(frame.get_by_role("option")).to_have_count(6)
        widget.press("Escape")
        if errors:
            raise AssertionError(errors)
        return {"load_ms": elapsed, "observed_ready_ms": observed, **timing}
    finally:
        context.close()


def run(source: Path, output: Path, cold_runs: int, warm_runs: int):
    from playwright.sync_api import sync_playwright

    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = output.with_suffix("")
    evidence.mkdir(exist_ok=True)
    if list(evidence.glob("cold-*-profile/*.json")):
        raise FileExistsError(f"Use a fresh output path to avoid mixing startup runs: {evidence}")
    cold, warm = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        browser_version = browser.version
        try:
            for index in range(cold_runs):
                profile = evidence / f"cold-{index}-profile"
                app = AppProcess(source, profile, evidence / f"cold-{index}.log")
                gateway = StartupGateway(app)
                thread = threading.Thread(target=gateway.serve_forever, daemon=True)
                thread.start()
                try:
                    cold.append(browser_load(browser, f"http://127.0.0.1:{gateway.server_port}/"))
                    print(f"cold {index + 1}/{cold_runs}: {cold[-1]['load_ms']:.1f} ms", flush=True)
                    if index == cold_runs - 1:
                        for run_index in range(warm_runs):
                            warm.append(browser_load(browser, app.url))
                            print(f"warm {run_index + 1}/{warm_runs}: {warm[-1]['load_ms']:.1f} ms", flush=True)
                finally:
                    gateway.shutdown()
                    gateway.server_close()
                    app.close()
        finally:
            browser.close()

    profiles = [json.loads(path.read_text()) for path in evidence.glob("cold-*-profile/*.json")]
    if len(profiles) < cold_runs + warm_runs:
        raise RuntimeError(f"Missing server startup profiles: {len(profiles)} recorded")
    operations = sorted(
        [{"operation": name, **stats} for profile in profiles for name, stats in profile["operations"].items()],
        key=lambda row: row["max_ms"], reverse=True,
    )
    report = {
        "measurement": "Browser URL hit to painted engine selector and upload control",
        "cold_definition": "Fresh Python process, launched by the first browser URL request",
        "profile_scope": "All app-script calls, including imports and nested calls, through initial page construction; browser long tasks through widget paint",
        "browser_version": browser_version, "python_version": sys.version,
        "source": str(source), "cold": cold, "warm": warm,
        "warm_median_ms": statistics.median(row["load_ms"] for row in warm),
        "cold_max_ms": max(row["load_ms"] for row in cold),
        "server_operation_max_ms": operations[0]["max_ms"],
        "browser_operation_max_ms": max(row["longtask_max_ms"] for row in [*cold, *warm]),
        "slowest_operations": operations[:30],
    }
    report["operation_max_ms"] = max(report["server_operation_max_ms"], report["browser_operation_max_ms"])
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare(report: dict, before: dict | None, *, enforce: bool = True) -> bool:
    failed = False
    print("\nURL-to-widget startup benchmark")
    print(f"{'Metric':24} {'Before':>12} {'After':>12} {'Delta':>12} {'Limit':>12}")
    summary = ["### URL-to-widget startup", "", "All values are milliseconds. Negative deltas are faster.", "",
               "| Metric | Before | After | Delta | Must be below |", "|---|---:|---:|---:|---:|"]
    for name, limit in LIMITS.items():
        value = report[name]
        previous = before[name] if before else None
        delta = f"{value - previous:+.1f}" if previous is not None else "n/a"
        display = f"{previous:.1f}" if previous is not None else "n/a"
        print(f"{name:24} {display:>12} {value:12.1f} {delta:>12} {limit:12.1f} ms")
        summary.append(f"| {name} | {display} | {value:.1f} | {delta} | {limit} |")
        if value >= limit:
            prefix = "::error::" if enforce else "Baseline outside budget: "
            print(f"{prefix}{name} must stay below {limit} ms, measured {value:.1f} ms")
            failed = True
    print("\nSlowest server calls (including nested work)")
    for row in report["slowest_operations"][:8]:
        print(f"{row['max_ms']:9.1f} ms inclusive, {row['max_self_ms']:9.1f} ms self  {row['operation']}")
    if enforce and os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write("\n".join(summary) + "\n")
    return not failed


def run_hosted(url: str, output: Path, runs: int, before: dict | None = None, *, enforce: bool = True) -> None:
    """Measure a reachable hosted instance without asserting it was restarted."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        try:
            samples = []
            for index in range(runs):
                try:
                    samples.append(browser_load(browser, url, hosted=True))
                except Exception as error:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps({"url": url, "samples": samples, "failure": str(error)}, indent=2) + "\n")
                    raise
                print(f"hosted {index + 1}/{runs}: {samples[-1]['load_ms']:.1f} ms", flush=True)
        finally:
            browser.close()
    report = {"url": url, "instance_state": "Not forced cold; server state must be recorded separately",
              "measurement": "Browser URL hit to painted engine selector and upload control, including Cloud iframe",
              "samples": samples, "median_ms": statistics.median(row["load_ms"] for row in samples),
              "browser_operation_max_ms": max(row["longtask_max_ms"] for row in samples)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    failed = False
    for name, limit in (("median_ms", 3000), ("browser_operation_max_ms", 500)):
        value = report[name]
        previous = before.get(name) if before else None
        display = f"{previous:.1f}" if previous is not None else "n/a"
        delta = f"{value - previous:+.1f}" if previous is not None else "n/a"
        line = f"Hosted {name}: before {display}, after {value:.1f}, delta {delta} ms; limit < {limit} ms"
        print(line, flush=True)
        if enforce and os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
                stream.write(line + "\n\n")
        failed |= value >= limit
    if failed and enforce:
        raise SystemExit("Hosted startup exceeded a warm/browser budget; inspect the saved report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--warm-runs", type=int, default=7)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--url", help="Measure the hosted Cloud iframe; does not force a cold server")
    args = parser.parse_args()
    if min(args.cold_runs, args.warm_runs) < 1:
        parser.error("At least one cold and warm run is required")
    if args.url:
        before = json.loads(args.compare.read_text()) if args.compare else None
        run_hosted(args.url, args.output.resolve(), args.warm_runs, before, enforce=not args.report_only)
        return
    report = run(args.source.resolve(), args.output.resolve(), args.cold_runs, args.warm_runs)
    before = json.loads(args.compare.read_text()) if args.compare else None
    passed = compare(report, before, enforce=not args.report_only)
    if not passed and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
