"""Build a resumable, local chroma pilot from the Sheet Sage notation release."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .features import DEFAULT_CONFIG, config_signature, histogram_from_audio
from .fmak import FMA_METADATA, HISTOGRAM, atomic_bytes, atomic_json, balanced_pick, digest
from .keys import key_name
from .rangezip import open_member, open_zip

ROOT = Path("data/work/hooktheory")
FMA_CACHE = Path("data/work/fmak-pilot/features.parquet")
URL = "https://raw.githubusercontent.com/chrisdonahue/sheetsage-data/main/hooktheory/Hooktheory.json.gz"
SHA256 = "917b7cd58f5f4e07d6c36acf7bfad958c99ee05472dab3555399141094698e0c"
VERSION = "sheetsage-fixed-synth-v1"
MAJOR = (2, 2, 1, 2, 2, 2)
MINOR = ((2, 1, 2, 2, 1, 2), (2, 1, 2, 2, 1, 3), (2, 1, 2, 2, 2, 2))
SAMPLE_RATE = DEFAULT_CONFIG.sample_rate
MAX_BEATS = 120


def normalized_name(name: str) -> str:
    """Normalize titles and artists for conservative known-overlap screening."""
    name = unicodedata.normalize("NFKD", name.casefold().strip())
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join(re.findall(r"[^\W_]+", name)).removeprefix("the ")


def identity(artist: str, song: str) -> tuple[str, str]:
    """Return normalized artist and title without common version suffixes."""
    song = re.sub(r"\s*[\[(]\s*(?:feat\.?|ft\.?|live\b|remaster|remix|radio edit|version\b).*?[\])]", "", song, flags=re.I)
    song = re.split(r"\s+-\s+(?:remaster|remix|live\b|radio edit)", song, maxsplit=1, flags=re.I)[0]
    return normalized_name(artist), normalized_name(song)


def annotated_key(annotations: dict) -> int:
    """Map a beat-zero, unchanging major/minor key to the canonical class."""
    keys = annotations["keys"]
    if len(keys) != 1 or keys[0]["beat"] != 0:
        raise ValueError("not one global key")
    tonic = keys[0]["tonic_pitch_class"]
    steps = tuple(keys[0]["scale_degree_intervals"])
    if not isinstance(tonic, int) or not 0 <= tonic < 12 or steps not in (MAJOR, *MINOR):
        raise ValueError("unsupported key")
    return tonic + (0 if steps == MAJOR else 12)


def note_events(annotations: dict) -> tuple[float, list[tuple[float, float, int, float]]]:
    """Convert absolute annotations to MIDI events; never inspect the key."""
    beats = float(annotations["num_beats"])
    if not np.isfinite(beats) or not 4 <= beats <= MAX_BEATS:
        raise ValueError("unsupported duration")
    events = []
    for name in ("melody", "harmony"):
        if not annotations.get(name) or len(annotations[name]) > 2000:
            raise ValueError("missing or oversized musical part")
        for row in annotations[name]:
            start, end = float(row["onset"]), float(row["offset"])
            if not (0 <= start < end <= beats):
                raise ValueError("invalid event timing")
            if name == "melody":
                pc, octave = row["pitch_class"], row["octave"]
                if not 0 <= pc < 12 or not -3 <= octave <= 3:
                    raise ValueError("invalid melody pitch")
                events.append((start, end, 72 + 12 * octave + pc, .20))
            else:
                root = row["root_pitch_class"]
                steps = row["root_position_intervals"]
                if not 0 <= root < 12 or len(steps) > 6 or any(not isinstance(s, int) or not 1 <= s <= 12 for s in steps):
                    raise ValueError("invalid chord")
                tones = np.r_[0, np.cumsum(steps, dtype=int)]
                inversion = row.get("inversion", 0)
                if not isinstance(inversion, int) or not 0 <= inversion < len(tones):
                    raise ValueError("invalid chord inversion")
                for i, semitones in enumerate(tones):
                    events.append((start, end, 48 + root + int(semitones) + (12 if i < inversion else 0),
                                   .30 / len(tones)))
    return beats, events


def render_audio(beats: float, events: list[tuple[float, float, int, float]]) -> np.ndarray:
    """Render at 120 BPM with a fixed additive synth and short note ramps."""
    seconds_per_beat = .5
    audio = np.zeros(round(beats * seconds_per_beat * SAMPLE_RATE), dtype=np.float32)
    for start, end, note, gain in events:
        a, z = (round(b * seconds_per_beat * SAMPLE_RATE) for b in (start, end))
        n = z - a
        if n < 2:
            continue
        angle = np.arange(n, dtype=np.float64) * (2 * np.pi * 440 * 2**((note - 69) / 12) / SAMPLE_RATE)
        wave = np.sin(angle) + .22 * np.sin(2 * angle) + .09 * np.sin(3 * angle)
        ramp = min(n // 2, round(.012 * SAMPLE_RATE))
        envelope = np.linspace(0, 1, ramp)
        wave[:ramp] *= envelope
        wave[-ramp:] *= envelope[::-1]
        audio[a:z] += gain * wave
    if not np.any(audio):
        raise ValueError("rendered excerpt is silent")
    return audio * (.8 / np.max(np.abs(audio)))


def acquire_release(path: Path) -> dict:
    """Verify the compressed release before trusting or expanding its JSON."""
    if not path.exists():
        with requests.get(URL, stream=True, timeout=(12, 45)) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(1024**2):
                size += len(chunk)
                if size > 26_000_000:
                    raise ValueError("notation download exceeds size limit")
                chunks.append(chunk)
        content = b"".join(chunks)
    else:
        if path.stat().st_size > 26_000_000:
            raise ValueError("notation file exceeds size limit")
        content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Sheet Sage release differs from its published digest")
    if not path.exists():
        atomic_bytes(path, content)
    # The verified archive is a pretty-printed object that expands to ~309 MB.
    # Stream its top-level records and keep only the fields used by this pilot.
    result = {}
    size = 0
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as member:
        first = member.readline()
        if first.strip() != b"{":
            raise ValueError("unexpected Sheet Sage object")
        record = []
        for line in member:
            size += len(line)
            if size > 330_000_000:
                raise ValueError("expanded notation release exceeds size limit")
            if line == b"}\n" or line == b"}":
                break
            record.append(line)
            if line.rstrip() not in (b"  },", b"  }", b"    },", b"    }"):
                continue
            # A top-level value has the same indent as its opening key.
            indent = len(record[0]) - len(record[0].lstrip())
            if len(line) - len(line.lstrip()) != indent:
                continue
            obj = json.loads(b"{" + b"".join(record).rstrip().removesuffix(b",") + b"}")
            k, v = next(iter(obj.items()))
            slim = {name: v[name] for name in ("split", "hooktheory", "youtube")}
            if v["split"] == "TRAIN":
                slim["annotations"] = v["annotations"]
            result[k] = slim
            record.clear()
        if record or not result:
            raise ValueError("incomplete Sheet Sage object")
    return result


def fma_identities(cache: Path, work: Path) -> set[tuple[str, str]]:
    """Read FMA test artist/title identities from the official metadata ZIP."""
    table = pd.read_parquet(cache, columns=["track_id", "split"])
    ids = sorted(int(t.removeprefix("fmak-")) for t in table[table.split == "test"].track_id)
    target = work / "inputs" / "fmak_test_identities.json"
    if target.exists():
        prior = json.loads(target.read_text())
        if prior["ids"] == ids:
            return {tuple(pair) for pair in prior["identities"]}
    requested = set(ids)
    found = {}
    with requests.Session() as session:
        stream, archive = open_zip(FMA_METADATA, session)
        with stream, archive, open_member(stream, archive, archive.getinfo("fma_metadata/tracks.csv"),
                                           max_size=300 * 1024**2) as member:
            csv.field_size_limit(8 * 1024**2)
            rows = csv.reader(io.TextIOWrapper(member, encoding="utf-8-sig", newline=""))
            columns = list(zip(next(rows), next(rows), strict=True))
            a, t = columns.index(("artist", "name")), columns.index(("track", "title"))
            for row in rows:
                if row and row[0].isdigit() and int(row[0]) in requested:
                    found[int(row[0])] = identity(row[a], row[t])
    if set(found) != requested or not all(all(pair) for pair in found.values()):
        raise ValueError("missing FMA test recording identities")
    atomic_json(target, {"ids": ids, "identities": list(found.values()), "source": FMA_METADATA})
    return set(found.values())


def candidate_rows(data: dict, excluded: set[tuple[str, str]]) -> tuple[list[dict], Counter]:
    """Remove nontraining tracks, known held-out matches, and duplicate songs."""
    def pair(row):
        return identity(row["hooktheory"]["artist"], row["hooktheory"]["song"])
    excluded = excluded | {pair(r) for r in data.values() if r["split"] != "TRAIN"}
    videos = {r["youtube"]["id"] for r in data.values() if r["split"] != "TRAIN" and r.get("youtube")}
    seen_names, seen_videos, candidates, audit = set(), set(), [], Counter()
    for record_id, row in sorted(data.items()):
        if row["split"] != "TRAIN":
            audit["not_train"] += 1
            continue
        song_pair = pair(row)
        video = (row.get("youtube") or {}).get("id")
        if not all(song_pair) or song_pair in excluded or (video and video in videos):
            audit["excluded_identity"] += 1
            continue
        if song_pair in seen_names or (video and video in seen_videos):
            audit["duplicate"] += 1
            continue
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", record_id):
                raise ValueError("unexpected record id")
            label = annotated_key(row["annotations"])
            note_events(row["annotations"])
        except (KeyError, ValueError, OverflowError, TypeError):
            audit["unsuitable_annotation"] += 1
            continue
        seen_names.add(song_pair)
        if video:
            seen_videos.add(video)
        candidates.append({"track_id": f"hook-{record_id}", "record_id": record_id,
                           "key_id": label, "key": key_name(label), "split": "train"})
    return candidates, audit


def create_pilot(work: Path, fma: Path, count: int = 500, extra_exclusions: Path | None = None) -> dict:
    """Freeze training-only IDs and resumably extract features from in-memory renders."""
    data = acquire_release(work / "inputs" / "Hooktheory.json.gz")
    excluded = fma_identities(fma, work)
    extra = json.loads(extra_exclusions.read_text()) if extra_exclusions else []
    excluded.update(identity(a, t) for a, t in extra)
    manifest_file = work / "manifest.json"
    inputs = {"release_sha256": SHA256, "renderer": VERSION, "config": config_signature(),
              "count": count, "seed": 42, "excluded_identities_sha256": digest({"pairs": sorted(excluded)})}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["inputs"] != inputs or manifest["signature"] != digest({k: v for k, v in manifest.items() if k != "signature"}):
            raise ValueError("frozen Hooktheory manifest does not match current inputs")
        selected = manifest["tracks"]
    else:
        pool, audit = candidate_rows(data, excluded)
        selected = sorted(balanced_pick(pool, min(count, len(pool)), 42), key=lambda row: row["track_id"])
        if not selected:
            raise ValueError("no eligible Hooktheory excerpts")
        manifest = {"inputs": inputs, "tracks": selected, "eligible": len(pool),
                    "release_rows": len(data), "filter_counts": dict(audit),
                    "test_identity_coverage": {"fmak": 75, "giantsteps": len(extra)}}
        manifest["signature"] = digest(manifest)
        atomic_json(manifest_file, manifest)
    print(f"Frozen Hooktheory pilot: {len(selected)} of {manifest['eligible']} eligible; {manifest['signature']}", flush=True)
    rows, counts = [], Counter()
    for index, row in enumerate(selected, 1):
        path = work / "checkpoints" / f"{row['track_id']}.json"
        fields = dict(row, manifest=manifest["signature"])
        cached = json.loads(path.read_text()) if path.exists() else {}
        h = np.array([cached.get(name, -1) for name in HISTOGRAM], dtype=float)
        if all(cached.get(k) == v for k, v in fields.items()) and np.isfinite(h).all() and (h >= 0).all() and np.isclose(h.sum(), 1):
            counts["resumed"] += 1
        else:
            audio = render_audio(*note_events(data[row["record_id"]]["annotations"]))
            h = histogram_from_audio(audio, SAMPLE_RATE)
            cached = dict(fields, **dict(zip(HISTOGRAM, h.tolist(), strict=True)))
            atomic_json(path, cached)
            counts["rendered"] += 1
        rows.append(cached)
        if index % 50 == 0 or index == len(selected):
            print(f"{index}/{len(selected)} chroma rows ready", flush=True)
    target = work / "features.parquet"
    table = pd.DataFrame(rows).sort_index(axis=1)
    if not target.exists() or not pd.read_parquet(target).equals(table):
        atomic_bytes(target, table.to_parquet(index=False))
    summary = {"completed": len(rows), "manifest": manifest["signature"], "counts": dict(counts),
               "audio_retained": False, "original_recordings_fetched": False}
    atomic_json(work / "extraction.json", summary)
    return summary


def main() -> None:
    """Render the bounded research pilot and retain only features and metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT)
    parser.add_argument("--fma", type=Path, default=FMA_CACHE)
    parser.add_argument("--count", type=int, choices=range(1, 501), metavar="1–500", default=500)
    parser.add_argument("--giantsteps-test-identities", type=Path, help="optional JSON array of authorized artist/title pairs")
    args = parser.parse_args()
    print(json.dumps(create_pilot(args.work, args.fma, args.count, args.giantsteps_test_identities)))


if __name__ == "__main__":
    main()
