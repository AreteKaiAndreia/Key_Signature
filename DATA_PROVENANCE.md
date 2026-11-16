# Data provenance and redistribution

The implementation uses the **GiantSteps Key Dataset** annotations and audio
previews described by Peter Knees et al., *Two Data Sets for Tempo Estimation
and Key Detection in Electronic Dance Music Annotated from User Corrections*
(ISMIR 2015).

- Dataset repository: https://github.com/GiantSteps/giantsteps-key-dataset
- Dataset page and requested citation: https://www.cp.jku.at/datasets/giantsteps/
- Primary preview mirror used by the upstream downloader:
  `https://www.cp.jku.at/datasets/giantsteps/backup/`
- Fallback preview host: `https://geo-samples.beatport.com/lofi/`

The project downloads one preview into a temporary file, verifies its upstream
MD5 where available, extracts a 36-number chroma feature vector, and deletes
the preview immediately. Raw audio is never committed or retained.

The MIT license in this repository covers only original project source code.
It does not purport to relicense GiantSteps annotations, audio, or derived
feature caches. No explicit redistribution license was found in the upstream
repository during implementation. Consequently, `data/annotations/`,
`data/features/`, `data/work/`, and trained models remain excluded from the
public repository until a license or permission basis is recorded here.

Until then, GitHub runs the harness self-tests on every push but does not claim
the real-data acceptance job passed. That job becomes active when an authorized
`KEYCLF_FEATURES_URL` is configured or `KEYCLF_FEATURES_COMMITTED=1` explicitly
records that the committed cache has been cleared for publication.

## Additional research metadata (2026-10-30)

The research audit also inspected the upstream
[GiantSteps-MTG key annotations](https://github.com/GiantSteps/giantsteps-mtg-key-dataset).
The current snapshot has 1,486 annotation files. The audit identified 1,349
single-key candidates and recorded 137 ambiguous/unknown annotations separately.
It identified 429 major-key candidates after checking exact track IDs and MD5
records against the original 604 tracks. These checks do not establish the
absence of remixes, near duplicates, or artist overlap.

This audit collected metadata only. No additional recordings have entered the
model, and this project does not grant redistribution rights over these
annotations or any corresponding audio. Acquisition manifests, derived
features, point coordinates, trained artifacts, and the technical presentation
remain local under ignored data/report directories until their publication
basis is documented.

## FMAKv2 pilot (2026-11-10)

- [FMAKv2 labels](https://zenodo.org/records/12759100) (CC BY 4.0; Kong, 2024)
  are matched against [original FMAK labels and audio](https://zenodo.org/records/10719860)
  (Wong and Hernandez). Tracks whose labels disagree are excluded before
  sampling. These are two annotation versions of the same audio, not two corpora.
- [Free Music Archive metadata](https://github.com/mdeff/fma) supplies the
  artist ID, duration, and recording license. The pilot accepts explicitly
  recognized Creative Commons attribution terms without NoDerivatives, or
  public-domain entries, for local noncommercial research. Audio terms vary by
  track and are distinct from the labels' license.
- Partial archive reads validate byte range boundaries and member CRCs. The
  archive-level published MD5 values are recorded for provenance, not claimed
  as verified without downloading the full archives. Pilot audio MD5 values
  are compared against all 604 original preview checksums and against each
  other. Exact checksum checks cannot rule out different encodes, remixes,
  excerpt overlap, or cross-dataset artist overlap.
- The original GiantSteps split stays unchanged. FMA artists do not cross
  FMA splits. Sampling prioritizes major keys and balances available tonics;
  the resulting test is a pilot sample, not representative of all FMAK songs.
- No audio, features, label manifests, or weights are released by this pilot.
  The original publication gate remains unchanged.

## Hooktheory notation pilot (2026-11-13)

The project used the [Sheet Sage research release of Hooktheory annotations](https://github.com/chrisdonahue/sheetsage#hooktheory-dataset),
attributed to Hooktheory and to Chris Donahue, John Thickstun, and Percy Liang
(*Melody transcription via generative pre-training*, ISMIR 2023). The authors
list the annotation release under CC BY-NC-SA 3.0. The compressed archive
`Hooktheory.json.gz` is pinned by SHA-256
`917b7cd58f5f4e07d6c36acf7bfad958c99ee05472dab3555399141094698e0c`.
The pilot uses annotations from the published training partition for local,
noncommercial research. It does not scrape TheoryTab or download linked media.

Five hundred excerpts were synthesized from annotated absolute melody and
chord pitches, with a fixed sound and tempo. The annotated key is used only as
a label. Audio exists only in memory. The pilot excludes duplicate titles and
artist identities, duplicate media IDs, and known held-out matches, including
screening against all 75 held-out FMAK track identities. The GiantSteps cache
does not have matched artist/title identities for this check, so overlap with
GiantSteps by composition or recording cannot be ruled out. The original test
IDs were unchanged, and no synthesized clips entered either test set.

### Hosted model decision

The project's release plan keeps the custom `hook-0.25` forest local until it
documents a basis for hosting or distributing the exact artifact. The exported
forest stores learned split thresholds and class counts from 12-value chroma
summaries. The release code packages the estimator and model metadata.
The dataset references above describe the training inputs. We found no
explicit provision in those references that prohibits sharing this fitted
forest. Whether a dataset license applies to the exported model depends on
the license and what the artifact contains. See
[Creative Commons' analysis of licenses and trained models](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/).

The Essentia engine in the [public app](https://musickeyanalyzer.streamlit.app/)
uses fixed key profiles and needs no trained weights or project datasets.
See [NOTICE.md](NOTICE.md) for the hosted software licenses.

## Training sources for the open key model

The new models use 480 original generated musical phrases and 662 loops from
[Ramires et al., The Freesound Loop Dataset and Annotation Tool](https://doi.org/10.5281/zenodo.3967852)
(ISMIR 2020). The generator synthesizes bass, chords, and melody directly
from seeded note events. It uses no samples or score transcriptions. The
Freesound subset contains 356 CC0 1.0 and 306 CC BY 3.0 sounds, as listed for
each recording in the publisher's archive metadata. It uses only consistent
human major/minor key annotations. The pipeline checks the annotation
archive's published checksum and the selected ZIP members' CRCs. It saves
pitch summaries and audit metadata locally. Per-sound credit and license
links are recorded in [FREESOUND_CREDITS.tsv](FREESOUND_CREDITS.tsv).

GiantSteps and FMAK are used to compare candidates and evaluate the final
one on their existing held-out partitions. The fit uses only Freesound and
generated phrases. Hooktheory is outside this training run. No audio,
annotation archives, per-recording features, or benchmark labels are part
of the open model artifact.

The [public artifact](keyclf/open_key_model.json) contains the selected
network's numerical coefficients and class names. SHA-256
`0ca344dd5d21b0153df0a93e9010858f59a45a64ace963366a81f2fc7ff2a92a`.
The original training and inference code and project contributions to this
artifact are available under [MIT](LICENSE). Freesound material keeps its
per-sound CC0 or CC BY terms. The combined web app includes the AGPL
Essentia dependency described in [NOTICE.md](NOTICE.md).

The [linear SVM](keyclf/linear_svm.json) uses the same 662 CC0/CC BY loops
and 480 original phrases. Training rotates each pitch summary and its label
through all 12 tonics. The release is a numerical export of 276 linear
decision boundaries, with the training scaler folded into the coefficients.
The JSON contains class names, weights, and intercepts. The development
estimator and its per-recording support vectors are retained locally.
GiantSteps and FMAK supplied validation and test clips. See
[Freesound credits](FREESOUND_CREDITS.tsv) for source attribution.
The SVM artifact SHA-256 is
`aff04bfe0e26d414306ed78d685415d9830b42d218d5947f1b1f128cc7301096`.
