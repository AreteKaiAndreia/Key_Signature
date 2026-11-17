"""Explanations and held-out benchmark results for the public prediction methods."""

from html import escape
import json
from pathlib import Path

from .method_diagrams import render_method_diagram

OPEN_ML = "Neural Network"
LINEAR_SVM = "Linear SVM"
FOREST = "Random Forest"
BOOSTING = "Gradient Boosting"
KS = "Essentia (Krumhansl–Schmuckler)"
SOFT_VOTE = "Soft Voting"

METHODS = {
    OPEN_ML: (
        "neural_network",
        "The network starts with three pitch summaries for the overall audio, its harmonic sounds, "
        "and the bass. It tries each of the 24 major and minor keys in turn, lining up those "
        "summaries so the proposed home note is in the same place each time.",
        "A small layer of connected calculations learns which combinations of notes make a proposed "
        "key plausible. During training, examples with known keys teach the network to raise the score "
        "of the correct key. At prediction time, it compares the scores of all 24 candidates.",
    ),
    LINEAR_SVM: (
        "linear_svm",
        "The support vector machine compares every pair of keys. For each pair it learns a formula "
        "that multiplies the pitch features by fixed weights and adds them up. The sign of the total "
        "decides which of the two keys wins.",
        "During training, it tries to leave a wide gap between the examples on either side of each "
        "decision boundary. For a new clip, the keys are ranked by how many comparisons they win, "
        "with the strength of those decisions used to break ties.",
    ),
    FOREST: (
        "random_forest",
        "A decision tree follows a series of yes-or-no questions about the audio. For example, it might "
        "ask whether a particular pitch is strong in the bass. The answers lead to a final branch that "
        "stores the mix of keys seen there during training.",
        "The forest trains many trees on different samples of the training data and considers a "
        "random subset of features at each split. Each tree contributes its distribution of likely "
        "keys. Averaging the trees reduces the influence of any one tree's decisions.",
    ),
    BOOSTING: (
        "gradient_boosting",
        "Gradient boosting builds small decision trees in sequence. It groups pitch feature values "
        "into ranges so it can quickly find splits that improve the current predictions.",
        "Each new tree learns an adjustment to the key scores produced by the earlier trees, with "
        "the goal of reducing their prediction error. At analysis time, the initial scores and all "
        "of the tree adjustments add up to a final ranking.",
    ),
    KS: (
        "essentia",
        "Essentia looks for strong frequencies in short sections of audio and combines them into a "
        "12-note pitch profile. The profile describes how much evidence the clip contains for each "
        "note name across octaves.",
        "The Krumhansl–Schmuckler method compares that profile with a major and a minor template "
        "shifted to each of the 12 possible starting notes. It ranks keys by how closely their templates "
        "match the clip. The templates are fixed, so the method can run without training on labeled recordings.",
    ),
    SOFT_VOTE: (
        "soft_voting",
        "Soft Voting runs all five other methods on the same clip. Each method contributes a score "
        "for every possible key. Because some methods produce probabilities and others produce "
        "ranking or match scores, the ensemble first puts them on a common probability scale.",
        "During calibration, a separate set of labeled clips determines how strongly each method favors "
        "its top choices. The ensemble then averages the five adjusted probabilities for each key. "
        "A strong second choice still contributes when that method's first choice differs.",
    ),
}

_BENCHMARK = json.loads(Path(__file__).with_name("method_benchmarks.json").read_text(encoding="utf-8"))


def render_method_details(engine: str) -> str:
    """Render the selected method's explanation and benchmark table."""
    method, first, second = METHODS[engine]
    results = _BENCHMARK["methods"][method]
    rows = []
    for name, label in (
        ("first_exact", "First exact key"),
        ("top3_exact", "Top 3 exact keys"),
        ("top3_pairs", "Top 3 key pairs"),
    ):
        row = results[name]
        accuracy = row["correct"] / _BENCHMARK["clips"]
        rows.append(
            f'<tr><th scope="row">{label}</th><td class="studio-measure">{accuracy:.1%}</td>'
            f'<td class="studio-measure">{row["recall"]:.1%}</td>'
            f'<td class="studio-measure">{row["precision"]:.1%}</td></tr>'
        )
    return f"""<section class="studio-method" aria-labelledby="method-title">
      <h2 class="studio-panel-title" id="method-title">How {escape(engine)} works</h2>
      <div class="studio-method-overview">
        <div class="studio-method-copy">
          <p>{escape(first)}</p>
          <p>{escape(second)}</p>
        </div>
        {render_method_diagram(method)}
      </div>
      <table class="studio-benchmark" aria-label="Held-out key prediction results">
        <thead><tr><th scope="col">Guesses</th><th scope="col">Accuracy</th>
        <th scope="col">Recall</th><th scope="col">Precision</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
      <p class="studio-benchmark-note">
        Scored on {_BENCHMARK["clips"]} held-out clips
        ({_BENCHMARK["datasets"]["FMAK"]} FMAK and {_BENCHMARK["datasets"]["GiantSteps"]} GiantSteps).
        Accuracy counts a clip as correct if any of the listed guesses matches.
        Precision measures how often a guess for a key is right, and recall measures how often
        the method finds that key when it is present. Each key or pair gets equal weight.
        The key-pair row matches the app’s display, where C major and A minor count as one guess.
      </p>
    </section>"""
