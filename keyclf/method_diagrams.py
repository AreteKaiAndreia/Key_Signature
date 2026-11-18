"""Illustrations of how the public key prediction engines reach a score."""

from .key_profiles import MAJOR_PROFILE

_ARROW = '<span class="diagram-arrow" aria-hidden="true">→</span>'


def _network_links(starts: tuple[float, ...], ends: tuple[float, ...]) -> str:
    links = "".join(
        f'<span style="--from:{start}%;--to:{end}%"></span>'
        for start in starts for end in ends
    )
    return f'<div class="diagram-links" aria-hidden="true">{links}</div>'


def _neural_network() -> str:
    rows = (16.7, 50, 83.3)
    return f"""<div class="diagram-network-labels">
      <span>Pitch inputs</span><span>Hidden units</span><span>Key score</span>
    </div>
    <div class="diagram-network" role="img" aria-label="Aligned pitch inputs go through learned
      connections to hidden units. Their outputs contribute to one score for the candidate C major.">
      <ul class="diagram-network-inputs" aria-hidden="true"><li>All notes</li><li>Harmony</li><li>Bass</li></ul>
      {_network_links(rows, rows)}
      <div class="diagram-network-units" aria-hidden="true"><span></span><span></span><span></span></div>
      {_network_links(rows, (50,))}
      <strong class="diagram-network-output" aria-hidden="true">C major</strong>
    </div>
    <p class="diagram-note">Align the pitches to the candidate’s home note and include its major or minor mode.
      The same network scores all 24 candidates.</p>"""


def _linear_boundary() -> str:
    """Draw an illustrative pairwise decision boundary with two pitch features."""
    positions = {
        "g-major": ((12, 25), (19, 42), (28, 20), (37, 54), (49, 47), (40, 36), (62, 24)),
        "c-major": ((37, 83), (49, 74), (56, 67), (65, 83), (70, 58), (81, 65), (86, 32)),
    }
    marks = "".join(
        f'<span class="diagram-point {group}" style="--x:{x}%;--y:{y}%"></span>'
        for group, points in positions.items()
        for x, y in points
    )
    return f"""<div class="diagram-svm">
      <div class="diagram-svm-y">G in the bass ↑</div>
      <div class="diagram-svm-plot" role="img" aria-label="Example scatterplot. A straight line
        separates round G major examples at the upper left from square C major examples at the lower right.">
        <div aria-hidden="true">{marks}</div>
      </div>
      <div class="diagram-svm-x">C in the bass →</div>
      <div class="diagram-legend"><span class="diagram-key g-major">G major</span>
        <span class="diagram-key c-major">C major</span><span class="diagram-key boundary">Decision boundary</span></div>
    </div>"""


def _forest() -> str:
    """Show three trees contributing to one illustrative key score."""
    trees = "".join(
        f'<div class="diagram-tree"><strong>{question}</strong>'
        '<div class="diagram-fork" aria-hidden="true"></div>'
        f'<div class="diagram-branches"><span class="{"decision" if yes else ""}">Yes</span>'
        f'<span class="{"" if yes else "decision"}">No</span></div>'
        f'<div class="diagram-number">{score}%</div></div>'
        for question, yes, score in (("C in bass?", True, 72), ("E in harmony?", False, 45), ("Strong G?", True, 57))
    )
    return f"""<div class="diagram-forest" aria-label="Three example trees">{trees}</div>
      <div class="diagram-merge"><span aria-hidden="true">↓</span>
        <strong class="diagram-number">58%</strong> average for C major</div>"""


def _pitch_profiles() -> str:
    """Compare an example clip profile with two actual major-key templates."""
    major = MAJOR_PROFILE
    profiles = (
        ("Example clip", (8, 1, 3, 1, 6, 2, 1, 7, 1, 2, 1, 1), "clip", ""),
        ("C major template", major, "match", "Closer match"),
        ("F♯ major template", major[6:] + major[:6], "other", "Weaker match"),
    )
    rows = []
    for label, weights, kind, match in profiles:
        peak = max(weights)
        bars = "".join(f'<span style="--height:{100 * weight / peak:.0f}%"></span>' for weight in weights)
        match_label = f'<small class="diagram-match {kind}">{match}</small>' if match else ""
        rows.append(
            f'<div class="diagram-profile {kind}"><div class="diagram-profile-label">{label}{match_label}</div>'
            f'<div class="diagram-profile-bars" aria-hidden="true">{bars}</div></div>'
        )
    notes = "".join(f"<span>{note}</span>" for note in ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"))
    return "".join(rows) + f'<div class="diagram-profile-notes" aria-hidden="true">{notes}</div>'


def _soft_vote() -> str:
    """Visualize equal-weight averaging for one candidate key."""
    inputs = (("Neural Network", 60), ("Linear SVM", 45), ("Random Forest", 55),
              ("Gradient Boosting", 50), ("Essentia", 65))
    mean = sum(score for _, score in inputs) / len(inputs)
    rows = "".join(
        f'<li><span>{label}</span><span class="diagram-vote-bar" aria-hidden="true" '
        f'style="--height:{score}%"></span><strong class="diagram-number">{score}%</strong></li>'
        for label, score in inputs
    )
    return (f'<ul class="diagram-votes">{rows}</ul>'
            f'<div class="diagram-merge"><span aria-hidden="true">↓</span>'
            f'<strong class="diagram-number">{mean:.0f}%</strong> equal-weight average</div>')


_DIAGRAMS = {
    "neural_network": ("How one candidate key gets a score", _neural_network()),
    "linear_svm": ("One matchup between C major and G major", _linear_boundary()),
    "random_forest": ("Three example trees vote on C major", _forest()),
    "gradient_boosting": (
        "Each tree adjusts the previous score",
        f"""<div class="diagram-flow">
          <div class="diagram-step"><strong class="diagram-number">0.0</strong><small>Starting score</small></div>
          {_ARROW}
          <div class="diagram-step"><strong class="diagram-number">0.8</strong><small>First tree adds 0.8</small></div>
          {_ARROW}
          <div class="diagram-step diagram-output"><strong class="diagram-number">0.6</strong>
            <small>Next tree subtracts 0.2</small></div>
        </div>
        <p class="diagram-note">These are example scores for C major. A later tree can raise or lower the score.</p>""",
    ),
    "essentia": ("Match the clip to a key template", _pitch_profiles()),
    "soft_voting": ("Example probabilities for C major", _soft_vote()),
}


def render_method_diagram(method: str) -> str:
    """Provide the selected method's diagram with a visible figure caption."""
    title, diagram = _DIAGRAMS[method]
    return (f'<figure class="studio-method-diagram" aria-labelledby="method-diagram-title" data-method="{method}">'
            f'<figcaption id="method-diagram-title">{title}</figcaption>{diagram}</figure>')
