"""Headline numbers, figures, and the rubric verdict. PRD §4, §9, §10.

The decision is mechanical: H5 gates everything (a failed H5 -> Abandon, no
re-scope), and every other Pass is conditional on beating the control by the margin
in config.Thresholds. No interpretation latitude (PRD §4, §9).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def calibrate_s3(rate_by_band: dict, bands: tuple, t5: float):
    """S3 := lower edge of the lowest band whose acceptability rate >= T5."""
    for lo, hi in bands:
        key = f"{lo:.2f}-{hi:.2f}"
        if rate_by_band.get(key, 0.0) >= t5:
            return lo
    return None


def decide(h: dict) -> dict:
    """Apply the PRD §4 matrix. H5 is evaluated first and gates everything."""
    if not h["H5"]["pass"]:
        return {
            "outcome": "Abandon",
            "action": "Document negative result in docs/findings.md. No re-scope on a failed H5.",
            "h5_gate": "FAILED",
        }
    p1, p2, p3, p4 = (h["H1"]["pass"], h["H2"]["pass"], h["H3"]["pass"], h["H4"]["pass"])
    if p1 and p2 and p3 and p4:
        out, act = "Strong positive", "Commit to project, design architecture"
    elif p1 and p2 and (not p3) and p4:
        out, act = "Weak positive", "Re-scope to in-conversation caching, not cross-user"
    elif (not p1) and p2 and p4:
        out, act = "Domain mismatch", "Re-run on a narrower slice (emails only, reports only)"
    elif (not p1) and (not p2) and (not p3) and (not p4):
        out, act = "Marginal", "Re-scope to narrowest viable slice; one attempt only"
    else:
        out, act = "Anomalous", "Investigate, then re-decide"
    return {"outcome": out, "action": act, "h5_gate": "PASSED"}


def _fig(path: Path, draw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    draw(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_figures(out_dir: Path, fig_data: dict) -> None:
    fdir = out_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    def cluster_sizes(ax):
        s = sorted(fig_data["cluster_sizes"], reverse=True)
        ax.bar(range(len(s)), s)
        ax.set_title("Cluster size distribution (subject)")
        ax.set_xlabel("cluster rank")
        ax.set_ylabel("size")

    def nn_dist(ax):
        ax.hist(fig_data["nn_subject"], bins=50, alpha=0.7, label="subject")
        ax.hist(fig_data["nn_scrambled"], bins=50, alpha=0.7, label="scrambled")
        ax.set_title("Nearest-neighbor cosine")
        ax.set_xlabel("cosine")
        ax.legend()

    def pr_scatter(ax):
        ax.scatter(fig_data["pr_prompt"], fig_data["pr_response"], s=4, alpha=0.3)
        ax.set_title("Prompt vs response cosine (H4 — inspect before trusting rho)")
        ax.set_xlabel("prompt cosine")
        ax.set_ylabel("response cosine")

    def subset_vs_control(ax):
        labels = ["H1 cov", "H3 frac", "H5 rate"]
        subj = [fig_data["h1_subject"], fig_data["h3_subject"], fig_data["h5_subject"]]
        ctrl = [fig_data["h1_control"], fig_data["h3_control"], fig_data["h5_control"]]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, subj, 0.4, label="subject")
        ax.bar(x + 0.2, ctrl, 0.4, label="control")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Subject vs control")
        ax.legend()

    _fig(fdir / "cluster_size_distribution.png", cluster_sizes)
    _fig(fdir / "nn_similarity_distribution.png", nn_dist)
    _fig(fdir / "prompt_vs_response_similarity.png", pr_scatter)
    _fig(fdir / "subset_vs_control.png", subset_vs_control)


def write_cluster_examples(out_dir: Path, examples: list[dict]) -> None:
    lines = ["# Top cluster examples (PRD §10)", ""]
    for e in examples:
        lines.append(f"## Cluster {e['cluster']} (size {e['size']})")
        for ex in e["examples"]:
            lines.append(f"- {ex.replace(chr(10), ' ')}")
        lines.append("")
    (out_dir / "cluster_examples.md").write_text("\n".join(lines), encoding="utf-8")


def write_headline(out_dir: Path, headline: dict) -> None:
    (out_dir / "headline_numbers.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )


def write_summary(out_dir: Path, headline: dict) -> None:
    d = headline["decision"]
    th = headline["thresholds"]
    lines = [
        "# Auto-generated summary (PRD §4 rubric)",
        "",
        f"**Run:** {headline['run']}  **N:** {headline['n']}  "
        f"**Dataset:** {headline['dataset']}",
        "",
        f"**Thresholds provisional:** {th.get('provisional')} "
        "(a run on placeholders is a pilot, not the decision run — PRD §4a)",
        "",
        "| Hypothesis | Value | Threshold | Control gap | Pass |",
        "| --- | --- | --- | --- | --- |",
        f"| H1 coverage{' (DEGENERATE)' if headline['H1'].get('degenerate') else ''} | "
        f"{headline['H1']['coverage_median']:.3f} | {th['T1']} | "
        f"{headline['H1']['gap']:.3f} | {headline['H1']['pass']} |",
        f"| H2 templated | {headline['H2'].get('n_yes')}/{headline['H2'].get('n_clusters')} | "
        f">=60% | n/a | {headline['H2']['pass']} |",
        f"| H3 frac@S3 | {headline['H3']['fraction_at_S3']:.3f} | {th['T3']} | "
        f"{headline['H3']['gap']:.3f} | {headline['H3']['pass']} |",
        f"| H4 spearman | {headline['H4']['spearman']:.3f} | {th['H4_rho']} | n/a | "
        f"{headline['H4']['pass']} |",
        f"| H5 (GATING) | {headline['H5'].get('best_rate')} | {th['T5']} | "
        f"{headline['H5']['gap']:.3f} | {headline['H5']['pass']} |",
        "",
        f"## Decision: {d['outcome']}",
        "",
        f"- H5 gate: **{d['h5_gate']}**",
        f"- Action: {d['action']}",
        "",
        "Decision is mechanical (PRD §9). Do not override with intuition. Do not "
        "re-scope on a failed H5.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
