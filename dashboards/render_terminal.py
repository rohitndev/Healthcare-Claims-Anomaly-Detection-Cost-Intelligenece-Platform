"""Render the real pipeline run log into a clean terminal-style PNG.

Reads the actual ``output/logs/platform.log`` produced by the most recent run and
draws it as a light-theme terminal window, so the README's terminal screenshot is
a faithful capture of real platform output (not a mockup).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DASHBOARD_DIR, LOG_DIR, PROJECT_ROOT

DOCS_DIR = PROJECT_ROOT / "docs"


def render() -> Path:
    log_path = LOG_DIR / "platform.log"
    lines = log_path.read_text("utf-8").splitlines()

    # Keep the tail of the most recent run (from the last banner onward).
    start = 0
    for i, line in enumerate(lines):
        if "Healthcare Claims Anomaly Detection & Cost Intelligence Platform" in line:
            start = i - 1
    run = lines[max(start, 0):]
    # Trim the verbose module column for readability in the image.
    cleaned = []
    for ln in run:
        parts = ln.split(" | ", 3)
        cleaned.append(parts[-1] if len(parts) == 4 else ln)
    cleaned = cleaned[:46]

    fig_h = max(6.0, 0.27 * len(cleaned) + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f8fa")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Window chrome: title bar + traffic-light dots.
    ax.add_patch(plt.Rectangle((0.0, 0.965), 1.0, 0.035, color="#e4e7eb"))
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        ax.add_patch(plt.Circle((0.018 + i * 0.022, 0.982), 0.007, color=c))
    ax.text(0.5, 0.982, "run_pipeline.py — claims intelligence platform",
            ha="center", va="center", fontsize=9, color="#5a6472")

    y = 0.945
    dy = 0.945 / (len(cleaned) + 1)
    for ln in cleaned:
        color = "#1f3b6f"
        if any(k in ln for k in ("WARNING", "fallback", "unavailable")):
            color = "#b8860b"
        elif "HIGH-risk" in ln or "alert" in ln.lower():
            color = "#c0392b"
        elif ln.strip().startswith(("=", "-", "PIPELINE SUMMARY")):
            color = "#2e7d52"
        ax.text(0.012, y, ln, ha="left", va="top", fontsize=8.2,
                family="monospace", color=color)
        y -= dy

    out = Path(DASHBOARD_DIR) / "pipeline-terminal-output.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, DOCS_DIR / "pipeline-terminal-output.png")
    plt.close(fig)
    print(f"Rendered terminal output -> {DOCS_DIR / 'pipeline-terminal-output.png'}")
    return out


if __name__ == "__main__":
    render()
