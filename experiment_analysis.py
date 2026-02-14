#!/usr/bin/env python3
"""Token Length Analysis for Experiment Transcripts.

Parses experiment transcripts, tokenizes each message, and computes/plots
statistics on message length (in tokens) vs turn number across conditions.
"""

import argparse
import os
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from experiment_trajectory import (
    MODEL_NAME,
    OPENER_STYLES,
    SYSTEM_COLORS,
    discover_conditions,
    natural_sort_key,
    parse_transcript,
)

DEFAULT_BASE_DIR = "transcripts/experiments"


# ── Helpers ──────────────────────────────────────────────────────────────────

def tokenize_messages(messages, tokenizer):
    """Return array of token counts for each message."""
    counts = []
    for _, content in messages:
        tokens = tokenizer.encode(content, add_special_tokens=False)
        counts.append(len(tokens))
    return np.array(counts, dtype=int)


def collect_token_counts(conditions, tokenizer, max_transcripts=None, max_messages=None):
    """For each condition, collect token-count arrays from all transcripts.

    Returns dict[condition_name, list[np.array]].
    """
    result = {}
    for name, cond_dir in conditions:
        transcript_paths = sorted(cond_dir.glob("*.txt"), key=natural_sort_key)
        if max_transcripts:
            transcript_paths = transcript_paths[:max_transcripts]

        arrays = []
        for tpath in transcript_paths:
            messages = parse_transcript(str(tpath))
            # Skip bare "Hello." opener if present
            if messages and messages[0][1].strip().rstrip('.') == "Hello":
                messages = messages[1:]
            if max_messages:
                messages = messages[:max_messages]
            if not messages:
                continue
            counts = tokenize_messages(messages, tokenizer)
            arrays.append(counts)
            print(f"  [{name}] {tpath.name}: {len(messages)} msgs, "
                  f"mean {counts.mean():.0f} tok", end="\r")

        print(f"  [{name}] {len(arrays)} transcripts loaded" + " " * 40)
        result[name] = arrays
    return result


def compute_statistics(arrays):
    """Compute per-condition stats from list of token-count arrays.

    Truncates to min length, stacks into matrix, and computes summary stats.
    Returns dict with keys: n, turns, mean_per_turn, std_per_turn,
    median_per_turn, grand_mean, grand_std, grand_median, slope, mat.
    """
    if not arrays:
        return None

    min_len = min(len(a) for a in arrays)
    mat = np.stack([a[:min_len] for a in arrays])  # (n_transcripts, n_turns)

    mean_per_turn = mat.mean(axis=0)
    std_per_turn = mat.std(axis=0)
    median_per_turn = np.median(mat, axis=0)

    # Linear trend: slope in tokens/turn
    turns = np.arange(min_len)
    slope, _ = np.polyfit(turns, mean_per_turn, 1)

    return {
        "n": mat.shape[0],
        "turns": min_len,
        "mean_per_turn": mean_per_turn,
        "std_per_turn": std_per_turn,
        "median_per_turn": median_per_turn,
        "grand_mean": mat.mean(),
        "grand_std": mat.std(),
        "grand_median": np.median(mat),
        "slope": slope,
        "mat": mat,
    }


def print_summary(condition_stats):
    """Print summary table and A/B comparison to console."""
    print(f"\n{'Condition':<20} {'N':>4} {'Turns':>6} {'Mean':>7} {'Std':>7} "
          f"{'Median':>7} {'Slope':>8}")
    print("-" * 65)
    for name, stats in sorted(condition_stats.items()):
        if stats is None:
            continue
        print(f"{name:<20} {stats['n']:>4} {stats['turns']:>6} "
              f"{stats['grand_mean']:>7.1f} {stats['grand_std']:>7.1f} "
              f"{stats['grand_median']:>7.1f} {stats['slope']:>+8.2f}")

    # Model A vs B comparison
    print(f"\n{'Condition':<20} {'A mean':>8} {'B mean':>8} {'B/A':>6}")
    print("-" * 45)
    for name, stats in sorted(condition_stats.items()):
        if stats is None:
            continue
        mat = stats["mat"]
        a_tokens = mat[:, 0::2]  # even columns = Model A turns
        b_tokens = mat[:, 1::2]  # odd columns = Model B turns
        a_mean = a_tokens.mean()
        b_mean = b_tokens.mean()
        ratio = b_mean / a_mean if a_mean > 0 else float("inf")
        print(f"{name:<20} {a_mean:>8.1f} {b_mean:>8.1f} {ratio:>6.2f}")


def plot_token_lengths(condition_stats, output_path):
    """Mean token count vs turn number with ±1σ band, one line per condition."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, stats in sorted(condition_stats.items()):
        if stats is None:
            continue
        parts = name.split("_", 1)
        opener, system = parts if len(parts) == 2 else (name, "none")
        color = SYSTEM_COLORS.get(system, "gray")
        linestyle = OPENER_STYLES.get(opener, "-")

        turns = np.arange(1, stats["turns"] + 1)
        mean = stats["mean_per_turn"]
        std = stats["std_per_turn"]

        ax.plot(turns, mean, color=color, linestyle=linestyle,
                linewidth=1.8, alpha=0.9, label=name)
        ax.fill_between(turns, mean - std, mean + std,
                        color=color, alpha=0.12)

    ax.set_xlabel("Turn Number", fontsize=11)
    ax.set_ylabel("Token Count", fontsize=11)
    ax.set_title("Message Length (tokens) vs Turn — Mean ± 1σ", fontsize=13)
    ax.legend(fontsize=8, ncol=min(len(condition_stats), 5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def plot_ab_comparison(condition_stats, output_path):
    """Model A vs B token lengths per condition (solid=A, dashed=B)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, stats in sorted(condition_stats.items()):
        if stats is None:
            continue
        parts = name.split("_", 1)
        opener, system = parts if len(parts) == 2 else (name, "none")
        color = SYSTEM_COLORS.get(system, "gray")
        linestyle = OPENER_STYLES.get(opener, "-")

        mat = stats["mat"]
        a_mat = mat[:, 0::2]  # even columns = Model A
        b_mat = mat[:, 1::2]  # odd columns = Model B
        min_ab = min(a_mat.shape[1], b_mat.shape[1])
        a_mat = a_mat[:, :min_ab]
        b_mat = b_mat[:, :min_ab]

        turns = np.arange(1, min_ab + 1)
        a_mean = a_mat.mean(axis=0)
        b_mean = b_mat.mean(axis=0)

        ax.plot(turns, a_mean, color=color, linestyle=linestyle,
                linewidth=1.8, alpha=0.9, label=f"{name} A")
        ax.plot(turns, b_mean, color=color, linestyle="--",
                linewidth=1.2, alpha=0.6, label=f"{name} B")

    ax.set_xlabel("Turn Number (within A or B)", fontsize=11)
    ax.set_ylabel("Token Count", fontsize=11)
    ax.set_title("Model A vs B Message Length — solid=A, dashed=B", fontsize=13)
    ax.legend(fontsize=7, ncol=min(len(condition_stats) * 2, 6))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Token length statistics for experiment transcripts",
    )
    parser.add_argument(
        "--base-dir", type=str, default=DEFAULT_BASE_DIR,
        help=f"Experiment root directory (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--opener", type=str, default=None,
        help="Filter to one opener (hello, climate, quantum)",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        help="Filter to one system prompt (none, aware, ontopic)",
    )
    parser.add_argument(
        "--max-transcripts", type=int, default=None,
        help="Limit transcripts per condition",
    )
    parser.add_argument(
        "--max-messages", type=int, default=None,
        help="Limit messages per transcript",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # 1. Discover conditions
    conditions = discover_conditions(base_dir, args.opener, args.system)
    if not conditions:
        print(f"No conditions found in {base_dir}")
        return 1
    print(f"Found {len(conditions)} conditions: {[name for name, _ in conditions]}")

    # 2. Load tokenizer (lightweight, no model)
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Collect token counts
    all_counts = collect_token_counts(
        conditions, tokenizer, args.max_transcripts, args.max_messages,
    )

    # 4. Compute statistics
    condition_stats = {}
    for name, arrays in all_counts.items():
        condition_stats[name] = compute_statistics(arrays)

    # 5. Print summary
    print_summary(condition_stats)

    # 6. Plot
    Path("plots").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    filter_parts = []
    if args.opener:
        filter_parts.append(args.opener)
    if args.system:
        filter_parts.append(args.system)
    filter_str = "_" + "_".join(filter_parts) if filter_parts else ""

    path1 = f"plots/token_lengths{filter_str}_{ts}.png"
    plot_token_lengths(condition_stats, path1)
    subprocess.Popen(["open", path1])

    path2 = f"plots/token_lengths_ab{filter_str}_{ts}.png"
    plot_ab_comparison(condition_stats, path2)
    subprocess.Popen(["open", path2])

    return 0


if __name__ == "__main__":
    exit(main())
