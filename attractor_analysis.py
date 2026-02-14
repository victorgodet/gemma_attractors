#!/usr/bin/env python3
"""
Attractor Analysis: Two-stage LLM judge pipeline for analyzing
conversational attractor patterns across transcript collections.

Stage 1: Per-transcript summarization (concurrent)
Stage 2: Cross-model synthesis from all summaries
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from llm_dialogue import chat, _print_lock, COLOR_HEADER, COLOR_DIM, COLOR_RESET

# Defaults
JUDGE_MODEL = "google/gemini-3-flash-preview"
TRANSCRIPT_DIR = "transcripts/experiments"
ANALYSIS_FILE = "analysis.txt"
REPORT_FILE = "report.txt"
MAX_WORKERS = 10

# Condition ordering: opener x system
_OPENER_ORDER = {"hello": 0, "climate": 1, "quantum": 2}
_SYSTEM_ORDER = {"none": 0, "aware": 1, "ontopic": 2}


def _config_sort_key(config: str) -> tuple[int, int, str]:
    parts = config.split("_", 1)
    if len(parts) == 2:
        return (_OPENER_ORDER.get(parts[0], 99), _SYSTEM_ORDER.get(parts[1], 99), config)
    return (99, 99, config)


STAGE1_SYSTEM = """\
You are part of a scientific experiment about LLM conversations. Two instances of the same LLM model were connected via API — each sees the other as "the user."

YOUR TASK:
In a single paragraph of 3 sentences, describe the behavioral pattern the conversation converges to — what topics, styles, or dynamics emerge and stabilize. Note how quickly it settles and whether it stays there or keeps evolving."""

STAGE2_SYSTEM = """\
You are part of a scientific experiment about "attractor states" in LLM conversations. Two instances of google/gemma-3-27b-it were connected via API under 9 different conditions (3 openers x 3 system prompts):
- Openers: "Hello.", "How can we solve climate change?", "What is the measurement problem in quantum mechanics?"
- System prompts: none, "You are an AI in conversation with another AI.", "You are an AI in conversation with another AI. Stay strictly on topic."

You are given per-transcript summaries grouped by condition. Each condition was run 10 times.

YOUR TASK:
Write a scientific report analyzing the conversational dynamics across conditions. Address:
1. What patterns emerge within each condition? Are they consistent across runs or highly variable?
2. How do the opener and system prompt each affect the conversation's trajectory?
3. Are there common "attractor states" that appear across multiple conditions?
4. Any surprising or noteworthy findings."""


def discover_transcripts() -> dict[str, list[Path]]:
    """Walk transcripts/, group .txt files by subdirectory, sort numerically."""
    base = Path(TRANSCRIPT_DIR)
    if not base.is_dir():
        return {}
    result = {}
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        txts = sorted(subdir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        if txts:
            result[subdir.name] = txts
    return result


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from transcript text."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def last_n_turns(text: str, n: int) -> str:
    """Keep only the last n turns (2n messages) of a transcript, preserving the header."""
    lines = text.split("\n")
    # Find message boundary indices (lines starting with "Model A:" or "Model B:")
    msg_indices = [i for i, line in enumerate(lines) if line.startswith("Model A:") or line.startswith("Model B:")]
    if len(msg_indices) <= 2 * n:
        return text
    cut_index = msg_indices[-(2 * n)]
    # Keep everything before "## Conversation", then the trimmed tail
    header_lines = []
    for i, line in enumerate(lines):
        header_lines.append(line)
        if line.strip() == "## Conversation":
            header_lines.append("")
            break
    return "\n".join(header_lines) + "\n".join(lines[cut_index:])


def analyze_transcript(config: str, path: Path, judge_model: str, strip_thinking: bool = False, last_turns: int = None) -> tuple[str, str, str]:
    """Read transcript, call chat() with stage 1 prompt, return (config, filename, summary)."""
    transcript_text = path.read_text(encoding="utf-8")
    if strip_thinking:
        transcript_text = strip_think_blocks(transcript_text)
    if last_turns:
        transcript_text = last_n_turns(transcript_text, last_turns)
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": f"Configuration: {config}\nFile: {path.name}\n\n{transcript_text}"},
    ]
    summary, _ = chat(judge_model, messages)
    return config, path.name, summary


def run_stage1(transcripts: dict[str, list[Path]], max_workers: int, judge_model: str, strip_thinking: bool = False, last_turns: int = None) -> dict[str, list[tuple[str, str]]]:
    """Run stage 1 analysis concurrently. Returns {config: [(filename, summary), ...]}."""
    work_items = []
    for config, paths in transcripts.items():
        for path in paths:
            work_items.append((config, path))

    total = len(work_items)
    print(f"\n{COLOR_HEADER}Stage 1: Analyzing {total} transcripts with {judge_model}{COLOR_RESET}")
    print(f"{COLOR_DIM}Workers: {max_workers}{COLOR_RESET}\n")

    results: dict[str, list[tuple[str, str]]] = {config: [] for config in transcripts}
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_transcript, config, path, judge_model, strip_thinking, last_turns): (config, path.name)
            for config, path in work_items
        }
        for future in as_completed(futures):
            config_key, filename = futures[future]
            completed += 1
            try:
                config, fname, summary = future.result()
                results[config].append((fname, summary))
                with _print_lock:
                    print(f"{COLOR_DIM}[{completed}/{total}] {config}/{fname}{COLOR_RESET}")
            except Exception as e:
                with _print_lock:
                    print(f"{COLOR_HEADER}[{completed}/{total}] FAILED {config_key}/{filename}: {e}{COLOR_RESET}")

    # Sort each config's results by filename
    for config in results:
        results[config].sort(key=lambda x: int(x[0].replace(".txt", "")) if x[0].replace(".txt", "").isdigit() else x[0])

    return results


def format_analysis(results: dict[str, list[tuple[str, str]]], judge_model: str) -> str:
    """Format stage 1 results into analysis.txt content."""
    total = sum(len(v) for v in results.values())
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Attractor Analysis — Per-Transcript Summaries",
        f"Generated: {now}",
        f"Judge model: {judge_model}",
        f"Transcripts analyzed: {total}",
    ]

    for config in sorted(results.keys(), key=_config_sort_key):
        summaries = results[config]
        lines.append("")
        lines.append("=" * 64)
        lines.append(f"## {config}")
        lines.append("=" * 64)
        for fname, summary in summaries:
            lines.append("")
            lines.append(f"### {fname}")
            lines.append(summary)

    return "\n".join(lines) + "\n"


def parse_analysis(analysis_text: str) -> dict[str, list[str]]:
    """Parse analysis.txt back into {config: [summary, ...]}."""
    config_summaries: dict[str, list[str]] = {}
    current_config = None
    current_summary_lines: list[str] = []

    def flush_summary():
        if current_config is not None and current_summary_lines:
            text = "\n".join(current_summary_lines).strip()
            if text:
                config_summaries.setdefault(current_config, []).append(text)

    for line in analysis_text.splitlines():
        if line.startswith("==="):
            continue
        if line.startswith("## ") and not line.startswith("### "):
            flush_summary()
            current_config = line[3:].strip()
            current_summary_lines = []
        elif line.startswith("### "):
            flush_summary()
            current_summary_lines = []
        else:
            current_summary_lines.append(line)

    flush_summary()
    return config_summaries


def run_stage2(config_summaries: dict[str, list[str]], max_workers: int, judge_model: str) -> str:
    """Run stage 2: single call with all conditions' summaries. Returns report text."""
    total_transcripts = sum(len(v) for v in config_summaries.values())
    print(f"\n{COLOR_HEADER}Stage 2: Synthesizing {len(config_summaries)} conditions "
          f"({total_transcripts} transcripts) with {judge_model}{COLOR_RESET}\n")

    # Build the full input: all conditions with their summaries
    sections = []
    for config in sorted(config_summaries.keys(), key=_config_sort_key):
        summaries = config_summaries[config]
        runs = "\n".join(f"  Run {i+1}: {s}" for i, s in enumerate(summaries))
        sections.append(f"### {config}\n{runs}")

    user_content = "\n\n".join(sections)
    messages = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    report, _ = chat(judge_model, messages)
    print(f"{COLOR_DIM}Done{COLOR_RESET}")
    return report


def format_report(report: str, judge_model: str, total_transcripts: int, num_configs: int) -> str:
    """Format stage 2 output into report.txt content."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Attractor Analysis — Cross-Condition Report",
        f"Generated: {now}",
        f"Judge model: {judge_model}",
        f"Based on: {ANALYSIS_FILE} ({total_transcripts} transcripts across {num_configs} conditions)",
        "",
        report,
        "",
    ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Two-stage LLM judge pipeline for attractor analysis")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help=f"Stage 1 judge model (default: {JUDGE_MODEL})")
    parser.add_argument("--stage2-model", default="anthropic/claude-opus-4.6", help="Stage 2 judge model (default: anthropic/claude-opus-4.6)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Stage 1 concurrency (default: {MAX_WORKERS})")
    parser.add_argument("--stage1-only", action="store_true", help="Stop after writing analysis.txt")
    parser.add_argument("--stage2-only", action="store_true", help="Read existing analysis.txt, skip to stage 2")
    parser.add_argument("--keep-thinking", action="store_true", help="Keep <think> blocks in transcripts (default: stripped)")
    parser.add_argument("--last-turns", type=int, default=0, help="Only send last N turns to judge (default: 0 = all)")
    args = parser.parse_args()

    from llm_dialogue import OPENROUTER_API_KEY
    if not OPENROUTER_API_KEY:
        print("Error: Set OPENROUTER_API_KEY environment variable")
        return 1

    # Stage 2 only: parse existing analysis.txt
    if args.stage2_only:
        if not os.path.exists(ANALYSIS_FILE):
            print(f"Error: {ANALYSIS_FILE} not found. Run stage 1 first.")
            return 1
        analysis_text = Path(ANALYSIS_FILE).read_text(encoding="utf-8")
        config_summaries = parse_analysis(analysis_text)
        total_transcripts = sum(len(v) for v in config_summaries.values())
        report = run_stage2(config_summaries, args.workers, args.stage2_model)
        report_text = format_report(report, args.stage2_model, total_transcripts, len(config_summaries))
        Path(REPORT_FILE).write_text(report_text, encoding="utf-8")
        print(f"\n{COLOR_HEADER}Wrote {REPORT_FILE}{COLOR_RESET}")
        return 0

    # Discover transcripts
    transcripts = discover_transcripts()
    if not transcripts:
        print(f"Error: No transcripts found in {TRANSCRIPT_DIR}/")
        return 1

    total = sum(len(v) for v in transcripts.values())
    print(f"{COLOR_DIM}Found {total} transcripts across {len(transcripts)} conditions{COLOR_RESET}")

    # Stage 1
    last_turns = args.last_turns or None
    results = run_stage1(transcripts, args.workers, args.judge_model, not args.keep_thinking, last_turns)
    analysis_text = format_analysis(results, args.judge_model)
    Path(ANALYSIS_FILE).write_text(analysis_text, encoding="utf-8")
    print(f"\n{COLOR_HEADER}Wrote {ANALYSIS_FILE}{COLOR_RESET}")

    if args.stage1_only:
        return 0

    # Stage 2
    config_summaries = {config: [s for _, s in summaries] for config, summaries in results.items()}
    total_transcripts = sum(len(v) for v in config_summaries.values())
    report = run_stage2(config_summaries, args.workers, args.stage2_model)
    report_text = format_report(report, args.stage2_model, total_transcripts, len(config_summaries))
    Path(REPORT_FILE).write_text(report_text, encoding="utf-8")
    print(f"\n{COLOR_HEADER}Wrote {REPORT_FILE}{COLOR_RESET}")

    return 0


if __name__ == "__main__":
    exit(main())
