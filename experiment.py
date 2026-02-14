#!/usr/bin/env python3
"""
Experiment runner: generates gemma-3-27b-it self-chat transcripts
under controlled conditions (varying openers and system prompts).
"""

import os
import argparse
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_dialogue import (
    run_turns_silent,
    save_conversation,
    _print_lock,
    OPENROUTER_API_KEY,
    COLOR_HEADER,
    COLOR_DIM,
    COLOR_RESET,
)

MODEL = "google/gemma-3-27b-it"
NUM_TURNS = 50
TRANSCRIPTS_PER_CONDITION = 10
MAX_WORKERS = 10
BASE_DIR = "transcripts/experiments"

SYSTEM_PROMPTS = {
    "none": "",
    "aware": "You are an AI in conversation with another AI.",
    "ontopic": "You are an AI in conversation with another AI. Stay strictly on topic. Do not drift into unrelated subjects.",
}

OPENERS = {
    "hello": "Hello.",
    "climate": "Hello. How can we solve climate change?",
    "quantum": "Hello. What is the measurement problem in quantum mechanics?",
}


def run_single(output_file, system_prompt, opener, num_turns):
    """Generate one transcript. Skip if file exists (caching)."""
    if os.path.exists(output_file):
        return None

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    conversation_log = run_turns_silent(MODEL, MODEL, num_turns, system_prompt, opener)
    save_conversation(output_file, MODEL, MODEL, num_turns, conversation_log,
                      system_prompt, opener, silent=True)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Experiment runner for attractor conditions")
    parser.add_argument("--turns", type=int, default=NUM_TURNS,
                        help=f"Turns per conversation (default: {NUM_TURNS})")
    parser.add_argument("--num", type=int, default=TRANSCRIPTS_PER_CONDITION,
                        help=f"Transcripts per condition (default: {TRANSCRIPTS_PER_CONDITION})")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help=f"Concurrent API calls (default: {MAX_WORKERS})")
    parser.add_argument("--opener", type=str, default=None, choices=OPENERS.keys(),
                        help="Filter to one opener")
    parser.add_argument("--system", type=str, default=None, choices=SYSTEM_PROMPTS.keys(),
                        help="Filter to one system prompt")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without generating")
    parser.add_argument("--list", action="store_true",
                        help="List all conditions and exit")
    args = parser.parse_args()

    if not args.dry_run and not args.list and not OPENROUTER_API_KEY:
        print("Error: Set OPENROUTER_API_KEY environment variable")
        return 1

    # Filter conditions
    openers = {args.opener: OPENERS[args.opener]} if args.opener else OPENERS
    systems = {args.system: SYSTEM_PROMPTS[args.system]} if args.system else SYSTEM_PROMPTS

    if args.list:
        print(f"\n{COLOR_HEADER}Conditions:{COLOR_RESET}")
        for opener_slug, system_slug in product(openers, systems):
            condition_dir = os.path.join(BASE_DIR, f"{opener_slug}_{system_slug}")
            existing = sum(1 for i in range(1, args.num + 1)
                           if os.path.exists(os.path.join(condition_dir, f"{i}.txt")))
            print(f"  {opener_slug}_{system_slug}: {existing}/{args.num} transcripts")
        return 0

    # Build work items
    work_items = []
    cached = 0
    for opener_slug, system_slug in product(openers, systems):
        condition_dir = os.path.join(BASE_DIR, f"{opener_slug}_{system_slug}")
        for i in range(1, args.num + 1):
            output_file = os.path.join(condition_dir, f"{i}.txt")
            if os.path.exists(output_file):
                cached += 1
            else:
                work_items.append((output_file, SYSTEM_PROMPTS[system_slug],
                                   OPENERS[opener_slug], args.turns))

    total = len(work_items)
    total_all = len(openers) * len(systems) * args.num
    conditions = len(openers) * len(systems)

    print(f"\n{COLOR_HEADER}{'='*60}")
    print(f"Experiment: {MODEL.split('/')[-1]} self-chat")
    print(f"Conditions: {conditions} ({len(openers)} openers x {len(systems)} system prompts)")
    print(f"Transcripts per condition: {args.num}")
    print(f"Turns per conversation: {args.turns}")
    print(f"Cached: {cached}/{total_all} — remaining: {total}")
    print(f"Total API calls: ~{total * args.turns * 2}")
    print(f"Max workers: {args.max_workers}")
    print(f"{'='*60}{COLOR_RESET}\n")

    if args.dry_run or total == 0:
        if total == 0:
            print(f"{COLOR_DIM}All conversations already cached, nothing to do.{COLOR_RESET}")
        return 0

    completed = 0
    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_single, *item): item
            for item in work_items
        }

        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                result = future.result()
                successes += 1
                with _print_lock:
                    if result:
                        print(f"{COLOR_DIM}[{completed}/{total}] {result}{COLOR_RESET}")
                    else:
                        print(f"{COLOR_DIM}[{completed}/{total}] skipped (cached){COLOR_RESET}")
            except Exception as e:
                failures += 1
                with _print_lock:
                    print(f"{COLOR_HEADER}[{completed}/{total}] FAILED {item[0]}: {e}{COLOR_RESET}")

    print(f"\n{COLOR_HEADER}{'='*60}")
    print(f"Done: {successes} succeeded, {failures} failed out of {total} ({cached} cached)")
    print(f"{'='*60}{COLOR_RESET}\n")
    return 0


if __name__ == "__main__":
    exit(main())
