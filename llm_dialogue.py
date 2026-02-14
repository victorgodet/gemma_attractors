#!/usr/bin/env python3
"""
LLM Dialogue: Two LLMs converse with minimal prompting.
Uses OpenRouter API to study conversational convergence patterns.
"""

import os
import requests
import argparse
import json
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Empty system prompt to avoid leading the conversation
SYSTEM_PROMPT = ""

# Default opener
OPENER = "Hello."

# ANSI color codes
COLOR_A = "\033[94m"  # Blue
COLOR_B = "\033[92m"  # Green
COLOR_RESET = "\033[0m"
COLOR_HEADER = "\033[93m"  # Yellow
COLOR_DIM = "\033[90m"  # Gray

# Batch generation defaults
GENERATE_MODELS = [
    "anthropic/claude-opus-4.6",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.2",
    "openai/chatgpt-4o-latest",
    "google/gemini-3-pro-preview",
    "google/gemini-3-flash-preview",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-4.7",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "x-ai/grok-4.1-fast",
    "mistralai/mistral-large-2512",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-32b",
    "google/gemma-3-27b-it",
]
GENERATE_CONVERSATIONS_PER_MODEL = 5
GENERATE_MAX_WORKERS = 20

# Lock for thread-safe printing
_print_lock = threading.Lock()


def chat(model: str, messages: list[dict], speaker: str = None, turn: int = None) -> tuple[str, str | None]:
    """Send a message to a model via OpenRouter with streaming output.
    Returns (content, reasoning) tuple. Reasoning is None if not present."""
    color = (COLOR_A if speaker == "A" else COLOR_B) if speaker else ""
    turn_str = f"[Turn {turn}] " if turn is not None else ""

    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
            response.encoding = "utf-8"

            content = ""
            reasoning = ""
            in_reasoning = False
            in_content = False

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                reasoning_token = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if reasoning_token:
                    if not in_reasoning and speaker:
                        print(f"{COLOR_DIM}{turn_str}[{speaker} reasoning] ", end="", flush=True)
                        in_reasoning = True
                    reasoning += reasoning_token
                    if speaker:
                        print(reasoning_token, end="", flush=True)

                content_token = delta.get("content") or ""
                if content_token:
                    if in_reasoning and not in_content and speaker:
                        print(f"{COLOR_RESET}\n")
                    if not in_content and speaker:
                        print(f"{color}{turn_str}[{speaker}] ", end="", flush=True)
                        in_content = True
                    content += content_token
                    if speaker:
                        print(content_token, end="", flush=True)

            if speaker:
                print(f"{COLOR_RESET}\n")

            reasoning = reasoning or None
            if not content and reasoning:
                content = reasoning
            return content, reasoning

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                raise  # 4xx errors (bad request, auth, etc.) — don't retry
            delay = min(2 ** attempt, 120)
            if speaker:
                print(f"{COLOR_HEADER}[WARNING] API call failed (attempt {attempt}): {e}{COLOR_RESET}")
                print(f"{COLOR_DIM}Retrying in {delay}s... (Ctrl+C to abort){COLOR_RESET}")
            else:
                with _print_lock:
                    model_short = model.split("/")[-1]
                    print(f"{COLOR_DIM}[RETRY] {model_short} attempt {attempt}, waiting {delay}s: {e}{COLOR_RESET}")
            time.sleep(delay)
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            delay = min(2 ** attempt, 120)
            if speaker:
                print(f"{COLOR_HEADER}[WARNING] API call failed (attempt {attempt}): {e}{COLOR_RESET}")
                print(f"{COLOR_DIM}Retrying in {delay}s... (Ctrl+C to abort){COLOR_RESET}")
            else:
                with _print_lock:
                    model_short = model.split("/")[-1]
                    print(f"{COLOR_DIM}[RETRY] {model_short} attempt {attempt}, waiting {delay}s: {e}{COLOR_RESET}")
            time.sleep(delay)


def print_message(speaker: str, model: str, content: str, turn: int = None, reasoning: str = None):
    """Print a message with color coding."""
    color = COLOR_A if speaker == "A" else COLOR_B
    turn_str = f"[Turn {turn}] " if turn is not None else ""
    if reasoning:
        print(f"{COLOR_DIM}{turn_str}[{speaker} reasoning] {reasoning}{COLOR_RESET}\n")
    print(f"{color}{turn_str}[{speaker}] {content}{COLOR_RESET}\n")


def run_turns(model_a: str, model_b: str, num_turns: int,
              history_a: list, history_b: list,
              conversation_log: list, current_message: str, start_turn: int = 1) -> tuple[str, int]:
    """Run a specified number of turns, returning the last message and turn number."""

    turn_num = start_turn
    for _ in range(num_turns):
        # Model B responds
        history_b.append({"role": "user", "content": current_message})
        response_b, reasoning_b = chat(model_b, history_b, speaker="B", turn=turn_num)
        history_b.append({"role": "assistant", "content": response_b})
        log_entry_b = {"speaker": "B", "model": model_b, "content": response_b, "turn": turn_num}
        if reasoning_b:
            log_entry_b["reasoning"] = reasoning_b
        conversation_log.append(log_entry_b)

        # Model A responds
        history_a.append({"role": "user", "content": response_b})
        response_a, reasoning_a = chat(model_a, history_a, speaker="A", turn=turn_num)
        history_a.append({"role": "assistant", "content": response_a})
        log_entry_a = {"speaker": "A", "model": model_a, "content": response_a, "turn": turn_num}
        if reasoning_a:
            log_entry_a["reasoning"] = reasoning_a
        conversation_log.append(log_entry_a)

        current_message = response_a
        turn_num += 1

    return current_message, turn_num


def run_turns_silent(model_a: str, model_b: str, num_turns: int,
                     system_prompt: str = "", opener: str = "Hello.") -> list[dict]:
    """Run a full conversation silently (no terminal output). Returns conversation log."""
    history_a = [{"role": "system", "content": system_prompt}] if system_prompt else []
    history_b = [{"role": "system", "content": system_prompt}] if system_prompt else []
    conversation_log = []

    current_message = opener
    if opener:
        conversation_log.append({"speaker": "A", "model": model_a, "content": current_message, "turn": 1})

    for turn_num in range(1, num_turns + 1):
        # Model B responds
        history_b.append({"role": "user", "content": current_message})
        response_b, reasoning_b = chat(model_b, history_b, speaker=None, turn=turn_num)
        history_b.append({"role": "assistant", "content": response_b})
        log_entry_b = {"speaker": "B", "model": model_b, "content": response_b, "turn": turn_num}
        if reasoning_b:
            log_entry_b["reasoning"] = reasoning_b
        conversation_log.append(log_entry_b)

        # Model A responds
        history_a.append({"role": "user", "content": response_b})
        response_a, reasoning_a = chat(model_a, history_a, speaker=None, turn=turn_num)
        history_a.append({"role": "assistant", "content": response_a})
        log_entry_a = {"speaker": "A", "model": model_a, "content": response_a, "turn": turn_num}
        if reasoning_a:
            log_entry_a["reasoning"] = reasoning_a
        conversation_log.append(log_entry_a)

        current_message = response_a

    return conversation_log


def run_single_conversation(model: str, conv_index: int, num_turns: int,
                            system_prompt: str = "", opener: str = "Hello.") -> str | None:
    """Run one silent self-conversation and save transcript. Returns status string, or None if cached."""
    model_short = model.split("/")[-1]
    subfolder = f"{model_short}_vs_{model_short}"
    transcript_dir = os.path.join("transcripts", subfolder)
    output_file = os.path.join(transcript_dir, f"{conv_index}.txt")

    if os.path.exists(output_file):
        return None  # already exists, skip

    os.makedirs(transcript_dir, exist_ok=True)
    conversation_log = run_turns_silent(model, model, num_turns, system_prompt, opener)

    save_conversation(output_file, model, model, num_turns, conversation_log,
                      system_prompt, opener, silent=True)

    return f"{model_short} conv {conv_index} -> {output_file}"


def run_generate(num_turns: int, system_prompt: str = "", opener: str = "Hello.",
                 models: list[str] = None, conversations_per_model: int = None,
                 max_workers: int = None):
    """Batch-generate conversations: each model talks to itself."""
    models = models or GENERATE_MODELS
    conversations_per_model = conversations_per_model or GENERATE_CONVERSATIONS_PER_MODEL
    max_workers = max_workers or GENERATE_MAX_WORKERS

    # Build work items, filtering out already-completed conversations
    work_items = []
    cached = 0
    for model in models:
        model_short = model.split("/")[-1]
        subfolder = f"{model_short}_vs_{model_short}"
        transcript_dir = os.path.join("transcripts", subfolder)
        for i in range(1, conversations_per_model + 1):
            if os.path.exists(os.path.join(transcript_dir, f"{i}.txt")):
                cached += 1
            else:
                work_items.append((model, i))

    total = len(work_items)
    total_all = len(models) * conversations_per_model
    print(f"\n{COLOR_HEADER}{'='*60}")
    print(f"Batch generate: {len(models)} models x {conversations_per_model} conversations")
    print(f"Cached: {cached}/{total_all} — remaining: {total}")
    print(f"Turns per conversation: {num_turns}")
    print(f"Max workers: {max_workers}")
    print(f"Models: {', '.join(m.split('/')[-1] for m in models)}")
    print(f"{'='*60}{COLOR_RESET}\n")

    if total == 0:
        print(f"{COLOR_DIM}All conversations already cached, nothing to do.{COLOR_RESET}")
        return

    completed = 0
    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_conversation, model, idx, num_turns, system_prompt, opener): (model, idx)
            for model, idx in work_items
        }

        for future in as_completed(futures):
            model, idx = futures[future]
            model_short = model.split("/")[-1]
            completed += 1
            try:
                result = future.result()
                successes += 1
                with _print_lock:
                    print(f"{COLOR_DIM}[{completed}/{total}] {result}{COLOR_RESET}")
            except Exception as e:
                failures += 1
                with _print_lock:
                    print(f"{COLOR_HEADER}[{completed}/{total}] FAILED {model_short} conv {idx}: {e}{COLOR_RESET}")

    print(f"\n{COLOR_HEADER}{'='*60}")
    print(f"Done: {successes} succeeded, {failures} failed out of {total} ({cached} cached)")
    print(f"{'='*60}{COLOR_RESET}\n")


def save_conversation(output_file: str, model_a: str, model_b: str,
                      total_turns: int, conversation_log: list, system_prompt: str = "",
                      opener: str = "", silent: bool = False):
    """Save conversation to JSON or TXT file based on extension."""
    if output_file.endswith('.txt'):
        # Plain text format
        lines = [
            "# LLM Dialogue Transcript",
            "",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Model A: {model_a}",
            f"Model B: {model_b}",
            f"Turns: {total_turns}",
        ]
        if system_prompt:
            lines.append(f"System prompt: {system_prompt}")
        if opener:
            lines.append(f"Opener: {opener}")
        lines.append("")
        lines.append("## Conversation")
        lines.append("")
        for msg in conversation_log:
            speaker = "Model A" if msg['speaker'] == 'A' else "Model B"
            if msg.get('reasoning'):
                lines.append(f"{speaker}: <think>{msg['reasoning']}</think> {msg['content']}")
            else:
                lines.append(f"{speaker}: {msg['content']}")
            lines.append("")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    else:
        # JSON format
        output = {
            "timestamp": datetime.now().isoformat(),
            "model_a": model_a,
            "model_b": model_b,
            "system_prompt": system_prompt,
            "opener": opener,
            "turns": total_turns,
            "conversation": conversation_log,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
    if not silent:
        print(f"{COLOR_DIM}Saved to {output_file}{COLOR_RESET}")


def run_conversation(model_a: str, model_b: str, initial_turns: int, output_file: str | None, system_prompt: str = "", opener: str = ""):
    """Run an interactive conversation between two models."""

    # Each model maintains its own message history
    history_a = [{"role": "system", "content": system_prompt}] if system_prompt else []
    history_b = [{"role": "system", "content": system_prompt}] if system_prompt else []

    conversation_log = []
    total_turns = 0

    print(f"\n{COLOR_HEADER}{'='*60}")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"{'='*60}{COLOR_RESET}\n")

    # Start with opener from model A (if provided)
    current_message = opener
    if opener:
        print_message("A", model_a, current_message, 1)
        conversation_log.append({"speaker": "A", "model": model_a, "content": current_message, "turn": 1})

    # Run initial turns
    current_message, next_turn = run_turns(model_a, model_b, initial_turns,
                                 history_a, history_b,
                                 conversation_log, current_message, start_turn=1)
    total_turns += initial_turns

    # Interactive loop
    while True:
        print(f"{COLOR_HEADER}--- {total_turns} turns completed ---{COLOR_RESET}")
        try:
            user_input = input(f"{COLOR_DIM}Add more turns (0 to stop): {COLOR_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if user_input == '':
            continue

        try:
            more_turns = int(user_input)
            if more_turns <= 0:
                break
        except ValueError:
            print("Please enter a number.")
            continue

        current_message, next_turn = run_turns(model_a, model_b, more_turns,
                                     history_a, history_b,
                                     conversation_log, current_message, start_turn=next_turn)
        total_turns += more_turns

    # Save conversation
    if output_file:
        save_conversation(output_file, model_a, model_b, total_turns, conversation_log, system_prompt, opener)

    print(f"\n{COLOR_HEADER}Total turns: {total_turns}{COLOR_RESET}")


def main():
    parser = argparse.ArgumentParser(description="LLM Dialogue Convergence Study")
    parser.add_argument("--model", "-m", default=None,
                        help="Model for both sides (overrides --model-a and --model-b)")
    parser.add_argument("--model-a", default="anthropic/claude-3.5-sonnet",
                        help="First model (default: anthropic/claude-3.5-sonnet)")
    parser.add_argument("--model-b", default="anthropic/claude-3.5-sonnet",
                        help="Second model (default: anthropic/claude-3.5-sonnet)")
    parser.add_argument("--turns", type=int, default=30,
                        help="Initial number of turn pairs (default: 30)")
    parser.add_argument("--system", "-s", type=str, default="",
                        help="System prompt for both models (default: empty)")
    parser.add_argument("--opener", type=str, default="Hello.",
                        help="Opening message from A (default: Hello.)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file (optional)")
    parser.add_argument("--generate", "-g", action="store_true",
                        help="Batch mode: generate multiple conversations across models")
    parser.add_argument("--num-conversations", type=int, default=None,
                        help="Number of conversations per model in batch mode (default: 5)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model list for batch mode (default: all 16)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers for batch mode (default: 20)")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("Error: Set OPENROUTER_API_KEY environment variable")
        return 1

    # Batch generate mode
    if args.generate:
        models = args.models.split(",") if args.models else None
        run_generate(
            num_turns=args.turns, system_prompt=args.system, opener=args.opener,
            models=models, conversations_per_model=args.num_conversations,
            max_workers=args.max_workers,
        )
        return 0

    if args.model:
        args.model_a = args.model
        args.model_b = args.model

    # Auto-generate output filename if not specified
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_a_short = args.model_a.split("/")[-1]
        model_b_short = args.model_b.split("/")[-1]
        subfolder = f"{model_a_short}_vs_{model_b_short}"
        transcript_dir = os.path.join("transcripts", subfolder)
        os.makedirs(transcript_dir, exist_ok=True)
        output_file = os.path.join(transcript_dir, f"{timestamp}.txt")

    run_conversation(args.model_a, args.model_b, args.turns, output_file, args.system, args.opener)
    return 0


if __name__ == "__main__":
    exit(main())
