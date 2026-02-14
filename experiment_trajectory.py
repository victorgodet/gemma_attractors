#!/usr/bin/env python3
"""Experiment Trajectory Comparison.

Plots PCA trajectories of residual stream activations, one trajectory per
experimental condition (opener x system prompt), averaged across all transcripts
in that condition. All conditions shown on the same axes for direct comparison.

Supports two modes:
  - Raw residual stream activations at multiple layers (default)
  - SAE feature vectors at layer 31 (--sae flag)

Both modes use per-message extraction (independent forward pass per message).
"""

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests as http_requests
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-3-27b-it"
DEFAULT_BASE_DIR = "transcripts/experiments"
DEFAULT_CACHE_DIR = "cache/experiment_trajectory"
DEFAULT_CACHE_DIR_SAE = "cache/experiment_trajectory_sae"
DEFAULT_LAYERS = [15, 31, 46]
MAX_MSG_TOKENS = 4096

SAE_REPO = "google/gemma-scope-2-27b-it"
SAE_FOLDER = "resid_post/layer_31_width_262k_l0_medium"
SAE_LAYER = 31
TOP_K = 10

NP_MODEL = "gemma-3-27b-it"
NP_SOURCE = "31-gemmascope-2-res-262k"


# ── Utilities (inlined from sae_batch.py) ─────────────────────────────────────

def natural_sort_key(path: Path):
    """Sort key that handles numeric filenames naturally (1, 2, 10 not 1, 10, 2)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.stem)]


def _moving_average(mat: np.ndarray, window: int) -> np.ndarray:
    """Rolling average over axis 0 with given window size."""
    n = mat.shape[0]
    out = np.zeros_like(mat)
    for i in range(n):
        start = i
        end = min(i + window, n)
        out[i] = mat[start:end].mean(axis=0)
    return out


def parse_transcript(path: str) -> list[tuple[str, str]]:
    """Parse transcript into (speaker, content) tuples."""
    text = Path(path).read_text()

    match = re.search(r"## Conversation\s*\n", text)
    if not match:
        raise ValueError(f"Could not find '## Conversation' in {path}")
    text = text[match.end():]

    messages = []
    parts = re.split(r"(Model [AB]):\s*", text)
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            messages.append((speaker, content))
        i += 2

    return messages


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name} (bfloat16)")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer, device


class JumpReLUSAE:
    """Minimal JumpReLU SAE for encoding only."""

    def __init__(self, repo_id: str, folder: str, device: str = "cpu"):
        print(f"Downloading SAE: {repo_id}/{folder}")
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")

        with open(config_path) as f:
            self.config = json.load(f)
        print(f"SAE config: width={self.config['width']}, l0={self.config['l0']}")

        state = load_file(weights_path)
        self.W_enc = state["w_enc"].to(device=device, dtype=torch.float32)
        self.b_enc = state["b_enc"].to(device=device, dtype=torch.float32)
        self.b_dec = state["b_dec"].to(device=device, dtype=torch.float32)
        self.threshold = state["threshold"].to(device=device, dtype=torch.float32)
        print(f"SAE loaded: W_enc {list(self.W_enc.shape)}, threshold {list(self.threshold.shape)}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """JumpReLU encode: features = relu(pre) * (pre > threshold)."""
        x = x.float()
        sae_in = x - self.b_dec
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        return torch.relu(hidden_pre) * (hidden_pre > self.threshold).float()


def extract_features_single(model, tokenizer, sae, messages, layer, device):
    """Run independent forward pass per message, encode through SAE."""
    n_features = sae.config["width"]
    feature_matrix = np.zeros((len(messages), n_features), dtype=np.float32)

    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["activations"] = output[0].detach()
        else:
            captured["activations"] = output.detach()

    target_layer = model.model.language_model.layers[layer]
    handle = target_layer.register_forward_hook(hook_fn)

    for i, (speaker, content) in enumerate(messages):
        chat = [{"role": "user", "content": content}]
        tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)

        if len(tokens) > MAX_MSG_TOKENS:
            tokens = tokens[:MAX_MSG_TOKENS - 2] + tokens[-2:]
            print(f"  Message {i+1}/{len(messages)}: truncated to {len(tokens)} tokens")

        input_ids = torch.tensor([tokens], dtype=torch.long)

        content_start = 4
        content_end = len(tokens) - 2

        print(f"  Message {i+1}/{len(messages)} ({len(tokens)} tokens)", end="\r")
        with torch.no_grad():
            model(input_ids.to(device))

        act = captured["activations"][0, content_start:content_end, :].cpu()
        features = sae.encode(act)
        feature_matrix[i] = features.mean(dim=0).numpy()

    handle.remove()
    print(f"\nFeature matrix: {feature_matrix.shape}")
    return feature_matrix


def fetch_neuronpedia_labels(feature_indices: np.ndarray) -> dict[int, str]:
    """Fetch auto-interp labels from Neuronpedia for given feature indices."""
    labels = {}
    print("Fetching feature labels from Neuronpedia...")
    for idx in feature_indices:
        try:
            url = f"https://www.neuronpedia.org/api/feature/{NP_MODEL}/{NP_SOURCE}/{idx}"
            resp = http_requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                explanations = data.get("explanations", [])
                if explanations:
                    label = explanations[0].get("description", "")
                    if not label:
                        label = explanations[0].get("text", "")
                    labels[idx] = label
                    print(f"  Feature {idx}: {label}")
                else:
                    labels[idx] = ""
            else:
                labels[idx] = ""
        except Exception:
            labels[idx] = ""
    return labels

# ── Style: opener -> linestyle, system -> color ───────────────────────────────

OPENER_STYLES = {
    "hello": "-",
    "climate": "--",
    "quantum": ":",
}

SYSTEM_COLORS = {
    "none": "#d62728",     # red
    "aware": "#1f77b4",    # blue
    "ontopic": "#2ca02c",  # green
}


# ── 1. Discover Conditions ────────────────────────────────────────────────────

def discover_conditions(base_dir, opener_filter=None, system_filter=None):
    """Find condition directories matching filters. Returns sorted list of (name, path) tuples."""
    conditions = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue
        opener, system = parts
        if opener_filter and opener != opener_filter:
            continue
        if system_filter and system != system_filter:
            continue
        conditions.append((name, d))
    return conditions


# ── 2. Per-Message Activation Extraction ──────────────────────────────────────

def extract_activations_per_message(model, tokenizer, messages, layers, device):
    """Run independent forward pass per message, extract mean residual stream activations.

    Each message is wrapped as a single user turn with no conversation context.
    Returns dict mapping layer_idx -> (n_messages, hidden_size) array.
    """
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0].detach()
            else:
                captured[layer_idx] = output.detach()
        return hook_fn

    handles = []
    for layer_idx in layers:
        target_layer = model.model.language_model.layers[layer_idx]
        handle = target_layer.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    hidden_size = model.config.text_config.hidden_size
    result = {layer_idx: np.zeros((len(messages), hidden_size), dtype=np.float32) for layer_idx in layers}

    for i, (speaker, content) in enumerate(messages):
        chat = [{"role": "user", "content": content}]
        tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)

        if len(tokens) > MAX_MSG_TOKENS:
            tokens = tokens[:MAX_MSG_TOKENS - 2] + tokens[-2:]

        input_ids = torch.tensor([tokens], dtype=torch.long)
        content_start = 4
        content_end = len(tokens) - 2

        print(f"    msg {i+1}/{len(messages)} ({len(tokens)} tok)", end="\r")
        with torch.no_grad():
            model(input_ids.to(device))

        for layer_idx in layers:
            act = captured[layer_idx][0, content_start:content_end, :]
            result[layer_idx][i] = act.float().mean(dim=0).cpu().numpy()

    for handle in handles:
        handle.remove()

    print(f"    Extracted {len(layers)} layers x {len(messages)} messages -> ({len(messages)}, {hidden_size})")
    return result


# ── 3. Extract or Load Cached ─────────────────────────────────────────────────

def process_condition(
    name, cond_dir, model, tokenizer, layers, device,
    max_messages, max_transcripts, cache_dir, skip_extraction, sae=None,
):
    """Process all transcripts in a condition. Returns list of cache paths.

    If sae is provided, extracts SAE features at SAE_LAYER instead of raw activations.
    """
    cond_cache = cache_dir / name
    cond_cache.mkdir(parents=True, exist_ok=True)

    transcript_paths = sorted(cond_dir.glob("*.txt"), key=natural_sort_key)
    if max_transcripts:
        transcript_paths = transcript_paths[:max_transcripts]

    cache_paths = []
    for i, tpath in enumerate(transcript_paths):
        cache_file = cond_cache / f"{tpath.stem}.npz"
        cache_paths.append(cache_file)

        if cache_file.exists():
            print(f"  [{name}] {tpath.name} -> cached (skip)")
            continue

        if skip_extraction:
            print(f"  [{name}] {tpath.name} -> no cache, skipping (--skip-extraction)")
            continue

        messages = parse_transcript(str(tpath))
        if messages and messages[0][1].strip().rstrip('.') == "Hello":
            messages = messages[1:]
        if max_messages:
            messages = messages[:max_messages]

        print(f"  [{name}] {tpath.name} ({len(messages)} msgs)")

        if sae is not None:
            feature_matrix = extract_features_single(
                model, tokenizer, sae, messages, SAE_LAYER, device,
            )
            np.savez(cache_file, sae_features=feature_matrix)
        else:
            activations = extract_activations_per_message(
                model, tokenizer, messages, layers, device,
            )
            save_dict = {f"layer_{l}": activations[l] for l in layers}
            np.savez(cache_file, **save_dict)

        print(f"    -> saved {cache_file}")

    return cache_paths


# ── 4. Aggregate Per Condition (Welford's) ────────────────────────────────────

def aggregate_condition(cache_paths, layers, sae_mode=False):
    """Aggregate cached .npz files for a single condition using Welford's algorithm.

    Returns dict mapping key -> mean array of shape (min_msgs, width),
    plus the min_msgs count.
    """
    if sae_mode:
        keys = ["sae_features"]
    else:
        keys = [f"layer_{l}" for l in layers]

    valid_paths = []
    msg_counts = []
    for p in cache_paths:
        if not p.exists():
            continue
        data = np.load(str(p))
        msg_counts.append(data[keys[0]].shape[0])
        valid_paths.append(p)

    if not valid_paths:
        return None, 0

    min_msgs = min(msg_counts)
    sample = np.load(str(valid_paths[0]))
    width = sample[keys[0]].shape[1]
    n = len(valid_paths)

    results = {}
    for key in keys:
        mean = np.zeros((min_msgs, width), dtype=np.float64)

        for i, p in enumerate(valid_paths):
            data = np.load(str(p))[key][:min_msgs].astype(np.float64)
            delta = data - mean
            mean += delta / (i + 1)

        results[key] = mean.astype(np.float32)

    return results, min_msgs


# ── 5. Plot Multi-Condition PCA ───────────────────────────────────────────────

def plot_experiment_trajectories(
    condition_data, layers, ma_window, output_path, sae_mode=False,
):
    """Plot PCA trajectories for all conditions on shared axes, one subplot per layer.

    condition_data: list of (name, layer_dict) where layer_dict maps
                    key -> mean array of shape (n_msgs, width)
    In SAE mode, automatically shows top 5 features by |PC1 loading| as biplot arrows.
    """
    if sae_mode:
        plot_keys = ["sae_features"]
        subplot_titles = ["SAE Features (Layer 31)"]
    else:
        plot_keys = [f"layer_{l}" for l in layers]
        subplot_titles = [f"Layer {l}" for l in layers]

    n_plots = len(plot_keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    for ax, key, title in zip(axes, plot_keys, subplot_titles):
        # Combine A+B into single trajectory per condition (average even/odd rows)
        trajectories = {}
        for name, layer_dict in condition_data:
            mat = layer_dict[key]
            model_a = mat[0::2]
            model_b = mat[1::2]
            min_len = min(len(model_a), len(model_b))
            combined = (model_a[:min_len] + model_b[:min_len]) / 2.0
            if ma_window > 1:
                combined = _moving_average(combined, ma_window)
            trajectories[name] = combined

        # Fit single PCA on all conditions' data
        all_data = np.vstack(list(trajectories.values()))
        pca = PCA(n_components=2)
        pca.fit(all_data)

        # Plot each condition
        for name, traj in trajectories.items():
            parts = name.split("_", 1)
            opener, system = parts if len(parts) == 2 else (name, "none")
            color = SYSTEM_COLORS.get(system, "gray")
            linestyle = OPENER_STYLES.get(opener, "-")

            projected = pca.transform(traj)
            n_turns = projected.shape[0]

            # Trajectory line
            ax.plot(
                projected[:, 0], projected[:, 1],
                color=color, linestyle=linestyle, linewidth=1.8, alpha=0.8,
                label=name, zorder=2,
            )

            # Start marker (circle)
            ax.scatter(
                *projected[0], color=color, s=100, marker='o',
                zorder=3, edgecolors='black', linewidths=0.8,
            )
            # End marker (square)
            ax.scatter(
                *projected[-1], color=color, s=100, marker='s',
                zorder=3, edgecolors='black', linewidths=0.8,
            )

            # Sparse turn annotations (every 20 turns)
            for t in range(0, n_turns, 20):
                ax.annotate(
                    str(t + 1), projected[t], fontsize=5, alpha=0.5,
                    xytext=(3, 3), textcoords='offset points', color=color,
                )

        # Biplot arrows for top 5 SAE features by |PC1 loading|
        if sae_mode:
            n_top = 5
            loadings_pc1 = np.abs(pca.components_[0])
            top_features = np.argsort(loadings_pc1)[::-1][:n_top]
            feature_labels = fetch_neuronpedia_labels(top_features)

            all_projected = pca.transform(all_data)
            centroid = all_projected.mean(axis=0)
            data_range = max(np.ptp(all_projected[:, 0]), np.ptp(all_projected[:, 1]))
            max_loading = max(
                np.max(np.abs(pca.components_[0, top_features])),
                np.max(np.abs(pca.components_[1, top_features])),
            )
            arrow_scale = data_range * 0.4 / max_loading if max_loading > 0 else 1.0

            print(f"\nTop {n_top} SAE features by |PC1 loading|:")
            print(f"{'Feature':>10} {'Loading':>10}  Label")
            print("-" * 70)

            cmap_tab10 = plt.cm.tab10
            for j, idx in enumerate(top_features):
                dx = pca.components_[0, idx] * arrow_scale
                dy = pca.components_[1, idx] * arrow_scale
                acolor = cmap_tab10(j % 10)
                ax.arrow(
                    centroid[0], centroid[1], dx, dy,
                    head_width=data_range * 0.015, head_length=data_range * 0.01,
                    fc=acolor, ec=acolor, alpha=0.8, linewidth=1.5, zorder=5,
                )
                label = feature_labels.get(idx, "")
                short_label = label[:35] if label else f"F{idx}"
                ax.text(
                    centroid[0] + dx, centroid[1] + dy, f"  F{idx}: {short_label}",
                    fontsize=6, color=acolor, alpha=0.9, zorder=5,
                )
                print(f"{idx:>10} {pca.components_[0, idx]:>+10.4f}  {label}")

        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=10)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    # Single legend from first subplot, plus start/end markers
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], marker='o', color='black', linestyle='None',
                          markersize=7, markeredgewidth=0.8, label='Start'))
    labels.append('Start')
    handles.append(Line2D([], [], marker='s', color='black', linestyle='None',
                          markersize=7, markeredgewidth=0.8, label='End'))
    labels.append('End')
    fig.legend(
        handles, labels, loc='lower center',
        ncol=min(len(condition_data) + 2, 7), fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    ma_label = f" — MA({ma_window})" if ma_window > 1 else ""
    sae_label = " (SAE)" if sae_mode else ""
    n_conds = len(condition_data)
    fig.suptitle(
        f"Experiment PCA Trajectories{sae_label} — {n_conds} conditions{ma_label} — gemma-3-27b-it",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ── 5b. Plot Delta Trajectories ────────────────────────────────────────────────

def plot_delta_trajectories(condition_data, layers, ma_window, output_path, sae_mode=False):
    """Plot PCA trajectories of activation *differences* (consecutive turns).

    For each condition, computes delta[t] = activation[t+1] - activation[t],
    applies moving average, then projects via PCA. If conversations collapse
    into repetitive loops, deltas should converge toward the origin.
    """
    if sae_mode:
        plot_keys = ["sae_features"]
        subplot_titles = ["SAE Features (Layer 31)"]
    else:
        plot_keys = [f"layer_{l}" for l in layers]
        subplot_titles = [f"Layer {l}" for l in layers]

    n_plots = len(plot_keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    for ax, key, title in zip(axes, plot_keys, subplot_titles):
        trajectories = {}
        for name, layer_dict in condition_data:
            mat = layer_dict[key]
            # Combine A+B
            model_a = mat[0::2]
            model_b = mat[1::2]
            min_len = min(len(model_a), len(model_b))
            combined = (model_a[:min_len] + model_b[:min_len]) / 2.0
            # Compute consecutive differences
            deltas = np.diff(combined, axis=0)  # (min_len-1, width)
            if ma_window > 1:
                deltas = _moving_average(deltas, ma_window)
            trajectories[name] = deltas

        # Fit PCA on all delta data
        all_data = np.vstack(list(trajectories.values()))
        pca = PCA(n_components=2)
        pca.fit(all_data)

        # Plot origin marker
        ax.scatter(0, 0, color='black', s=200, marker='+', linewidths=2,
                   zorder=10, label='Origin (zero change)')

        for name, traj in trajectories.items():
            parts = name.split("_", 1)
            opener, system = parts if len(parts) == 2 else (name, "none")
            color = SYSTEM_COLORS.get(system, "gray")
            linestyle = OPENER_STYLES.get(opener, "-")

            projected = pca.transform(traj)
            n_turns = projected.shape[0]

            ax.plot(
                projected[:, 0], projected[:, 1],
                color=color, linestyle=linestyle, linewidth=1.8, alpha=0.8,
                label=name, zorder=2,
            )
            # Start marker
            ax.scatter(*projected[0], color=color, s=100, marker='o',
                       zorder=3, edgecolors='black', linewidths=0.8)
            # End marker
            ax.scatter(*projected[-1], color=color, s=100, marker='s',
                       zorder=3, edgecolors='black', linewidths=0.8)

            for t in range(0, n_turns, 20):
                ax.annotate(
                    str(t + 1), projected[t], fontsize=5, alpha=0.5,
                    xytext=(3, 3), textcoords='offset points', color=color,
                )

        # Biplot arrows for top 5 SAE features by |PC1 loading|
        if sae_mode:
            n_top = 5
            loadings_pc1 = np.abs(pca.components_[0])
            top_idx = np.argsort(loadings_pc1)[::-1][:n_top]

            all_projected = pca.transform(all_data)
            data_range = max(np.ptp(all_projected[:, 0]), np.ptp(all_projected[:, 1]))
            max_loading = max(
                np.max(np.abs(pca.components_[0, top_idx])),
                np.max(np.abs(pca.components_[1, top_idx])),
            )
            arrow_scale = data_range * 0.4 / max_loading if max_loading > 0 else 1.0

            # Fetch labels (only once across subplots)
            if not hasattr(plot_delta_trajectories, '_cached_labels'):
                plot_delta_trajectories._cached_labels = {}
            for idx in top_idx:
                if idx not in plot_delta_trajectories._cached_labels:
                    labels = fetch_neuronpedia_labels(top_idx)
                    plot_delta_trajectories._cached_labels.update(labels)
                    break

            print(f"\nTop {n_top} SAE features by |PC1 loading| (delta PCA):")
            print(f"{'Feature':>10} {'Loading':>10}  Label")
            print("-" * 70)

            cmap_tab10 = plt.cm.tab10
            for j, idx in enumerate(top_idx):
                dx = pca.components_[0, idx] * arrow_scale
                dy = pca.components_[1, idx] * arrow_scale
                acolor = cmap_tab10(j % 10)
                ax.arrow(
                    0, 0, dx, dy,
                    head_width=data_range * 0.015, head_length=data_range * 0.01,
                    fc=acolor, ec=acolor, alpha=0.8, linewidth=1.5, zorder=5,
                )
                label = plot_delta_trajectories._cached_labels.get(idx, "")
                short_label = label[:35] if label else f"F{idx}"
                ax.text(
                    dx, dy, f"  F{idx}: {short_label}",
                    fontsize=6, color=acolor, alpha=0.9, zorder=5,
                )
                print(f"{idx:>10} {pca.components_[0, idx]:>+10.4f}  {label}")

        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=10)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=10)
        ax.set_title(f"{title} — Deltas", fontsize=11)
        ax.grid(True, alpha=0.3)

    # Legend with start/end markers
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], marker='o', color='black', linestyle='None',
                          markersize=7, markeredgewidth=0.8, label='Start'))
    labels.append('Start')
    handles.append(Line2D([], [], marker='s', color='black', linestyle='None',
                          markersize=7, markeredgewidth=0.8, label='End'))
    labels.append('End')
    fig.legend(
        handles, labels, loc='lower center',
        ncol=min(len(condition_data) + 3, 8), fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    ma_label = f" — MA({ma_window})" if ma_window > 1 else ""
    sae_label = " (SAE)" if sae_mode else ""
    n_conds = len(condition_data)
    fig.suptitle(
        f"Activation Deltas{sae_label} — {n_conds} conditions{ma_label} — gemma-3-27b-it",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ── 6. Feature Evolution Plot (SAE mode) ──────────────────────────────────────

def plot_feature_evolution(condition_data, feature_indices, labels, output_path):
    """Plot activation of specific SAE features over turns, one line per condition."""
    key = "sae_features"
    n_features = len(feature_indices)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]

    for ax, idx in zip(axes, feature_indices):
        for name, layer_dict in condition_data:
            mat = layer_dict[key]
            # Average A+B
            model_a = mat[0::2, idx]
            model_b = mat[1::2, idx]
            min_len = min(len(model_a), len(model_b))
            combined = (model_a[:min_len] + model_b[:min_len]) / 2.0

            parts = name.split("_", 1)
            opener, system = parts if len(parts) == 2 else (name, "none")
            color = SYSTEM_COLORS.get(system, "gray")
            linestyle = OPENER_STYLES.get(opener, "-")
            ax.plot(np.arange(1, min_len + 1), combined, color=color, linestyle=linestyle,
                    linewidth=1.5, alpha=0.8, label=name)

        label_text = (labels or {}).get(idx, "")
        short = f": {label_text[:50]}" if label_text else ""
        ax.set_ylabel(f"F{idx}{short}", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Turn Number", fontsize=10)
    axes[0].legend(fontsize=7, ncol=min(len(condition_data), 5), loc='upper left')

    fig.suptitle(f"SAE Feature Evolution — {len(condition_data)} conditions — gemma-3-27b-it", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ── 7. Interactive Loop ───────────────────────────────────────────────────────

def interactive_loop(condition_data, layers, sae_mode, filter_str, sae_str, initial_ma):
    """REPL for re-plotting with different settings without recomputing."""
    n_conds = len(condition_data)
    n_msgs = len(next(iter(condition_data[0][1].values())))

    print(f"\n{'=' * 60}")
    print(f"  Experiment Trajectory Explorer")
    print(f"  {n_conds} conditions, {n_msgs} messages each")
    print(f"  Commands:")
    print(f"    pca [MA]       — PCA trajectory plot (default MA=1)")
    print(f"    delta [MA]     — PCA of activation differences (convergence plot)")
    if sae_mode:
        print(f"    diff [E L]     — top features by increase (first E vs last L turns)")
        print(f"    end [N]        — top features by activation in last N turns")
        print(f"    F1,F2,...      — plot specific feature IDs over turns")
    print(f"    q              — quit")
    print(f"{'=' * 60}")

    ma = initial_ma

    while True:
        try:
            user_input = input("\nCommand: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() == 'q':
            break

        # ── pca [MA] ──
        if user_input == '' or user_input.lower().startswith('pca'):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    ma = max(1, int(parts[1]))
                except ValueError:
                    print("Usage: pca [MA_window]")
                    continue
            elif user_input == '':
                pass  # reuse current ma

            Path("plots").mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ma_str = f"_ma{ma}" if ma > 1 else ""
            output_path = f"plots/experiment_trajectory{sae_str}{filter_str}{ma_str}_{ts}.png"

            plot_experiment_trajectories(
                condition_data, layers, ma, output_path, sae_mode=sae_mode,
            )
            subprocess.Popen(["open", output_path])
            continue

        # ── delta [MA] ──
        if user_input.lower().startswith('delta'):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    ma = max(1, int(parts[1]))
                except ValueError:
                    print("Usage: delta [MA_window]")
                    continue

            Path("plots").mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ma_str = f"_ma{ma}" if ma > 1 else ""
            output_path = f"plots/experiment_trajectory_delta{sae_str}{filter_str}{ma_str}_{ts}.png"

            plot_delta_trajectories(
                condition_data, layers, ma, output_path, sae_mode=sae_mode,
            )
            subprocess.Popen(["open", output_path])
            continue

        if not sae_mode:
            # Non-SAE mode only supports pca
            try:
                ma = max(1, int(user_input))
                Path("plots").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                ma_str = f"_ma{ma}" if ma > 1 else ""
                output_path = f"plots/experiment_trajectory{sae_str}{filter_str}{ma_str}_{ts}.png"
                plot_experiment_trajectories(condition_data, layers, ma, output_path, sae_mode=False)
                subprocess.Popen(["open", output_path])
            except ValueError:
                print("Unknown command. Try 'pca [MA]' or 'q'.")
            continue

        # ── SAE-only commands below ──
        key = "sae_features"

        # ── diff [E L] ──
        if user_input.lower().startswith('diff'):
            parts = user_input.split()
            early = int(parts[1]) if len(parts) > 1 else 5
            late = int(parts[2]) if len(parts) > 2 else 5

            # Average diff across all conditions
            diffs = []
            for name, layer_dict in condition_data:
                mat = layer_dict[key]
                # Average A+B
                model_a = mat[0::2]
                model_b = mat[1::2]
                min_len = min(len(model_a), len(model_b))
                combined = (model_a[:min_len] + model_b[:min_len]) / 2.0
                diffs.append(combined[-late:].mean(axis=0) - combined[:early].mean(axis=0))
            diff = np.mean(diffs, axis=0)

            top_features = np.argsort(diff)[::-1][:TOP_K]
            feature_labels = fetch_neuronpedia_labels(top_features)

            print(f"\nTop {TOP_K} features by activation increase (last {late} - first {early} turns):")
            print(f"{'Feature':>10} {'Diff':>10}  Label")
            print("-" * 70)
            for idx in top_features:
                label = feature_labels.get(idx, "")
                print(f"{idx:>10} {diff[idx]:>10.2f}  {label}")

            # Plot feature evolution
            Path("plots").mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"plots/experiment_trajectory_sae_diff_{ts}.png"
            plot_feature_evolution(condition_data, top_features, feature_labels, output_path)
            subprocess.Popen(["open", output_path])
            continue

        # ── end [N] ──
        if user_input.lower().startswith('end'):
            parts = user_input.split()
            last_n = int(parts[1]) if len(parts) > 1 else 2

            # Average late activations across conditions
            lates = []
            for name, layer_dict in condition_data:
                mat = layer_dict[key]
                model_a = mat[0::2]
                model_b = mat[1::2]
                min_len = min(len(model_a), len(model_b))
                combined = (model_a[:min_len] + model_b[:min_len]) / 2.0
                lates.append(combined[-last_n:].mean(axis=0))
            late_mean = np.mean(lates, axis=0)

            top_features = np.argsort(late_mean)[::-1][:TOP_K]
            feature_labels = fetch_neuronpedia_labels(top_features)

            print(f"\nTop {TOP_K} features by mean activation in last {last_n} turns:")
            print(f"{'Feature':>10} {'Late':>10}  Label")
            print("-" * 70)
            for idx in top_features:
                label = feature_labels.get(idx, "")
                print(f"{idx:>10} {late_mean[idx]:>10.2f}  {label}")

            # Plot feature evolution
            Path("plots").mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"plots/experiment_trajectory_sae_end_{ts}.png"
            plot_feature_evolution(condition_data, top_features, feature_labels, output_path)
            subprocess.Popen(["open", output_path])
            continue

        # ── Feature IDs (comma-separated) ──
        try:
            feature_ids = np.array([int(x.strip()) for x in user_input.split(",")])
        except ValueError:
            print("Unknown command. Try 'pca', 'diff', 'end', feature IDs, or 'q'.")
            continue

        n_features_total = condition_data[0][1][key].shape[1]
        out_of_range = [i for i in feature_ids if i < 0 or i >= n_features_total]
        if out_of_range:
            print(f"Feature IDs out of range [0, {n_features_total}): {out_of_range}")
            continue

        top_features = feature_ids
        feature_labels = fetch_neuronpedia_labels(feature_ids)

        Path("plots").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"plots/experiment_trajectory_sae_features_{ts}.png"
        plot_feature_evolution(condition_data, feature_ids, feature_labels, output_path)
        subprocess.Popen(["open", output_path])


# ── 7. Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment Trajectory Comparison")
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
        "--layers", type=int, nargs='+', default=DEFAULT_LAYERS,
        help=f"Layers to extract (default: {DEFAULT_LAYERS})",
    )
    parser.add_argument(
        "--max-messages", type=int, default=None,
        help="Limit messages per transcript",
    )
    parser.add_argument(
        "--max-transcripts", type=int, default=None,
        help="Limit transcripts per condition",
    )
    parser.add_argument(
        "--ma", type=int, default=1,
        help="Moving average window (default: 1 = no smoothing)",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Use cached .npz only, don't load model",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Cache root (default: auto based on mode)",
    )
    parser.add_argument(
        "--sae", action="store_true",
        help="Encode activations through JumpReLU SAE into feature vectors",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="L2-normalize each activation vector to unit norm",
    )
    parser.add_argument(
        "--diffstart", action="store_true",
        help="Subtract first activation from all (trajectories start at origin)",
    )
    parser.add_argument(
        "--diffend", action="store_true",
        help="Subtract last activation from all (trajectories end at origin)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    sae_mode = args.sae

    # In SAE mode, override layers to just SAE_LAYER
    if sae_mode:
        layers = [SAE_LAYER]
    else:
        layers = args.layers

    # Build cache dir
    if args.cache_dir:
        cache_base = Path(args.cache_dir)
    elif sae_mode:
        cache_base = Path(DEFAULT_CACHE_DIR_SAE)
    else:
        cache_base = Path(DEFAULT_CACHE_DIR)

    if sae_mode:
        msgs_str = f"M{args.max_messages}" if args.max_messages else "Mall"
        cache_dir = cache_base / msgs_str
    else:
        layers_str = "L" + "_".join(str(l) for l in sorted(layers))
        msgs_str = f"_M{args.max_messages}" if args.max_messages else "_Mall"
        cache_dir = cache_base / f"{layers_str}{msgs_str}"

    # 1. Discover conditions
    conditions = discover_conditions(base_dir, args.opener, args.system)
    if not conditions:
        print(f"No conditions found in {base_dir}")
        return 1

    print(f"Found {len(conditions)} conditions: {[name for name, _ in conditions]}")

    # 2. Check if we need the model
    need_model = False
    if not args.skip_extraction:
        for name, cond_dir in conditions:
            cond_cache = cache_dir / name
            transcript_paths = sorted(cond_dir.glob("*.txt"), key=natural_sort_key)
            if args.max_transcripts:
                transcript_paths = transcript_paths[:args.max_transcripts]
            for tpath in transcript_paths:
                if not (cond_cache / f"{tpath.stem}.npz").exists():
                    need_model = True
                    break
            if need_model:
                break

    model, tokenizer, device = (None, None, None)
    sae = None
    if need_model:
        model, tokenizer, device = load_model(MODEL_NAME)
        if sae_mode:
            sae = JumpReLUSAE(SAE_REPO, SAE_FOLDER, device="cpu")

    # 3. Process each condition
    condition_cache_paths = {}
    for name, cond_dir in conditions:
        print(f"\nCondition: {name}")
        cache_paths = process_condition(
            name, cond_dir, model, tokenizer, layers, device,
            args.max_messages, args.max_transcripts, cache_dir,
            args.skip_extraction, sae=sae,
        )
        condition_cache_paths[name] = cache_paths

    # Free model
    if model is not None:
        del model
        if sae is not None:
            del sae
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # 4. Aggregate per condition
    condition_data = []
    min_msgs_global = float('inf')

    for name, _ in conditions:
        cache_paths = condition_cache_paths[name]
        layer_dict, min_msgs = aggregate_condition(cache_paths, layers, sae_mode=sae_mode)
        if layer_dict is None:
            print(f"  Warning: no cached data for {name}, skipping")
            continue
        condition_data.append((name, layer_dict, min_msgs))
        if min_msgs < min_msgs_global:
            min_msgs_global = min_msgs
        n_cached = sum(1 for p in cache_paths if p.exists())
        print(f"  {name}: {n_cached} transcripts, {min_msgs} msgs each")

    if not condition_data:
        print("No condition data available. Run extraction first.")
        return 1

    # 5. Truncate all conditions to global min messages + optional transforms
    print(f"\nGlobal min messages: {min_msgs_global}")
    if args.normalize:
        print("L2-normalizing activation vectors")
    if args.diffstart:
        print("Subtracting first activation (--diffstart)")
    if args.diffend:
        print("Subtracting last activation (--diffend)")
    truncated = []
    for name, layer_dict, _ in condition_data:
        trunc_dict = {}
        for key, mat in layer_dict.items():
            mat = mat[:min_msgs_global]
            if args.normalize:
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                mat = mat / norms
            if args.diffstart:
                mat = mat - mat[0:1]
            if args.diffend:
                mat = mat - mat[-1:]
            trunc_dict[key] = mat
        truncated.append((name, trunc_dict))

    # 6. Build filter string for filenames
    filter_parts = []
    if args.opener:
        filter_parts.append(args.opener)
    if args.system:
        filter_parts.append(args.system)
    filter_str = "_" + "_".join(filter_parts) if filter_parts else ""
    sae_str = "_sae" if sae_mode else ""

    # 7. Interactive loop
    interactive_loop(truncated, layers, sae_mode, filter_str, sae_str, args.ma)

    return 0


if __name__ == "__main__":
    exit(main())
