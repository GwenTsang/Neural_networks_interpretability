#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimize_paren_neuron_2layers.py

Optuna hyperparameter search to maximize "R2 local" (best single LSTM cell-state neuron)
for inside-parentheses detection.

Key changes vs previous version:
- n_layers fixed to 2.
- Probes BOTH layers (not just the last one) to find the best neuron.
- Reports which layer contains the best neuron.

Example:
  python optimize_paren_neuron_2layers.py --trials 40 --epochs 20 --gpus 0,1 --out-dir ./optuna_runs
"""

from __future__ import annotations

# IMPORTANT: set allocator conf before importing torch
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import pickle
import random
import re
import shutil
import socket
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.model_selection import train_test_split

import optuna


UNK_CHAR = "\u0000"  # unknown char token
N_LAYERS = 2  # Fixed: always use 2 layers


# -------------------------
# Text preprocessing & paragraph cache
# -------------------------

@dataclass(frozen=True)
class ParagraphInfo:
    text: str
    has_pair: bool
    balanced: bool
    max_pair_dist: int
    max_depth: int


def _analyze_parentheses(par: str) -> Tuple[bool, bool, int, int]:
    has_pair = "(" in par and ")" in par
    stack: List[int] = []
    max_dist = 0
    depth = 0
    max_depth = 0
    balanced = True

    for i, ch in enumerate(par):
        if ch == "(":
            stack.append(i)
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            if not stack:
                balanced = False
                break
            start = stack.pop()
            max_dist = max(max_dist, i - start)
            depth = max(0, depth - 1)

    if stack:
        balanced = False

    return has_pair, balanced, max_dist, max_depth


def load_paragraphs(files: Sequence[Path]) -> List[str]:
    paragraphs: List[str] = []
    for fp in files:
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        ps = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        paragraphs.extend(ps)
    return paragraphs


def build_paragraph_cache(files: Sequence[Path], cache_path: Path, force: bool = False) -> List[ParagraphInfo]:
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    raw = load_paragraphs(files)
    infos: List[ParagraphInfo] = []
    for p in raw:
        has_pair, balanced, max_dist, max_depth = _analyze_parentheses(p)
        infos.append(ParagraphInfo(p, has_pair, balanced, max_dist, max_depth))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(infos, f)
    return infos


@dataclass(frozen=True)
class TextNormConfig:
    lowercase: bool
    fold_diacritics: bool
    digit_map: str       # "none" | "hash" | "0"
    collapse_whitespace: bool


def normalize_text(s: str, cfg: TextNormConfig) -> str:
    if cfg.lowercase:
        s = s.lower()

    if cfg.fold_diacritics:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))

    if cfg.digit_map != "none":
        repl = "#" if cfg.digit_map == "hash" else "0"
        s = re.sub(r"\d", repl, s)

    if cfg.collapse_whitespace:
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)

    return s


def make_training_text(
    infos: Sequence[ParagraphInfo],
    *,
    max_pair_dist: int,
    max_depth: int,
    n_paragraphs: int,
    seed: int,
    norm: TextNormConfig,
) -> str:
    candidates = [
        pi.text for pi in infos
        if pi.has_pair and pi.balanced and pi.max_pair_dist <= max_pair_dist and pi.max_depth <= max_depth
    ]
    if not candidates:
        return ""

    rng = np.random.default_rng(seed)
    if n_paragraphs < len(candidates):
        idx = rng.choice(len(candidates), size=n_paragraphs, replace=False)
        chosen = [candidates[i] for i in idx]
    else:
        chosen = candidates

    chosen = [normalize_text(p, norm) for p in chosen]
    return "\n\n".join(chosen)


def build_vocab_and_encode(
    text: str,
    *,
    min_char_freq: int,
    force_keep: Sequence[str] = ("(", ")", "\n", " "),
) -> Tuple[Dict[str, int], np.ndarray]:
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    # Force keep UNK and critical chars
    freq.setdefault(UNK_CHAR, 10**9)
    for ch in force_keep:
        freq.setdefault(ch, 10**9)

    chars = sorted([ch for ch, c in freq.items() if c >= min_char_freq])
    chars = sorted(set(chars))
    if UNK_CHAR not in chars:
        chars.insert(0, UNK_CHAR)
    if "(" not in chars:
        chars.append("(")
    if ")" not in chars:
        chars.append(")")

    chars = sorted(set(chars))
    char2int = {ch: i for i, ch in enumerate(chars)}
    unk = char2int[UNK_CHAR]
    encoded = np.fromiter((char2int.get(ch, unk) for ch in text), dtype=np.int64)
    return char2int, encoded


# -------------------------
# Model
# -------------------------

class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward(self, x: torch.Tensor, hc: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb, hc)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, (h, c)


def init_forget_gate_bias(lstm: nn.LSTM, forget_bias: float) -> None:
    # Gate order: i, f, g, o
    with torch.no_grad():
        for layer in range(lstm.num_layers):
            for name in (f"bias_ih_l{layer}", f"bias_hh_l{layer}"):
                b = getattr(lstm, name)
                h = lstm.hidden_size
                b[h:2*h].fill_(forget_bias)


def detach_hidden(hc: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return (hc[0].detach(), hc[1].detach())


# -------------------------
# Probe dataset & R2 local (ALL LAYERS)
# -------------------------

def inside_after_reading(s: str) -> np.ndarray:
    d = 0
    y = np.zeros(len(s), dtype=np.int64)
    for t, ch in enumerate(s):
        if ch == "(":
            d += 1
        elif ch == ")":
            d = max(0, d - 1)
        y[t] = 1 if d > 0 else 0
    return y


@torch.no_grad()
def cell_states_all_layers(
    model: CharLSTM,
    s: str,
    char2int: Dict[str, int],
    device: torch.device,
    *,
    max_len: int,
) -> np.ndarray:
    """
    Extract cell states from ALL layers at each timestep.
    
    Returns:
        np.ndarray of shape (T, n_layers * hidden_size)
        where the first hidden_size elements are layer 0,
        the next hidden_size elements are layer 1, etc.
    """
    if len(s) > max_len:
        s = s[:max_len]

    unk = char2int[UNK_CHAR]
    idxs = [char2int.get(ch, unk) for ch in s]
    x = torch.tensor(idxs, device=device, dtype=torch.long).unsqueeze(0)  # (1,T)

    h, c = model.init_hidden(1, device)
    Cs: List[np.ndarray] = []
    model.lstm.flatten_parameters()

    for t in range(x.size(1)):
        emb = model.embedding(x[:, t:t+1])
        _, (h, c) = model.lstm(emb, (h, c))
        # c has shape (n_layers, 1, hidden_size)
        # Flatten all layers: (n_layers * hidden_size,)
        c_all = c[:, 0, :].flatten().float().cpu().numpy()
        Cs.append(c_all)

    return np.stack(Cs, axis=0).astype(np.float32)  # (T, n_layers * hidden_size)


def build_probe_dataset_two_files(
    model: CharLSTM,
    char2int: Dict[str, int],
    device: torch.device,
    *,
    fr_path: Path,
    en_path: Path,
    max_paras_per_lang: int,
    max_len: int,
    exclude_paren_positions: bool,
    subsample_every: int,
    test_size: float,
    seed: int,
    norm: TextNormConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    model.eval()

    def paragraphs(path: Path) -> List[str]:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        ps = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        ps = [p for p in ps if "(" in p and ")" in p]
        if len(ps) > max_paras_per_lang:
            idx = rng.choice(len(ps), size=max_paras_per_lang, replace=False)
            ps = [ps[i] for i in idx]
        return ps

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    langs: List[np.ndarray] = []

    for lang, path in (("fr", fr_path), ("en", en_path)):
        for p in paragraphs(path):
            p = normalize_text(p, norm)
            if len(p) > max_len:
                start = int(rng.integers(0, len(p) - max_len + 1))
                p = p[start:start + max_len]

            y = inside_after_reading(p)
            # Use all layers now
            X = cell_states_all_layers(model, p, char2int, device, max_len=max_len)

            mask = np.ones(len(p), dtype=bool)
            if exclude_paren_positions:
                mask &= np.array([c not in "()" for c in p], dtype=bool)
            if subsample_every > 1:
                take = np.zeros(len(p), dtype=bool)
                take[::subsample_every] = True
                mask &= take

            if mask.sum() < 10:
                continue

            Xs.append(X[mask])
            ys.append(y[mask])
            langs.append(np.full(mask.sum(), lang, dtype=object))

    if not Xs:
        # Degenerate: no probe data
        total_neurons = model.hidden_size * model.n_layers
        X0 = np.zeros((10, total_neurons), np.float32)
        y0 = np.zeros((10,), np.float32)
        return X0, X0, y0, y0

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    langs_arr = np.concatenate(langs, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=langs_arr
    )
    return X_train, X_test, y_train, y_test


def ridge_r2_local_vectorized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float,
) -> Tuple[float, int]:
    """
    Best single-feature ridge R2 (with intercept), vectorized.
    Matches sklearn Ridge objective: ||y - Xw||^2 + alpha ||w||^2 (no 1/n).
    
    Returns:
        (best_r2, best_neuron_index) where best_neuron_index is the global index
        across all layers (0..n_layers*hidden_size-1)
    """
    Xtr = X_train.astype(np.float64, copy=False)
    Xte = X_test.astype(np.float64, copy=False)
    ytr = y_train.astype(np.float64, copy=False)
    yte = y_test.astype(np.float64, copy=False)

    x_mean = Xtr.mean(axis=0)
    y_mean = ytr.mean()
    Xtr_c = Xtr - x_mean
    ytr_c = ytr - y_mean

    cov = Xtr_c.T @ ytr_c
    var = np.sum(Xtr_c * Xtr_c, axis=0)
    w = cov / (var + alpha)

    Xte_c = Xte - x_mean
    a = yte - y_mean

    Sa2 = float(np.sum(a * a))
    cross = Xte_c.T @ a
    var_te = np.sum(Xte_c * Xte_c, axis=0)

    sse = Sa2 - 2.0 * (w * cross) + (w * w) * var_te

    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    if ss_tot <= 1e-12:
        best_i = int(np.argmin(sse))
        return 0.0, best_i

    r2 = 1.0 - (sse / ss_tot)
    best_i = int(np.argmax(r2))
    return float(r2[best_i]), best_i


def decode_neuron_index(global_idx: int, hidden_size: int) -> Tuple[int, int]:
    """
    Given a global neuron index (0..n_layers*hidden_size-1),
    return (layer_index, neuron_index_within_layer).
    """
    layer = global_idx // hidden_size
    neuron = global_idx % hidden_size
    return layer, neuron


# -------------------------
# DDP worker / trial execution
# -------------------------

@dataclass(frozen=True)
class TrialParams:
    # data / preprocessing
    max_pair_dist: int
    max_depth: int
    n_paragraphs: int
    min_char_freq: int
    lowercase: bool
    fold_diacritics: bool
    digit_map: str
    collapse_whitespace: bool

    # model (n_layers is fixed to 2)
    embedding_dim: int
    hidden_size: int
    dropout: float
    forget_bias: float

    # training
    seq_length: int
    batch_size_per_gpu: int
    lr: float
    weight_decay: float
    grad_clip: float
    epochs: int

    # probe
    probe_alpha: float
    probe_max_paras_per_lang: int
    probe_max_len: int
    probe_subsample_every: int
    probe_exclude_parens: bool
    probe_test_size: float

    # misc
    seed: int


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _safe_barrier(rank: int) -> None:
    try:
        dist.barrier(device_ids=[rank])
    except TypeError:
        dist.barrier()


def _ddp_setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _ddp_cleanup() -> None:
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "cuda out of memory" in s


def ddp_worker(
    rank: int,
    world_size: int,
    port: int,
    params: TrialParams,
    train_files: List[str],
    cache_path: str,
    eval_fr_path: str,
    eval_en_path: str,
    trial_dir: str,
    result_q: mp.SimpleQueue,
) -> None:
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    _ddp_setup(rank, world_size, port)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    try:
        infos = build_paragraph_cache([Path(p) for p in train_files], Path(cache_path), force=False)

        norm = TextNormConfig(
            lowercase=params.lowercase,
            fold_diacritics=params.fold_diacritics,
            digit_map=params.digit_map,
            collapse_whitespace=params.collapse_whitespace,
        )

        trial_dir_p = Path(trial_dir)
        trial_dir_p.mkdir(parents=True, exist_ok=True)
        blob_path = trial_dir_p / "train_blob.pt"

        # Rank0 builds text/vocab/encoded; others read
        if rank == 0:
            text = make_training_text(
                infos,
                max_pair_dist=params.max_pair_dist,
                max_depth=params.max_depth,
                n_paragraphs=params.n_paragraphs,
                seed=params.seed,
                norm=norm,
            )
            if len(text) < 20000:
                char2int, encoded = build_vocab_and_encode(text, min_char_freq=1)
            else:
                char2int, encoded = build_vocab_and_encode(text, min_char_freq=params.min_char_freq)

            blob = {
                "char2int_items": list(char2int.items()),
                "vocab_size": len(char2int),
                "encoded_int32": torch.tensor(encoded, dtype=torch.int32),
            }
            torch.save(blob, blob_path)

        _safe_barrier(rank)

        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        vocab_size = int(blob["vocab_size"])
        char2int = dict(blob["char2int_items"])
        if UNK_CHAR not in char2int:
            char2int[UNK_CHAR] = len(char2int)

        encoded = blob["encoded_int32"].to(device, non_blocking=True).long()
        n_tokens = int(encoded.numel())
        if n_tokens < 100000:
            if rank == 0:
                result_q.put({
                    "r2_local": -1.0,
                    "best_neuron_global": -1,
                    "best_layer": -1,
                    "best_neuron_in_layer": -1,
                })
            _safe_barrier(rank)
            return

        # Model with fixed n_layers=2
        model = CharLSTM(
            vocab_size=vocab_size,
            embedding_dim=params.embedding_dim,
            hidden_size=params.hidden_size,
            n_layers=N_LAYERS,  # Fixed to 2
            dropout=params.dropout,
        ).to(device)
        init_forget_gate_bias(model.lstm, params.forget_bias)
        model.lstm.flatten_parameters()

        ddp = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        opt = torch.optim.AdamW(ddp.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        criterion = nn.CrossEntropyLoss()

        scaler = torch.amp.GradScaler("cuda", enabled=True)

        # Stateful TBPTT
        B = params.batch_size_per_gpu
        T = params.seq_length

        rng = torch.Generator(device=device)
        rng.manual_seed(params.seed + 1234 * rank + 17)

        ddp.train()

        oom_flag = torch.zeros((), device=device, dtype=torch.int32)

        for epoch in range(params.epochs):
            if rank == 0:
                off = int(torch.randint(0, T, (1,), generator=rng, device=device).item())
            else:
                off = 0
            off_t = torch.tensor(off, device=device, dtype=torch.int64)
            dist.broadcast(off_t, src=0)
            off = int(off_t.item())

            usable = n_tokens - off - 1
            global_B = B * world_size
            n_batches = usable // (global_B * T)
            if n_batches < 1:
                if rank == 0:
                    result_q.put({
                        "r2_local": -1.0,
                        "best_neuron_global": -1,
                        "best_layer": -1,
                        "best_neuron_in_layer": -1,
                    })
                _safe_barrier(rank)
                return

            span = n_batches * global_B * T + 1
            data = encoded[off: off + span]

            mat = data[:-1].reshape(global_B, -1)
            mat_y = data[1:].reshape(global_B, -1)

            row0 = rank * B
            row1 = (rank + 1) * B
            streams = mat[row0:row1].contiguous()
            streams_y = mat_y[row0:row1].contiguous()

            hc = ddp.module.init_hidden(B, device)

            total_steps = streams.size(1) // T
            for step in range(total_steps):
                x = streams[:, step*T:(step+1)*T]
                y = streams_y[:, step*T:(step+1)*T]

                hc = detach_hidden(hc)
                opt.zero_grad(set_to_none=True)

                oom_flag.zero_()
                try:
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                        logits, hc = ddp(x, hc)
                        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

                    scaler.scale(loss).backward()
                    if params.grad_clip > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(ddp.parameters(), params.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                except Exception as e:
                    if _is_oom(e):
                        oom_flag.fill_(1)
                        torch.cuda.empty_cache()
                    else:
                        raise

                dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
                if int(oom_flag.item()) == 1:
                    if rank == 0:
                        result_q.put({
                            "r2_local": -1.0,
                            "best_neuron_global": -1,
                            "best_layer": -1,
                            "best_neuron_in_layer": -1,
                        })
                    _safe_barrier(rank)
                    return

        # Evaluation on rank 0
        if rank == 0:
            fr_p = Path(eval_fr_path)
            en_p = Path(eval_en_path)

            X_train, X_test, y_train, y_test = build_probe_dataset_two_files(
                model=ddp.module,
                char2int=char2int,
                device=device,
                fr_path=fr_p,
                en_path=en_p,
                max_paras_per_lang=params.probe_max_paras_per_lang,
                max_len=params.probe_max_len,
                exclude_paren_positions=params.probe_exclude_parens,
                subsample_every=params.probe_subsample_every,
                test_size=params.probe_test_size,
                seed=params.seed,
                norm=norm,
            )

            r2_local, best_neuron_global = ridge_r2_local_vectorized(
                X_train, y_train, X_test, y_test, alpha=params.probe_alpha
            )
            
            # Decode which layer the best neuron is in
            best_layer, best_neuron_in_layer = decode_neuron_index(
                best_neuron_global, params.hidden_size
            )
            
            result_q.put({
                "r2_local": float(r2_local),
                "best_neuron_global": int(best_neuron_global),
                "best_layer": int(best_layer),
                "best_neuron_in_layer": int(best_neuron_in_layer),
            })

        _safe_barrier(rank)

    finally:
        _ddp_cleanup()


def run_ddp_trial(
    params: TrialParams,
    *,
    out_dir: Path,
    cache_path: Path,
    train_files: List[Path],
    eval_fr: Path,
    eval_en: Path,
    trial_id: str,
    world_size: int,
) -> Dict:
    port = _find_free_port()
    trial_dir = out_dir / "trials" / trial_id
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    q: mp.SimpleQueue = mp.SimpleQueue()

    mp.spawn(
        ddp_worker,
        args=(
            world_size,
            port,
            params,
            [str(p) for p in train_files],
            str(cache_path),
            str(eval_fr),
            str(eval_en),
            str(trial_dir),
            q,
        ),
        nprocs=world_size,
        join=True,
    )

    return q.get()


# -------------------------
# Optuna search space
# -------------------------

def _max_batch_safe(hidden: int, seq: int) -> int:
    """
    Heuristic cap to avoid OOM on ~16GB GPUs for 2-layer LSTM.
    """
    # With 2 layers, we need more memory
    budget = 80_000_000  # more conservative for 2 layers
    denom = max(1, seq * hidden * N_LAYERS)
    max_b = max(32, budget // denom)
    return int(max_b)


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> TrialParams:
    # Data / preprocessing - based on best results from 1-layer experiment
    max_pair_dist = trial.suggest_int("max_pair_dist", 200, 400, step=50)
    max_depth = trial.suggest_int("max_depth", 2, 4)
    n_paragraphs = trial.suggest_int("n_paragraphs", 2000, 4000, step=500)
    min_char_freq = trial.suggest_categorical("min_char_freq", [2, 5, 10])

    lowercase = trial.suggest_categorical("lowercase", [True, False])
    fold_diacritics = trial.suggest_categorical("fold_diacritics", [False, True])
    digit_map = trial.suggest_categorical("digit_map", ["none", "0"])
    collapse_whitespace = True  # fixed

    # Model - n_layers is fixed to 2
    hidden_size = trial.suggest_categorical("hidden_size", [256, 384, 512])
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.35)
    forget_bias = trial.suggest_float("forget_bias", 0.5, 2.0)

    # Training
    seq_length = trial.suggest_categorical("seq_length", [256, 384, 512])

    batch_size_requested = trial.suggest_int("batch_size_per_gpu", 32, 256, step=32)
    max_b = _max_batch_safe(hidden_size, seq_length)
    batch_size_per_gpu = min(batch_size_requested, max_b)
    batch_size_per_gpu = max(32, batch_size_per_gpu)
    trial.set_user_attr("batch_size_actual", batch_size_per_gpu)

    lr = trial.suggest_float("lr", 3e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    grad_clip = trial.suggest_categorical("grad_clip", [1.0, 2.0, 5.0])

    # Probe
    probe_alpha = trial.suggest_categorical("probe_alpha", [1.0, 10.0])
    probe_subsample_every = trial.suggest_categorical("probe_subsample_every", [1, 2])

    return TrialParams(
        max_pair_dist=max_pair_dist,
        max_depth=max_depth,
        n_paragraphs=n_paragraphs,
        min_char_freq=min_char_freq,
        lowercase=lowercase,
        fold_diacritics=fold_diacritics,
        digit_map=digit_map,
        collapse_whitespace=collapse_whitespace,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        dropout=dropout,
        forget_bias=forget_bias,
        seq_length=seq_length,
        batch_size_per_gpu=batch_size_per_gpu,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        epochs=args.epochs,
        probe_alpha=probe_alpha,
        probe_max_paras_per_lang=args.probe_max_paras_per_lang,
        probe_max_len=args.probe_max_len,
        probe_subsample_every=probe_subsample_every,
        probe_exclude_parens=not args.probe_include_parens_positions,
        probe_test_size=args.probe_test_size,
        seed=args.seed + trial.number * 1009,
    )


# -------------------------
# CLI / main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids, e.g. '0,1'.")

    p.add_argument("--data-dir", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="./optuna_runs_2layers")
    p.add_argument("--study-name", type=str, default="paren_r2_local_2layers")
    p.add_argument("--storage", type=str, default="", help="Optuna storage URL, e.g. sqlite:///study.db")

    p.add_argument("--train-files", type=str, nargs="*", default=["french.txt", "l.txt", "all_Flaubert.txt"])
    p.add_argument("--eval-fr", type=str, default="all_Flaubert.txt")
    p.add_argument("--eval-en", type=str, default="french.txt")

    p.add_argument("--probe-max-paras-per-lang", type=int, default=200)
    p.add_argument("--probe-max-len", type=int, default=400)
    p.add_argument("--probe-test-size", type=float, default=0.2)
    p.add_argument("--probe-include-parens-positions", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_list:
        raise RuntimeError("No GPUs specified.")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    world_size = len(gpu_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_files = [data_dir / f for f in args.train_files]
    for fp in train_files:
        if not fp.exists():
            raise FileNotFoundError(f"Training file not found: {fp}")

    eval_fr = data_dir / args.eval_fr
    eval_en = data_dir / args.eval_en
    if not eval_fr.exists():
        raise FileNotFoundError(f"Eval FR file not found: {eval_fr}")
    if not eval_en.exists():
        raise FileNotFoundError(f"Eval EN file not found: {eval_en}")

    cache_path = out_dir / "cache" / "paragraph_infos.pkl"
    build_paragraph_cache(train_files, cache_path, force=False)

    if args.storage.strip():
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=args.storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )

    # Track layer statistics
    layer_counts = {0: 0, 1: 0}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args)
        trial_id = f"trial_{trial.number:05d}"
        try:
            result = run_ddp_trial(
                params,
                out_dir=out_dir,
                cache_path=cache_path,
                train_files=train_files,
                eval_fr=eval_fr,
                eval_en=eval_en,
                trial_id=trial_id,
                world_size=world_size,
            )
            
            r2 = result["r2_local"]
            trial.set_user_attr("best_neuron_global", result["best_neuron_global"])
            trial.set_user_attr("best_layer", result["best_layer"])
            trial.set_user_attr("best_neuron_in_layer", result["best_neuron_in_layer"])
            
            # Track which layer wins
            if r2 > 0 and result["best_layer"] in layer_counts:
                layer_counts[result["best_layer"]] += 1
            
            return r2
        except Exception as e:
            trial.set_user_attr("exception", repr(e))
            if _is_oom(e):
                return -1.0
            return -1.0

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True, catch=(Exception,))

    best = study.best_trial
    summary = {
        "experiment": "2-layer LSTM with both layers probed",
        "n_layers": N_LAYERS,
        "best_value_r2_local": float(best.value),
        "best_trial_number": int(best.number),
        "best_params": dict(best.params),
        "best_neuron_global": best.user_attrs.get("best_neuron_global", None),
        "best_layer": best.user_attrs.get("best_layer", None),
        "best_neuron_in_layer": best.user_attrs.get("best_neuron_in_layer", None),
        "batch_size_actual": best.user_attrs.get("batch_size_actual", None),
        "layer_win_counts": layer_counts,
    }
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("="*60)
    print(f"\nBest neuron is in LAYER {summary['best_layer']} "
          f"(neuron #{summary['best_neuron_in_layer']} within that layer)")
    print(f"Layer 0 won {layer_counts[0]} times, Layer 1 won {layer_counts[1]} times")
    print("="*60 + "\n")
    
    (out_dir / "best_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
