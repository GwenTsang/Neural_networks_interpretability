#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimize_paren_neuron.py

DDP (2 GPUs) + Optuna to maximize "R2 local" (best single neuron) for inside-parentheses state.

Fixes:
- No dynamic categorical spaces (Optuna error).
- Prunes unsafe batch sizes deterministically.
- OOM-safe inside DDP; failed trials return r2=-1 rather than crashing.
- lambda_l1_h is a real penalty on LSTM hidden outputs (out.abs().mean()).

Run:
  python optimize_paren_neuron.py --trials 50 --epochs 20 --gpus 0,1 --out-dir ./optuna_runs
"""

from __future__ import annotations

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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
import optuna


UNK_CHAR = "\u0000"


# -------------------------
# Paragraph cache / preprocessing
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
    out: List[str] = []
    for fp in files:
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        ps = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        out.extend(ps)
    return out


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
    digit_map: str  # "none" | "hash" | "0"
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

    freq.setdefault(UNK_CHAR, 10**9)
    for ch in force_keep:
        freq.setdefault(ch, 10**9)

    chars = sorted([ch for ch, c in freq.items() if c >= min_char_freq])
    chars = sorted(set(chars))
    if UNK_CHAR not in chars:
        chars.insert(0, UNK_CHAR)
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

    def forward(
        self,
        x: torch.Tensor,
        hc: Tuple[torch.Tensor, torch.Tensor],
        return_out: bool = False,
    ):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb, hc)   # out: (B,T,H)
        out_d = self.drop(out)
        logits = self.fc(out_d)            # (B,T,V)
        if return_out:
            return logits, (h, c), out
        return logits, (h, c)


def init_forget_gate_bias(lstm: nn.LSTM, forget_bias: float) -> None:
    with torch.no_grad():
        for layer in range(lstm.num_layers):
            for name in (f"bias_ih_l{layer}", f"bias_hh_l{layer}"):
                b = getattr(lstm, name)
                h = lstm.hidden_size
                # gate order i, f, g, o
                b[h:2*h].fill_(forget_bias)


def detach_hidden(hc: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return (hc[0].detach(), hc[1].detach())


# -------------------------
# Probe & R2 local
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
def cell_states_last_layer(model: CharLSTM, s: str, char2int: Dict[str, int], device: torch.device, *, max_len: int) -> np.ndarray:
    if len(s) > max_len:
        s = s[:max_len]

    unk = char2int[UNK_CHAR]
    idxs = [char2int.get(ch, unk) for ch in s]
    x = torch.tensor(idxs, device=device, dtype=torch.long).unsqueeze(0)

    h, c = model.init_hidden(1, device)
    model.lstm.flatten_parameters()
    Cs: List[np.ndarray] = []
    for t in range(x.size(1)):
        emb = model.embedding(x[:, t:t+1])
        _, (h, c) = model.lstm(emb, (h, c))
        Cs.append(c[-1, 0].float().cpu().numpy())
    return np.stack(Cs, axis=0).astype(np.float32)


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
            X = cell_states_last_layer(model, p, char2int, device, max_len=max_len)

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
        X0 = np.zeros((10, model.hidden_size), np.float32)
        y0 = np.zeros((10,), np.float32)
        return X0, X0, y0, y0

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    langs_arr = np.concatenate(langs, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=langs_arr
    )
    return X_train, X_test, y_train, y_test


def ridge_r2_local_vectorized(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, *, alpha: float) -> Tuple[float, int]:
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


# -------------------------
# DDP execution
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

    # model
    embedding_dim: int
    hidden_size: int
    n_layers: int
    dropout: float
    forget_bias: float

    # training
    seq_length: int
    batch_size_per_gpu: int
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_l1_h: float
    epochs: int

    # probe
    probe_alpha: float
    probe_max_paras_per_lang: int
    probe_max_len: int
    probe_subsample_every: int
    probe_exclude_parens: bool
    probe_test_size: float

    seed: int


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ddp_setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _ddp_cleanup() -> None:
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _safe_barrier(rank: int) -> None:
    try:
        dist.barrier(device_ids=[rank])
    except TypeError:
        dist.barrier()


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

        blob = torch.load(blob_path, map_location="cpu")
        vocab_size = int(blob["vocab_size"])
        char2int = dict(blob["char2int_items"])
        if UNK_CHAR not in char2int:
            char2int[UNK_CHAR] = len(char2int)

        encoded = blob["encoded_int32"].to(device, non_blocking=True).long()
        n_tokens = int(encoded.numel())
        if n_tokens < 100000:
            if rank == 0:
                result_q.put({"r2_local": -1.0, "best_neuron": -1})
            return

        model = CharLSTM(
            vocab_size=vocab_size,
            embedding_dim=params.embedding_dim,
            hidden_size=params.hidden_size,
            n_layers=params.n_layers,
            dropout=params.dropout,
        ).to(device)
        init_forget_gate_bias(model.lstm, params.forget_bias)
        model.lstm.flatten_parameters()

        ddp = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        opt = torch.optim.AdamW(ddp.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        B = params.batch_size_per_gpu
        T = params.seq_length
        global_B = B * world_size

        # OOM flag to avoid deadlocks
        oom_flag = torch.zeros((), device=device, dtype=torch.int32)

        ddp.train()
        rng = torch.Generator(device=device)
        rng.manual_seed(params.seed + 999 * rank + 17)

        for epoch in range(params.epochs):
            # align offset across ranks
            if rank == 0:
                off = int(torch.randint(0, T, (1,), generator=rng, device=device).item())
            else:
                off = 0
            off_t = torch.tensor(off, device=device, dtype=torch.int64)
            dist.broadcast(off_t, src=0)
            off = int(off_t.item())

            usable = n_tokens - off - 1
            n_batches = usable // (global_B * T)
            if n_batches < 1:
                if rank == 0:
                    result_q.put({"r2_local": -1.0, "best_neuron": -1})
                return

            span = n_batches * global_B * T + 1
            data = encoded[off: off + span]

            mat_x = data[:-1].reshape(global_B, -1)
            mat_y = data[1:].reshape(global_B, -1)

            r0, r1 = rank * B, (rank + 1) * B
            streams = mat_x[r0:r1].contiguous()
            streams_y = mat_y[r0:r1].contiguous()

            hc = ddp.module.init_hidden(B, device)

            steps = streams.size(1) // T
            for s in range(steps):
                x = streams[:, s*T:(s+1)*T]
                y = streams_y[:, s*T:(s+1)*T]

                hc = detach_hidden(hc)
                opt.zero_grad(set_to_none=True)

                oom_flag.zero_()
                try:
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                        logits, hc, out = ddp(x, hc, return_out=True)  # out: (B,T,H)
                        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                        if params.lambda_l1_h > 0.0:
                            loss = loss + params.lambda_l1_h * out.abs().mean()

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
                        result_q.put({"r2_local": -1.0, "best_neuron": -1})
                    return

        if rank == 0:
            X_train, X_test, y_train, y_test = build_probe_dataset_two_files(
                model=ddp.module,
                char2int=char2int,
                device=device,
                fr_path=Path(eval_fr_path),
                en_path=Path(eval_en_path),
                max_paras_per_lang=params.probe_max_paras_per_lang,
                max_len=params.probe_max_len,
                exclude_paren_positions=params.probe_exclude_parens,
                subsample_every=params.probe_subsample_every,
                test_size=params.probe_test_size,
                seed=params.seed,
                norm=norm,
            )
            r2_local, best_neuron = ridge_r2_local_vectorized(
                X_train, y_train, X_test, y_test, alpha=params.probe_alpha
            )
            result_q.put({"r2_local": float(r2_local), "best_neuron": int(best_neuron)})

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
) -> Tuple[float, int]:
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

    res = q.get()
    return float(res["r2_local"]), int(res["best_neuron"])


# -------------------------
# Search space + pruning for batch safety
# -------------------------

def max_batch_safe(hidden: int, seq: int, layers: int) -> int:
    # Conservative heuristic for ~16GB GPUs (T4). Avoids most OOM configs.
    budget = 100_000_000
    denom = max(1, hidden * seq * layers)
    return int(max(32, budget // denom))


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> TrialParams:
    max_pair_dist = trial.suggest_categorical("max_pair_dist", [80, 120, 150, 250, 400, 600])
    max_depth = trial.suggest_categorical("max_depth", [1, 2, 4, 8])
    n_paragraphs = trial.suggest_categorical("n_paragraphs", [600, 1000, 1600, 2400, 3200])
    min_char_freq = trial.suggest_categorical("min_char_freq", [1, 2, 5, 10, 20])

    lowercase = trial.suggest_categorical("lowercase", [True, False])
    fold_diacritics = trial.suggest_categorical("fold_diacritics", [False, True])
    digit_map = trial.suggest_categorical("digit_map", ["none", "hash", "0"])
    collapse_whitespace = trial.suggest_categorical("collapse_whitespace", [False, True])

    hidden_size = trial.suggest_categorical("hidden_size", [256, 384, 512, 768, 1024])
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    forget_bias = trial.suggest_float("forget_bias", 0.0, 3.0)

    seq_length = trial.suggest_categorical("seq_length", [128, 192, 256, 384, 512])

    # FIX: static distribution (no dynamic candidates)
    batch_size_per_gpu = trial.suggest_int("batch_size_per_gpu", 64, 1024, step=64)

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    grad_clip = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0, 5.0, 10.0])

    lambda_l1_h = trial.suggest_categorical("lambda_l1_h", [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5])

    probe_alpha = trial.suggest_categorical("probe_alpha", [0.1, 1.0, 10.0])
    probe_subsample_every = trial.suggest_categorical("probe_subsample_every", [1, 2, 3])

    # prune unsafe batch sizes deterministically (pre-DDP)
    safe = max_batch_safe(hidden_size, seq_length, n_layers)
    trial.set_user_attr("max_batch_safe", int(safe))
    if batch_size_per_gpu > safe:
        raise optuna.TrialPruned(f"batch_size_per_gpu={batch_size_per_gpu} > safe={safe}")

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
        n_layers=n_layers,
        dropout=dropout,
        forget_bias=forget_bias,
        seq_length=seq_length,
        batch_size_per_gpu=batch_size_per_gpu,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        lambda_l1_h=lambda_l1_h,
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
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpus", type=str, default="0,1")

    p.add_argument("--data-dir", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="./optuna_runs")
    p.add_argument("--study-name", type=str, default="paren_r2_local_v2")
    p.add_argument("--storage", type=str, default="", help="e.g. sqlite:///study.db")

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

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
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

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args)
        trial_id = f"trial_{trial.number:05d}"
        try:
            r2, best_neuron = run_ddp_trial(
                params,
                out_dir=out_dir,
                cache_path=cache_path,
                train_files=train_files,
                eval_fr=eval_fr,
                eval_en=eval_en,
                trial_id=trial_id,
                world_size=world_size,
            )
            trial.set_user_attr("best_neuron", best_neuron)
            return r2
        except Exception as e:
            trial.set_user_attr("exception", repr(e))
            if _is_oom(e):
                return -1.0
            return -1.0

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True, catch=(Exception,))

    best = study.best_trial
    summary = {
        "best_value_r2_local": float(best.value),
        "best_trial_number": int(best.number),
        "best_params": dict(best.params),
        "best_neuron": best.user_attrs.get("best_neuron", None),
        "max_batch_safe": best.user_attrs.get("max_batch_safe", None),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "best_result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
