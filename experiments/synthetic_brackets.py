#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimize_paren_neuron_v3.py

DDP (2 GPUs) + Optuna to maximize "R2 local" (best single neuron) for inside-parentheses state.

Key improvements vs v2:
- Prevent "too few steps/epoch": constrain batch sizes + prune by token-batch size.
- Overlapping TBPTT via stride (stride = seq_length // stride_div), increases updates.
- Bracket augmentation: () -> [] / {} / <> copies added to training text.
- Optional synthetic bracket paragraphs (long spans) appended to training text.
- Probe dataset filters out OOV chars (closer to notebook behavior).

Run:
  python optimize_paren_neuron_v3.py --trials 40 --epochs 20 --gpus 0,1 --study-name paren_r2_local_v3
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import pickle
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

BRACKET_PAIRS: List[Tuple[str, str]] = [
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
]


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


def _replace_parens_with_pair(s: str, open_ch: str, close_ch: str) -> str:
    # Replace only parentheses; keep the rest unchanged
    return s.replace("(", open_ch).replace(")", close_ch)


def synth_bracket_paragraph(
    rng: np.random.Generator,
    max_span: int,
    repeats: int,
    alphabet: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.;:-",
) -> str:
    # Make a paragraph consisting of balanced bracketed spans with random filler.
    # This is purely for LM training (no labels).
    out: List[str] = []
    for _ in range(repeats):
        op, cl = BRACKET_PAIRS[int(rng.integers(0, len(BRACKET_PAIRS)))]
        L = int(rng.integers(8, max_span + 1))
        filler = "".join(rng.choice(list(alphabet), size=L, replace=True))
        out.append(op + filler + cl)
        out.append(" ")
    return "".join(out).strip()


def make_training_text(
    infos: Sequence[ParagraphInfo],
    *,
    max_pair_dist: int,
    max_depth: int,
    n_paragraphs: int,
    seed: int,
    norm: TextNormConfig,
    bracket_augment_prob: float,
    synth_n_paragraphs: int,
    synth_max_span: int,
    synth_repeats: int,
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

    augmented: List[str] = []
    for p in chosen:
        p = normalize_text(p, norm)
        augmented.append(p)

        # Add bracket-augmented copies ([], {}, <>), preserving distances/nesting structure
        if bracket_augment_prob > 0 and "(" in p and ")" in p:
            if float(rng.random()) < bracket_augment_prob:
                # pick one non-paren pair
                op, cl = BRACKET_PAIRS[int(rng.integers(1, len(BRACKET_PAIRS)))]
                augmented.append(_replace_parens_with_pair(p, op, cl))

    # Add synthetic bracket paragraphs (helps long-range delimiter memory)
    for _ in range(int(synth_n_paragraphs)):
        augmented.append(synth_bracket_paragraph(rng, max_span=synth_max_span, repeats=synth_repeats))

    return "\n\n".join(augmented)


def build_vocab_and_encode(
    text: str,
    *,
    min_char_freq: int,
) -> Tuple[Dict[str, int], np.ndarray]:
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    # Force keep UNK + all bracket chars + whitespace
    for ch in [UNK_CHAR, "\n", " ", "(", ")", "[", "]", "{", "}", "<", ">"]:
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

    def forward(self, x: torch.Tensor, hc: Tuple[torch.Tensor, torch.Tensor], return_out: bool = False):
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
                b[h:2*h].fill_(forget_bias)


def detach_hidden(hc: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return (hc[0].detach(), hc[1].detach())


# -------------------------
# Probe & R2 local (parentheses labels only)
# -------------------------

def inside_after_reading_parentheses_only(s: str) -> np.ndarray:
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

    # IMPORTANT: filter out OOV chars (closer to your notebook)
    s = "".join([c for c in s if c in char2int])
    if len(s) < 5:
        return np.zeros((0, model.hidden_size), np.float32)

    idxs = [char2int[c] for c in s]
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
            if len(p) > max_len:
                start = int(rng.integers(0, len(p) - max_len + 1))
                p = p[start:start + max_len]

            # Filter to vocab *before* making labels, so X and y align
            p = "".join([c for c in p if c in char2int])
            if len(p) < 10:
                continue

            y = inside_after_reading_parentheses_only(p)
            X = cell_states_last_layer(model, p, char2int, device, max_len=max_len)
            if X.shape[0] != len(p):
                continue

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
# DDP utilities
# -------------------------

@dataclass(frozen=True)
class TrialParams:
    # data
    max_pair_dist: int
    max_depth: int
    n_paragraphs: int
    min_char_freq: int
    norm_lowercase: bool
    norm_fold_diacritics: bool
    norm_digit_map: str
    norm_collapse_whitespace: bool

    bracket_augment_prob: float
    synth_n_paragraphs: int
    synth_max_span: int
    synth_repeats: int

    # model
    embedding_dim: int
    hidden_size: int
    n_layers: int
    dropout: float
    forget_bias: float

    # training
    seq_length: int
    stride_div: int
    batch_size_per_gpu: int
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_l1_h: float
    epochs: int
    max_steps_per_epoch: int

    # probe
    probe_alpha: float
    probe_subsample_every: int

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
            lowercase=params.norm_lowercase,
            fold_diacritics=params.norm_fold_diacritics,
            digit_map=params.norm_digit_map,
            collapse_whitespace=params.norm_collapse_whitespace,
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
                bracket_augment_prob=params.bracket_augment_prob,
                synth_n_paragraphs=params.synth_n_paragraphs,
                synth_max_span=params.synth_max_span,
                synth_repeats=params.synth_repeats,
            )
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
        encoded = blob["encoded_int32"].to(device, non_blocking=True).long()
        n_tokens = int(encoded.numel())
        if n_tokens < 200_000:
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
        stride = max(1, T // params.stride_div)
        global_B = B * world_size

        # Precompute how many "rows" we can make for streaming; we want enough columns
        # We use the full text; smaller stride gives more update steps.
        usable = n_tokens - 1
        n_cols = usable // global_B
        if n_cols <= T + 2:
            if rank == 0:
                result_q.put({"r2_local": -1.0, "best_neuron": -1})
            return

        # Make stream matrix once (cheap view) then iterate with stride
        data = encoded[: global_B * n_cols + 1]
        mat_x = data[:-1].reshape(global_B, n_cols)
        mat_y = data[1:].reshape(global_B, n_cols)

        r0, r1 = rank * B, (rank + 1) * B
        streams = mat_x[r0:r1].contiguous()
        streams_y = mat_y[r0:r1].contiguous()

        ddp.train()
        oom_flag = torch.zeros((), device=device, dtype=torch.int32)

        for epoch in range(params.epochs):
            hc = ddp.module.init_hidden(B, device)
            steps_done = 0

            for pos in range(0, n_cols - T - 1, stride):
                x = streams[:, pos:pos + T]
                y = streams_y[:, pos:pos + T]

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

                steps_done += 1
                if params.max_steps_per_epoch > 0 and steps_done >= params.max_steps_per_epoch:
                    break

        if rank == 0:
            X_train, X_test, y_train, y_test = build_probe_dataset_two_files(
                model=ddp.module,
                char2int=char2int,
                device=device,
                fr_path=Path(eval_fr_path),
                en_path=Path(eval_en_path),
                max_paras_per_lang=200,
                max_len=400,
                exclude_paren_positions=True,
                subsample_every=params.probe_subsample_every,
                test_size=0.2,
                seed=params.seed,
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
# Optuna search space + pruning (prevents low-step regimes)
# -------------------------

def suggest_params(trial: optuna.Trial, args: argparse.Namespace, world_size: int) -> TrialParams:
    # data
    max_pair_dist = trial.suggest_categorical("max_pair_dist", [150, 250, 400, 600])
    max_depth = trial.suggest_categorical("max_depth", [1, 2, 4, 8])
    n_paragraphs = trial.suggest_categorical("n_paragraphs", [1200, 2400, 3200])
    min_char_freq = trial.suggest_categorical("min_char_freq", [1, 2, 5, 10])

    norm_lowercase = trial.suggest_categorical("lowercase", [False, True])
    norm_fold = trial.suggest_categorical("fold_diacritics", [False, True])
    norm_digit = trial.suggest_categorical("digit_map", ["none", "hash", "0"])
    norm_ws = trial.suggest_categorical("collapse_whitespace", [False, True])

    # bracket augmentation (main change you asked for)
    bracket_augment_prob = trial.suggest_categorical("bracket_augment_prob", [0.0, 0.25, 0.5, 0.75])

    # synthetic bracket paragraphs (optional but helpful)
    synth_n_paragraphs = trial.suggest_categorical("synth_n_paragraphs", [0, 200, 500])
    synth_max_span = trial.suggest_categorical("synth_max_span", [64, 128, 256, 512])
    synth_repeats = trial.suggest_categorical("synth_repeats", [1, 2, 4])

    # model (bias toward notebook-ish sizes but allow smaller)
    hidden_size = trial.suggest_categorical("hidden_size", [256, 384, 512, 768])
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    forget_bias = trial.suggest_float("forget_bias", 0.0, 3.0)

    # training
    seq_length = trial.suggest_categorical("seq_length", [152, 192, 256, 384, 512])
    stride_div = trial.suggest_categorical("stride_div", [1, 2, 4])  # 1=no overlap, 2=half overlap, 4=quarter overlap

    # IMPORTANT: keep batches moderate so we get enough optimizer steps (closer to notebook)
    batch_size_per_gpu = trial.suggest_int("batch_size_per_gpu", 128, 512, step=64)

    # prune token-batch regimes that produce too few steps/epoch
    token_batch = world_size * batch_size_per_gpu * seq_length
    trial.set_user_attr("token_batch", int(token_batch))
    if token_batch > args.token_batch_max:
        raise optuna.TrialPruned(f"token_batch={token_batch} > token_batch_max={args.token_batch_max}")

    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    grad_clip = trial.suggest_categorical("grad_clip", [1.0, 2.0, 5.0])
    lambda_l1_h = trial.suggest_categorical("lambda_l1_h", [0.0, 1e-7, 3e-7, 1e-6, 3e-6])

    # probe
    probe_alpha = trial.suggest_categorical("probe_alpha", [0.1, 1.0, 10.0])
    probe_subsample_every = trial.suggest_categorical("probe_subsample_every", [1, 2, 3])

    return TrialParams(
        max_pair_dist=max_pair_dist,
        max_depth=max_depth,
        n_paragraphs=n_paragraphs,
        min_char_freq=min_char_freq,
        norm_lowercase=norm_lowercase,
        norm_fold_diacritics=norm_fold,
        norm_digit_map=norm_digit,
        norm_collapse_whitespace=norm_ws,
        bracket_augment_prob=bracket_augment_prob,
        synth_n_paragraphs=synth_n_paragraphs,
        synth_max_span=synth_max_span,
        synth_repeats=synth_repeats,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        forget_bias=forget_bias,
        seq_length=seq_length,
        stride_div=stride_div,
        batch_size_per_gpu=batch_size_per_gpu,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        lambda_l1_h=lambda_l1_h,
        epochs=args.epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        probe_alpha=probe_alpha,
        probe_subsample_every=probe_subsample_every,
        seed=args.seed + trial.number * 1009,
    )


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps-per-epoch", type=int, default=0,
                   help="Cap optimizer updates per epoch (0 = no cap). Useful with stride overlap.")
    p.add_argument("--token-batch-max", type=int, default=100_000,
                   help="Prune trials where world_size*batch_per_gpu*seq_len exceeds this.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpus", type=str, default="0,1")

    p.add_argument("--data-dir", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="./optuna_runs")
    p.add_argument("--study-name", type=str, default="paren_r2_local_v3")
    p.add_argument("--storage", type=str, default="", help="e.g. sqlite:///study.db")

    p.add_argument("--train-files", type=str, nargs="*", default=["french.txt", "l.txt", "all_Flaubert.txt"])
    p.add_argument("--eval-fr", type=str, default="all_Flaubert.txt")
    p.add_argument("--eval-en", type=str, default="french.txt")
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

    # Seed the study with a notebook-like baseline (helps Optuna converge faster)
    # Note: batch_size_per_gpu=256 with world_size=2 => global batch 512 (matches notebook scale).
    study.enqueue_trial({
        "max_pair_dist": 150,
        "max_depth": 8,
        "n_paragraphs": 3200,
        "min_char_freq": 1,
        "lowercase": False,
        "fold_diacritics": False,
        "digit_map": "none",
        "collapse_whitespace": False,
        "bracket_augment_prob": 0.0,
        "synth_n_paragraphs": 0,
        "synth_max_span": 256,
        "synth_repeats": 2,
        "hidden_size": 512,
        "embedding_dim": 512,
        "n_layers": 2,
        "dropout": 0.5,
        "forget_bias": 1.0,
        "seq_length": 152,
        "stride_div": 1,
        "batch_size_per_gpu": 256,
        "lr": 0.002,
        "weight_decay": 0.0,
        "grad_clip": 5.0,
        "lambda_l1_h": 0.0,
        "probe_alpha": 1.0,
        "probe_subsample_every": 1,
    })

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args, world_size)
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
        "token_batch": best.user_attrs.get("token_batch", None),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "best_result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
