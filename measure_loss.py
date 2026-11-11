#!/usr/bin/env python3
"""
Truncated HOSVD (Tucker) of the Llama-2-7b layer tensor, plus reconstruction loss.

Tensor X has shape (layers, out, in). HOSVD:
    1. For each mode n, unfold X into X_(n) and take the top-r_n left singular
       vectors  ->  factor U_n   (computed from the Gram matrix X_(n) X_(n)^T,
       whose eigenvectors are the left singular vectors of X_(n)).
    2. Core G = X x_1 U_1^T x_2 U_2^T x_3 U_3^T
    3. Reconstruction X_hat = G x_1 U_1 x_2 U_2 x_3 U_3

Loss = difference between X and X_hat:
    - MAIN: squared relative error  ||X - X_hat||_F^2 / ||X||_F^2
    - Frobenius error        ||X - X_hat||_F
    - relative error         ||X - X_hat||_F / ||X||_F
    - MSE                    mean((X - X_hat)^2)
Also reports the identity-based error sqrt(||X||^2 - ||G||^2) (exact for
orthonormal factors) and the HOSVD a-priori bound from discarded singular values.

Usage:
    python hosvd_llama.py --matrix q_proj --ranks 16,512,512
    python hosvd_llama.py --matrix q_proj --sweep          # several rank settings
    python hosvd_llama.py --tensor q_proj_tensor.pt --ranks 32,1024,1024

Requirements: torch, safetensors, huggingface_hub (and llama_layer_tensor.py
in the same folder if you don't pass --tensor).
"""
import argparse
import json
import math

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = "meta-llama/Llama-2-7b-hf"
MATRICES = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


_ST_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def fetch_tensors_remote(model_id: str, shard: str, names, dtype):
    """Fetch only the named tensors from a remote .safetensors file using HTTP
    range requests (no multi-GB shard download, almost no disk used)."""
    import struct
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(model_id, shard)
    headers = build_hf_headers()  # picks up your HF token
    sess = requests.Session()

    def get_range(start, end):  # inclusive byte range
        r = sess.get(url, headers={**headers, "Range": f"bytes={start}-{end}"},
                     allow_redirects=True, timeout=300)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("Server ignored the Range request; cannot fetch partially.")
        return r.content

    header_len = struct.unpack("<Q", get_range(0, 7))[0]
    header = json.loads(get_range(8, 8 + header_len - 1))
    base = 8 + header_len

    out = {}
    for name in names:
        meta = header[name]
        begin, end = meta["data_offsets"]
        raw = bytearray(get_range(base + begin, base + end - 1))
        t = torch.frombuffer(raw, dtype=_ST_DTYPES[meta["dtype"]]).reshape(meta["shape"])
        out[name] = t.to(dtype)
        print(f"  fetched {name} {tuple(meta['shape'])}", flush=True)
    return out


def build_layer_tensor(matrix: str, model_id: str = MODEL_ID,
                       dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Stack one weight-matrix type across layers -> (n_layers, out, in)."""
    with open(hf_hub_download(model_id, "config.json")) as f:
        n_layers = json.load(f)["num_hidden_layers"]
    try:
        with open(hf_hub_download(model_id, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
    except Exception:  # single-file checkpoint
        weight_map = None

    names = [f"model.layers.{i}.{MATRICES[matrix]}.weight" for i in range(n_layers)]
    shards = {}
    for name in names:
        shard = weight_map[name] if weight_map else "model.safetensors"
        shards.setdefault(shard, []).append(name)

    loaded = {}
    for shard, shard_names in shards.items():
        loaded.update(fetch_tensors_remote(model_id, shard, shard_names, dtype))
    return torch.stack([loaded[n] for n in names], dim=0)


def build_multi_tensor(matrices, model_id: str = MODEL_ID,
                       dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Stack several matrix types along mode 0 -> (n_types * n_layers, out, in).

    Order along mode 0 is type-major: all layers of the first type, then all
    layers of the second, etc. All types must share the same (out, in) shape
    (e.g. q/k/v/o, or gate/up)."""
    unknown = [m for m in matrices if m not in MATRICES]
    if unknown:
        raise ValueError(f"Unknown matrix type(s) {unknown}; choose from {list(MATRICES)}")

    parts = []
    for m in matrices:
        print(f"Loading {m} ...")
        parts.append(build_layer_tensor(m, model_id, dtype))

    shapes = {tuple(p.shape[1:]) for p in parts}
    if len(shapes) > 1:
        raise ValueError(
            "Matrix types have different shapes and can't be stacked in one 3-mode tensor: "
            + ", ".join(f"{m}={tuple(p.shape[1:])}" for m, p in zip(matrices, parts)))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------- tensor ops
def unfold(T: torch.Tensor, mode: int) -> torch.Tensor:
    """Mode-n unfolding: (I_n, product of other dims)."""
    return torch.movedim(T, mode, 0).reshape(T.shape[mode], -1)


def mode_product(T: torch.Tensor, M: torch.Tensor, mode: int) -> torch.Tensor:
    """T x_mode M, with M of shape (new_dim, T.shape[mode])."""
    out = torch.tensordot(M, T, dims=([1], [mode]))  # new axis first
    return torch.movedim(out, 0, mode)


# ------------------------------------------------------------------- HOSVD
def hosvd(X: torch.Tensor, ranks):
    """Truncated HOSVD. Returns core, factors, and per-mode singular values."""
    factors, sing_vals = [], []
    for mode, r in enumerate(ranks):
        A = unfold(X, mode)
        gram = A @ A.T                                  # (I_n, I_n)
        evals, evecs = torch.linalg.eigh(gram.double())  # ascending
        evals = evals.flip(0).clamp_min(0)
        evecs = evecs.flip(1)
        U = evecs[:, :r].to(X.dtype)                    # (I_n, r)
        factors.append(U)
        sing_vals.append(evals.sqrt())                  # all singular values of X_(n)
        del A, gram

    core = X
    for mode, U in enumerate(factors):
        core = mode_product(core, U.T, mode)
    return core, factors, sing_vals


def reconstruct(core, factors):
    X_hat = core
    for mode, U in enumerate(factors):
        X_hat = mode_product(X_hat, U, mode)
    return X_hat


# -------------------------------------------------------------------- loss
def measure_loss(X, core, factors, sing_vals, ranks, direct=True):
    norm_X = torch.linalg.norm(X).item()
    norm_G = torch.linalg.norm(core).item()
    res = {"ranks": list(ranks), "norm_X": norm_X}

    # Exact for orthonormal factors: HOSVD reconstruction is an orthogonal projection
    res["fro_err_identity"] = math.sqrt(max(norm_X**2 - norm_G**2, 0.0))

    # A-priori HOSVD bound: ||X - X_hat||^2 <= sum_n sum_{i>r_n} sigma_i^(n)^2
    bound_sq = sum(float((s[r:] ** 2).sum()) for s, r in zip(sing_vals, ranks))
    res["fro_err_bound"] = math.sqrt(bound_sq)

    if direct:  # literal difference between actual tensor and reconstruction
        X_hat = reconstruct(core, factors)
        diff = X - X_hat
        res["fro_err"] = torch.linalg.norm(diff).item()
        res["mse"] = diff.pow(2).mean().item()
        res["max_abs_err"] = diff.abs().max().item()
        del X_hat, diff
    else:
        res["fro_err"] = res["fro_err_identity"]
        res["mse"] = res["fro_err"] ** 2 / X.numel()

    res["rel_err"] = res["fro_err"] / norm_X

    # Primary metric: squared Frobenius error over squared Frobenius norm of the weights
    res["rel_sq_err"] = res["fro_err"] ** 2 / norm_X**2
    res["rel_sq_err_identity"] = res["fro_err_identity"] ** 2 / norm_X**2
    res["rel_sq_err_bound"] = bound_sq / norm_X**2

    # Compression
    stored = core.numel() + sum(U.numel() for U in factors)
    res["compression_ratio"] = X.numel() / stored
    return res


def print_result(r):
    print(f"\nranks {r['ranks']}")
    print(f"  ||X - X_hat||^2/||X||^2 : {r['rel_sq_err']:.6e}   <-- main loss")
    print(f"    via identity          : {r['rel_sq_err_identity']:.6e}")
    print(f"    HOSVD bound           : {r['rel_sq_err_bound']:.6e}")
    print(f"  ||X||_F                 : {r['norm_X']:.4f}")
    print(f"  ||X - X_hat||_F         : {r['fro_err']:.4f}")
    print(f"  relative error          : {r['rel_err']:.4%}")
    print(f"  MSE                     : {r['mse']:.3e}")
    if "max_abs_err" in r:
        print(f"  max |X - X_hat|         : {r['max_abs_err']:.4e}")
    print(f"  err via ||X||^2-||G||^2 : {r['fro_err_identity']:.4f}  (should match)")
    print(f"  HOSVD upper bound       : {r['fro_err_bound']:.4f}")
    print(f"  compression ratio       : {r['compression_ratio']:.2f}x")


# -------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="q_proj",
                   help="One or more matrix types, comma-separated (same shape required), "
                        "e.g. q_proj or q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--tensor", default=None, help="Path to a saved .pt tensor (skips HF download)")
    p.add_argument("--ranks", default="16,512,512", help="Tucker ranks: layers,out,in")
    p.add_argument("--sweep", action="store_true", help="Run a few rank settings instead")
    p.add_argument("--no-direct", action="store_true", help="Skip explicit reconstruction (saves memory)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", default=None, help="Write results to this JSON file")
    args = p.parse_args()

    if args.tensor:
        X = torch.load(args.tensor)
    else:
        X = build_multi_tensor([m.strip() for m in args.matrix.split(",")])
    X = X.float().to(args.device)
    print(f"Tensor shape {tuple(X.shape)} on {args.device}")

    if args.sweep:
        L, O, I = X.shape
        settings = [(L // 4, O // 16, I // 16), (L // 2, O // 8, I // 8),
                    (L // 2, O // 4, I // 4), (L, O // 2, I // 2)]
    else:
        settings = [tuple(int(r) for r in args.ranks.split(","))]

    results = []
    for ranks in settings:
        core, factors, svals = hosvd(X, ranks)
        r = measure_loss(X, core, factors, svals, ranks, direct=not args.no_direct)
        print_result(r)
        results.append(r)
        del core, factors

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.save}")


if __name__ == "__main__":
    main()