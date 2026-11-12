"""
inference.py -- using a trained DiffMap network.

  * oneshot_pip(...)  : marginal PIPs from ONE forward pass with every coordinate
                        masked (README Eq. 3). This is the amortized posterior.
  * cond_pip(...)     : PIPs conditional on a set of variants being causal
                        (those coordinates are given the symbol 1 instead of MASK).
  * diffmap_cs(...)   : credible sets by iterative conditioning (README Algorithm 1).

All functions work on CPU or GPU. bfloat16 autocast is used automatically on CUDA.
"""
import contextlib
import numpy as np
import torch

from model import MASK


def _autocast(device):
    """bfloat16 autocast on GPUs that support it (Ampere or newer); full precision otherwise."""
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_tensors(z, maf, R, device):
    """NumPy arrays -> float32 tensors on `device` (z: (p,), maf: (p,), R: (p, p))."""
    zt = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device)
    mt = torch.as_tensor(np.asarray(maf, dtype=np.float32), device=device)
    Rt = torch.as_tensor(np.asarray(R, dtype=np.float32), device=device)
    return zt, mt, Rt


@torch.no_grad()
def cond_pip(model, zt, mt, Rt, fixed, device, n=1e5):
    """Conditional PIPs Pr(gamma_j = 1 | gamma_F = 1, z, R) for the fixed set F.

    zt, mt: (p,) tensors; Rt: (p, p) tensor; fixed: list of variant indices set to causal.
    With fixed = [] this is exactly the one-shot marginal PIP of README Eq. (3).
    """
    p = zt.shape[0]
    g = torch.full((1, p), MASK, device=device, dtype=torch.long)
    if fixed:
        g[0, torch.as_tensor(fixed, device=device, dtype=torch.long)] = 1
    t = torch.ones(1, device=device)                      # t = 1: fully masked input
    nn_ = torch.full((1,), float(n), device=device)
    with _autocast(device):
        logits = model(zt[None], g, mt[None], Rt, t, nn_)
    return torch.sigmoid(logits.float())[0].cpu().numpy()


def oneshot_pip(model, z, maf, R, n=1e5, device="cpu"):
    """Convenience wrapper: NumPy in, NumPy PIPs out (one forward pass)."""
    zt, mt, Rt = to_tensors(z, maf, R, device)
    return cond_pip(model, zt, mt, Rt, [], device, n=n)


def diffmap_cs(model, zt, mt, Rt, Rnp, device, n=1e5, max_signals=10,
               lead_thr=0.1, ld_thr=0.4, level=0.95, max_size=200):
    """Credible sets by iterative conditioning (README Algorithm 1).

    Repeat:
      1. conditional PIPs given the leads already fixed as causal;
      2. lead = highest-PIP variant not yet used; stop if its PIP < lead_thr;
      3. candidate set S = variants in LD with the lead (|R_lead,j| >= ld_thr), plus the lead;
      4. sort S by PIP and keep the shortest prefix whose normalised PIP mass reaches `level`;
      5. record that prefix as one credible set and fix the lead as causal.
    The default balanced operating point is lead_thr = 0.1, ld_thr = 0.4.
    Returns a list of credible sets (lists of variant indices).
    """
    fixed, CSs, used = [], [], set()
    p = Rnp.shape[0]
    for _ in range(max_signals):
        pip = cond_pip(model, zt, mt, Rt, fixed, device, n=n)
        cand = [j for j in range(p) if j not in used and j not in fixed]
        if not cand:
            break
        lead = max(cand, key=lambda j: pip[j])
        if pip[lead] < lead_thr:
            break
        cluster = [j for j in cand if abs(Rnp[lead, j]) >= ld_thr]
        if lead not in cluster:
            cluster.append(lead)
        cp = np.array([pip[j] for j in cluster])
        if cp.sum() <= 0:
            CSs.append([lead]); used.add(lead); fixed.append(lead)
            continue
        order = np.argsort(-cp)
        csum = np.cumsum(cp[order]) / cp.sum()
        k = min(int(np.searchsorted(csum, level) + 1), len(cluster), max_size)
        cs = [cluster[order[i]] for i in range(k)]
        CSs.append(cs)
        used.update(cs)
        fixed.append(lead)
    return CSs
