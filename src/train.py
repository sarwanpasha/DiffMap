#!/usr/bin/env python
"""
train.py -- train DiffMap with diverse-reference augmentation.

Every training example is generated on the fly:
  1. pick a random LD block from the training bank (built by build_ld_refs.py);
  2. simulate a causal configuration gamma and z-scores from the block's TRUE
     in-sample LD `R` (README Eq. 1, see sim.py);
  3. choose the LD matrix SHOWN to the network:
        with probability --matched-prob (default 0.2) the matched matrix R,
        otherwise a reference drawn uniformly from --train-refs
        (default: eur200, eur100, eur50, afr200, eas200; amr200 and sas200 are
        never shown during training and are used only to test generalization);
  4. draw a diffusion time t ~ U(t_min, 1] and mask each coordinate of gamma
     independently with probability t (absorbing-state forward process);
  5. loss = re-weighted binary cross-entropy over the masked coordinates,
        L = E_t [ (1/t) * sum_{j masked} BCE(f_theta(...)_j, gamma_j) ] / p
     which is the absorbing-diffusion variational bound of README Eq. (2).

Because the z-scores are generated with R but the network sees a different,
misspecified matrix most of the time, calibration under reference mismatch is
LEARNED rather than assumed -- this is DiffMap's robustness mechanism.

Recommended settings
  small : --d 128 --heads 4 --layers 6 --steps 30000
  big   : --d 256 --heads 8 --layers 8 --steps 60000     (all main results)
  both  : --batch-size 8 --lr 3e-4 --matched-prob 0.2

Example
  python train.py --bank DATA/ld_refs_train --out runs/diffmap_big \
                  --d 256 --heads 8 --layers 8 --steps 60000 --workers 4

Output: <out>/diffmap_final.pt  = {"model": state_dict, "args": vars(args), "step": int}
"""
import argparse
import glob
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from model import DiffMap, MASK, count_parameters
from sim import psd_sqrt, simulate_locus


class LocusStream(IterableDataset):
    """Infinite stream of simulated training loci (one fresh simulation per example)."""

    def __init__(self, files, train_refs, matched_prob, k_max, n_range, ncp_range, seed, cache_size=8):
        self.files, self.train_refs = files, train_refs
        self.matched_prob, self.k_max = matched_prob, k_max
        self.n_range, self.ncp_range = n_range, ncp_range
        self.seed, self.cache_size = seed, cache_size

    def _load(self, cache, path):
        """Load a block (R, its square root, MAF and the reference matrices) with an LRU cache.

        psd_sqrt(R) needs an eigendecomposition, so it is computed once per cached block.
        """
        if path in cache:
            cache.move_to_end(path)
            return cache[path]
        d = np.load(path, allow_pickle=True)
        R = d["R"].astype(np.float64)
        entry = dict(R=R, Rh=psd_sqrt(R), maf=d["maf"].astype(np.float32),
                     refs={r: d["ref_" + r].astype(np.float32) for r in self.train_refs})
        cache[path] = entry
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return entry

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed + 1000 * wid)
        cache = OrderedDict()
        while True:
            b = self._load(cache, self.files[rng.integers(len(self.files))])
            z, gamma, n = simulate_locus(b["R"], b["maf"], b["Rh"], rng, k_max=self.k_max,
                                         n_range=self.n_range, ncp_range=self.ncp_range)
            if rng.random() < self.matched_prob:
                Rshow = b["R"].astype(np.float32)                    # matched (in-sample) LD
            else:
                Rshow = b["refs"][self.train_refs[rng.integers(len(self.train_refs))]]  # misspecified
            yield (torch.from_numpy(z), torch.from_numpy(gamma.astype(np.int64)),
                   torch.from_numpy(b["maf"]), torch.from_numpy(Rshow), torch.tensor(n, dtype=torch.float32))


def usable_blocks(bank, train_refs):
    """Blocks that contain every requested reference and share the modal number of SNPs
    (so they can be stacked into a batch; short trailing blocks are skipped)."""
    files = sorted(glob.glob(os.path.join(bank, "*.npz")))
    ps, keep = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            if all(("ref_" + r) in d.files for r in train_refs):
                keep.append(f); ps.append(d["maf"].shape[0])
    if not keep:
        raise SystemExit(f"no usable blocks in {bank} (check --train-refs)")
    p_mode = max(set(ps), key=ps.count)
    return [f for f, p in zip(keep, ps) if p == p_mode], p_mode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, help="training LD bank (build_ld_refs.py output)")
    ap.add_argument("--out", required=True, help="output directory for checkpoints")
    ap.add_argument("--train-refs", default="eur200,eur100,eur50,afr200,eas200")
    ap.add_argument("--matched-prob", type=float, default=0.2)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--t-min", type=float, default=0.02, help="smallest diffusion time (mask rate)")
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-min", type=float, default=2e4)
    ap.add_argument("--n-max", type=float, default=5e5)
    ap.add_argument("--ncp-min", type=float, default=4.0)
    ap.add_argument("--ncp-max", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-size", type=int, default=8, help="LD blocks cached per data worker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    on_gpu = dev.startswith("cuda")
    use_amp = on_gpu and torch.cuda.is_bf16_supported()      # bf16 autocast on Ampere or newer
    print(f"device: {torch.cuda.get_device_name(torch.device(dev)) if on_gpu else 'cpu'}"
          f" | bf16 autocast: {use_amp}", flush=True)

    train_refs = [r.strip() for r in args.train_refs.split(",") if r.strip()]
    files, p = usable_blocks(args.bank, train_refs)
    print(f"train refs: {train_refs} | matched_prob {args.matched_prob} | {len(files)} blocks of p={p}", flush=True)

    ds = LocusStream(files, train_refs, args.matched_prob, args.k_max, (args.n_min, args.n_max),
                     (args.ncp_min, args.ncp_max), args.seed, args.cache_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                    pin_memory=on_gpu, persistent_workers=args.workers > 0)

    model = DiffMap(args.d, args.heads, args.layers).to(dev)
    print(f"params: {count_parameters(model) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # linear warm-up followed by cosine decay to 10% of the peak learning rate
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, args.steps))))))

    def save(step, name):
        torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
                   os.path.join(args.out, name))

    model.train()
    step, skips, t0 = 0, 0, time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for z, gamma, maf, Rshow, n in dl:
        z, gamma, maf, Rshow, n = (x.to(dev, non_blocking=True) for x in (z, gamma, maf, Rshow, n))
        B, P = gamma.shape

        # ---- absorbing-state forward corruption: mask each coordinate w.p. t ----
        t = args.t_min + (1 - args.t_min) * torch.rand(B, generator=gen).to(dev)       # (B,)
        mask = torch.rand(B, P, generator=gen).to(dev) < t[:, None]
        mask[torch.arange(B, device=dev), torch.randint(0, P, (B,), generator=gen).to(dev)] = True  # >=1 masked
        g = torch.where(mask, torch.full_like(gamma, MASK), gamma)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(z, g, maf, Rshow, t, n)
        bce = F.binary_cross_entropy_with_logits(logits.float(), gamma.float(), reduction="none")
        # README Eq. (2): BCE over masked coordinates, re-weighted by 1/t, normalised per variant
        loss = ((bce * mask).sum(1) / (t * P)).mean()

        if not torch.isfinite(loss):
            skips += 1
            opt.zero_grad(set_to_none=True)
            if skips > 1000:
                raise SystemExit("too many non-finite losses; check the input data")
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        step += 1

        if step % args.log_every == 0:
            with torch.no_grad():
                pr = torch.sigmoid(logits.float())
                c = mask & (gamma == 1); nc = mask & (gamma == 0)
                pc = pr[c].mean().item() if c.any() else float("nan")
                pn = pr[nc].mean().item() if nc.any() else float("nan")
            print(f"step {step:6d} loss {loss.item():.4f} | P(causal)~{pc:.3f} P(noncausal)~{pn:.3f} | "
                  f"{step / (time.time() - t0):.1f} it/s (skips {skips})", flush=True)
        if args.save_every and step % args.save_every == 0:
            save(step, f"diffmap_step{step}.pt")
        if step >= args.steps:
            break

    save(step, "diffmap_final.pt")
    print(f"done. skips {skips} -> {os.path.join(args.out, 'diffmap_final.pt')}", flush=True)


if __name__ == "__main__":
    main()
