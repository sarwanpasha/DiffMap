"""
model.py -- the DiffMap denoiser network f_theta.

DiffMap is an *amortized* posterior for statistical fine-mapping. A single
network is trained once on simulated loci and then applied to any new locus in
one forward pass. The network is the denoiser of an absorbing-state ("masking")
discrete diffusion over the binary causal-indicator vector gamma in {0,1}^p
(Austin et al., NeurIPS 2021):

  * Each variant j carries a symbol g_j in {0, 1, MASK}. During training a
    random fraction t of the coordinates of the true gamma is replaced by MASK
    and the network predicts Pr(gamma_j = 1 | unmasked coordinates, z, R, ...)
    for every masked coordinate.
  * At test time every coordinate is MASK (t = 1). One forward pass then gives
    the marginal posterior inclusion probabilities (PIPs) -- README Eq. (3).
  * Setting selected coordinates to 1 instead of MASK conditions the posterior
    on those variants being causal; this is how credible sets are extracted by
    iterative conditioning (README Algorithm 1, see inference.py).

Architecture:
  * one token per variant; token features = z-score, |z|, MAF, current symbol
  * a global embedding of the diffusion time t and of log10(sample size n)
  * a pre-norm transformer encoder over the p variant tokens
  * the supplied LD matrix R enters ONLY as an additive attention bias: for
    every layer and head the pre-softmax logit between variants i and j is
    augmented by a learned *monotone* function of R_ij. The supplied LD can
    therefore shape how information flows between variants, but it is never
    treated as ground truth (which is what makes robustness learnable).
  * there are no positional embeddings: the model is equivariant to the order
    of the variants, and a locus may have any number of variants p.

Recommended model sizes:
  small: d=128, heads=4, layers=6      big: d=256, heads=8, layers=8 (main results)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Symbol used for a masked ("unknown") coordinate of gamma. 0 = non-causal, 1 = causal.
MASK = 2


class MonotoneLDBias(nn.Module):
    """Learned monotone (non-decreasing) map  R_ij in [-1, 1]  ->  attention bias, per head.

    bias_h(r) = b_h + sum_k softplus(w_hk) * ramp_k(r)

    where ramp_k is a piecewise-linear step that rises from 0 to 1 on the k-th of
    K equal-width bins covering [-1, 1]. Because every coefficient softplus(.)
    is positive and every ramp is non-decreasing, bias_h is non-decreasing in r
    for every head, i.e. a monotone function of R_ij.
    The ramps depend only on R, so they are computed once per forward pass
    (ld_ramps) and shared by all layers; each layer has its own coefficients.
    """

    def __init__(self, heads, n_bins=8):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(heads, n_bins))   # slopes (through softplus)
        self.b = nn.Parameter(torch.zeros(heads))            # per-head offset

    def forward(self, ramps):
        # ramps: (B, p, p, K)  ->  bias: (B, heads, p, p)
        coef = F.softplus(self.w)                                       # (H, K) >= 0
        bias = torch.einsum("bijk,hk->bhij", ramps, coef)
        return (bias + self.b.view(1, -1, 1, 1)).contiguous()


def ld_ramps(R, n_bins=8):
    """Piecewise-linear ramp features of R used by MonotoneLDBias: (B,p,p) -> (B,p,p,K)."""
    edges = torch.linspace(-1.0, 1.0, n_bins + 1, device=R.device, dtype=R.dtype)
    lo, width = edges[:-1], edges[1:] - edges[:-1]
    return ((R.clamp(-1.0, 1.0).unsqueeze(-1) - lo) / width).clamp(0.0, 1.0)


class LDBiasedBlock(nn.Module):
    """Pre-norm transformer block whose self-attention logits carry an LD bias."""

    def __init__(self, d, heads, ff_mult=4, dropout=0.0):
        super().__init__()
        assert d % heads == 0, "model width d must be divisible by the number of heads"
        self.h, self.dh = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ld_bias = MonotoneLDBias(heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ramps):
        # x: (B, p, d) token states; ramps: (B, p, p, K) LD ramp features (see ld_ramps)
        B, p, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, p, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        # q, k, v: (B, H, p, dh). Explicit attention: logits = q k^T / sqrt(dh) + LD bias.
        # (Written out rather than via scaled_dot_product_attention so that gradients reach
        #  the LD-bias parameters identically on every PyTorch version and device.)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        logits = logits.float() + self.ld_bias(ramps).float()          # softmax in float32
        attn = torch.softmax(logits, dim=-1).to(v.dtype)
        a = (attn @ v).transpose(1, 2).reshape(B, p, d)                 # (B, p, d)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class DiffMap(nn.Module):
    """The DiffMap denoiser f_theta(gamma^(t), z, MAF, R, t, n) -> per-variant logits."""

    def __init__(self, d=256, heads=8, layers=8, dropout=0.0):
        super().__init__()
        self.d = d
        # Per-variant continuous features: z, |z|, MAF  (3 numbers per token)
        self.feat = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        # Embedding of the current symbol of gamma_j in {0, 1, MASK}
        self.sym = nn.Embedding(3, d)
        # Global context: diffusion time t in (0, 1] and log10(n) (sample size embedded globally)
        self.glob = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([LDBiasedBlock(d, heads, dropout=dropout) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, z, g, maf, R, t, n=None):
        """
        Args
          z   : (B, p) marginal z-scores
          g   : (B, p) long tensor of symbols in {0, 1, MASK}
          maf : (B, p) minor-allele frequencies in [0, 0.5]
          R   : (B, p, p) or (p, p) supplied LD (correlation) matrix
          t   : (B,) diffusion time = fraction of masked coordinates; use 1.0 at inference
          n   : (B,) GWAS sample size (optional; 1e5 is assumed when omitted)
        Returns
          logits: (B, p).  sigmoid(logits) = Pr(gamma_j = 1 | ...) for masked coordinates.
        """
        z = z.float(); maf = maf.float()
        if R.dim() == 2:                                  # allow a single unbatched LD matrix
            R = R.unsqueeze(0).expand(z.shape[0], -1, -1)
        R = R.float()
        B = z.shape[0]
        if n is None:
            n = torch.full((B,), 1e5, device=z.device)
        t = t.float().view(B, 1) if t.numel() == B else t.float().view(1, 1).expand(B, 1)
        n = n.float().view(B, 1) if n.numel() == B else n.float().view(1, 1).expand(B, 1)
        logn = (torch.log10(n.clamp_min(1.0)) - 5.0)      # centred around n = 1e5

        x = self.feat(torch.stack([z, z.abs(), maf], dim=-1))       # (B, p, d)
        x = x + self.sym(g)
        x = x + self.glob(torch.cat([t, logn], dim=-1)).unsqueeze(1)
        ramps = ld_ramps(R)                                           # computed once, shared by layers
        for blk in self.blocks:
            x = blk(x, ramps)
        return self.head(self.ln_out(x)).squeeze(-1)                 # (B, p)


def count_parameters(model):
    """Number of trainable parameters (1.22M for the small and 6.45M for the big configuration)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint written by train.py and return the model in eval mode.

    A checkpoint is a dict {"model": state_dict, "args": dict of training arguments}
    and the architecture (d, heads, layers) is read from "args".
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    model = DiffMap(a["d"], a["heads"], a["layers"]).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, a
