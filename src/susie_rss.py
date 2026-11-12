"""
susie_rss.py -- a self-contained NumPy implementation of SuSiE-RSS.

An independent NumPy reimplementation of SuSiE fitted to summary statistics
(Wang et al., JRSS-B 2020; Zou et al., PLoS Genet 2022). It is used for fast
development runs and for the latency benchmark. For final comparisons use the
canonical R package susieR (src/susie_official.R, flag --with-susieR); the two
agree closely.

Model:  the SuSiE "sum of single effects" prior with L single effects, each
placing one non-zero standardized effect on exactly one variant, fitted by
Iterative Bayesian Stepwise Selection (IBSS) on the sufficient statistics
implied by (z, R, n):

    X'X = (n-1) R ,   X'y = sqrt(n-1) * z_tilde ,   y'y = n-1 ,

with residual variance fixed to 1 (standardized phenotype). z_tilde applies the
small-sample adjustment of Zou et al. (2022):  z_tilde = z * sqrt((n-1)/(z^2+n-2)).
Each single effect's prior variance V_l is re-estimated every iteration by
maximising its marginal likelihood (set to 0 when that is not supported).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def _lbf(betahat, shat2, V):
    """Log Bayes factor of N(0, V) effect vs no effect, for each variant."""
    if V <= 0:
        return np.zeros_like(betahat)
    return 0.5 * np.log(shat2 / (V + shat2)) + 0.5 * betahat ** 2 / shat2 * (V / (V + shat2))


def _estimate_V(betahat, shat2, log_prior, V_init):
    """Maximise the single-effect marginal likelihood over log V (bounded search)."""
    def neg(logV):
        return -logsumexp(_lbf(betahat, shat2, np.exp(logV)) + log_prior)
    res = minimize_scalar(neg, bounds=(-30.0, 15.0), method="bounded")
    V = float(np.exp(res.x))
    # SuSiE convention: if the best V gives no evidence over V = 0, switch the effect off.
    if -res.fun <= logsumexp(log_prior) + 1e-8:
        return 0.0
    return V


def susie_rss(z, R, n, L=10, lam=0.0, max_iter=100, tol=1e-4, prior_weights=None):
    """Fit SuSiE-RSS.

    Args
      z    : (p,) marginal z-scores
      R    : (p, p) LD matrix (the reference panel supplied to the method)
      n    : GWAS sample size
      L    : maximum number of single effects (default L = 10)
      lam  : optional ridge regularisation R <- (1-lam) R + lam I
    Returns
      pip   : (p,) posterior inclusion probabilities
      alpha : (L, p) per-effect posterior over which variant carries the effect
              (rows of switched-off effects, V_l = 0, are set to 0)
    """
    z = np.asarray(z, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    p = z.shape[0]
    if lam > 0:
        R = (1.0 - lam) * R + lam * np.eye(p)
    n = float(n)
    ztil = z * np.sqrt((n - 1.0) / (z ** 2 + n - 2.0))
    XtX = (n - 1.0) * R
    Xty = np.sqrt(n - 1.0) * ztil
    d = np.diag(XtX).copy()
    d[d <= 0] = 1e-8
    sigma2 = 1.0
    prior = np.full(p, 1.0 / p) if prior_weights is None else np.asarray(prior_weights) / np.sum(prior_weights)
    log_prior = np.log(prior)

    alpha = np.full((L, p), 1.0 / p)
    mu = np.zeros((L, p))
    V = np.full(L, 0.2)              # initial prior variance of each standardized effect
    B = alpha * mu                   # (L, p) posterior mean effect of each single effect
    XtXb = XtX @ B.sum(0)            # running X'X b_total

    for _ in range(max_iter):
        alpha_old = alpha.copy()
        for l in range(L):
            XtXb -= XtX @ B[l]                       # remove effect l
            r = Xty - XtXb                           # residualised X'y
            betahat = r / d
            shat2 = sigma2 / d
            V[l] = _estimate_V(betahat, shat2, log_prior, V[l])
            lbf = _lbf(betahat, shat2, V[l])
            w = lbf + log_prior
            alpha[l] = np.exp(w - logsumexp(w))
            if V[l] > 0:
                post_var = 1.0 / (1.0 / V[l] + d / sigma2)
                mu[l] = post_var * r / sigma2
            else:
                mu[l] = 0.0
            B[l] = alpha[l] * mu[l]
            XtXb += XtX @ B[l]                       # add the updated effect back
        if np.max(np.abs(alpha - alpha_old)) < tol:
            break

    alpha_out = alpha.copy()
    alpha_out[V <= 0] = 0.0
    pip = 1.0 - np.prod(1.0 - alpha_out, axis=0)
    return pip, alpha_out


def credible_sets(alpha, R, level=0.95, purity=0.5):
    """SuSiE credible sets from the per-effect posteriors alpha.

    For every active single effect, take the smallest set of variants whose
    alpha mass reaches `level`; keep it only if its purity (minimum absolute LD
    between any two members) is at least `purity` (susieR default 0.5).
    Duplicate sets produced by different effects are reported once.
    Returns a list of lists of variant indices.
    """
    R = np.asarray(R)
    out, seen = [], set()
    for a in np.asarray(alpha):
        if a.sum() <= 0:
            continue
        order = np.argsort(-a)
        k = int(np.searchsorted(np.cumsum(a[order]), level) + 1)
        cs = sorted(int(i) for i in order[:k])
        key = tuple(cs)
        if key in seen:
            continue
        sub = np.abs(R[np.ix_(cs, cs)])
        if len(cs) > 1 and sub.min() < purity:
            continue
        seen.add(key)
        out.append(cs)
    return out
