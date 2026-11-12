"""
sim.py -- simulation of GWAS summary statistics from the RSS likelihood.

Eq. (1) (README):   z | beta, R  ~  N( sqrt(n) * R @ beta ,  R )

where R is the *true* (in-sample) LD matrix of the locus. To simulate we draw a
sparse causal configuration gamma, give each causal variant a non-zero
standardized effect, form the mean sqrt(n) R beta, and add correlated noise
R^{1/2} @ eps with eps ~ N(0, I).

We parameterise the effect of each causal variant by its non-centrality
parameter (NCP) sqrt(n) * beta_j, i.e. the expected z-score the variant would
have if it were in perfect isolation. Sampling the NCP directly (rather than
beta) keeps the simulated loci in a realistic power range for any n.
"""
import numpy as np


def psd_sqrt(R, eps=1e-6):
    """Symmetric square root of an LD matrix, made PSD by clipping eigenvalues.

    Empirical LD matrices are often slightly non-PSD (finite samples, numerical
    error). We clip negative eigenvalues to `eps` so that R^{1/2} exists and the
    simulated noise R^{1/2} eps has covariance (approximately) R.
    """
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R.astype(np.float64))
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T


def simulate_locus(R, maf, Rh, rng, k_max=3, n_range=(2e4, 5e5), ncp_range=(4.0, 12.0)):
    """Simulate one fine-mapping locus.

    Args
      R        : (p, p) true in-sample LD used to GENERATE the z-scores
      maf      : (p,) minor-allele frequencies (returned unchanged; kept for API symmetry)
      Rh       : (p, p) psd_sqrt(R), precomputed once per LD block for speed
      rng      : numpy Generator
      k_max    : number of causal variants is drawn uniformly from {1, ..., k_max}
      n_range  : GWAS sample size drawn log-uniformly from this interval
      ncp_range: |NCP| of each causal variant drawn uniformly from this interval,
                 with a random sign
    Returns
      z (p,) float32, gamma (p,) int8 causal indicators, n (float)
    """
    p = R.shape[0]
    k = int(rng.integers(1, k_max + 1))
    causal = rng.choice(p, size=k, replace=False)
    gamma = np.zeros(p, dtype=np.int8)
    gamma[causal] = 1

    n = float(np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1]))))
    ncp = np.zeros(p)
    ncp[causal] = rng.uniform(*ncp_range, size=k) * rng.choice([-1.0, 1.0], size=k)

    # mean = sqrt(n) R beta = R @ ncp ;  noise ~ N(0, R)
    mean = R.astype(np.float64) @ ncp
    z = mean + Rh @ rng.standard_normal(p)
    return z.astype(np.float32), gamma, n
