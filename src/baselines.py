"""
baselines.py -- adapters that run the competing methods on one locus.

  * susieR_run(z, R, n)          : canonical SuSiE-RSS from the CRAN package susieR
                                   (via the R script susie_official.R in this folder)
  * rsparsepro_run(z, R, src)    : RSparsePro (Zhang et al., 2024), via its own CLI
  * numpy SuSiE                  : see susie_rss.py (no adapter needed)

Every adapter takes the SAME z-scores and the SAME supplied LD matrix as DiffMap,
writes them to a temporary directory, calls the external tool, and parses its
output back into (pip, credible_sets). Temporary files are always cleaned up.

How the external tools are found
--------------------------------
susieR : the R command is read from the environment variable DIFFMAP_RSCRIPT
         (default "Rscript"). If R/susieR live in a container, set e.g.
             export DIFFMAP_RSCRIPT="apptainer exec --bind /path/to/data r.sif Rscript"
         and, if susieR is in a personal library, export R_LIBS_USER=/path/to/Rlibs .
RSparsePro : pass the path of a local clone of https://github.com/zhwm/RSparsePro_LD
         (the script src/rsparsepro_ld.py inside it is called with default settings).
"""
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUSIE_R_SCRIPT = os.path.join(HERE, "susie_official.R")


def _groups_to_sets(cs_col):
    """Convert a per-variant credible-set label column into a list of index lists."""
    cs_col = np.asarray(cs_col)
    labels = [g for g in sorted(set(cs_col.tolist())) if g > 0]
    return [[int(i) for i in np.where(cs_col == g)[0]] for g in labels]


def susieR_available():
    """True if the configured Rscript command can load susieR."""
    cmd = shlex.split(os.environ.get("DIFFMAP_RSCRIPT", "Rscript")) + \
        ["-e", "suppressMessages(library(susieR)); cat('ok')"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode == 0 and "ok" in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def susieR_run(z, R, n, L=10):
    """Canonical susieR::susie_rss on one locus. Returns (pip, credible_sets)."""
    td = tempfile.mkdtemp(prefix="diffmap_susieR_")
    try:
        zf, rf, of = (os.path.join(td, x) for x in ("z.txt", "R.txt", "out.txt"))
        np.savetxt(zf, np.asarray(z, dtype=float))
        np.savetxt(rf, np.asarray(R, dtype=float))
        cmd = shlex.split(os.environ.get("DIFFMAP_RSCRIPT", "Rscript")) + \
            [SUSIE_R_SCRIPT, zf, rf, str(float(n)), of, str(int(L))]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        import pandas as pd
        df = pd.read_csv(of, sep="\t")
        return df["pip"].to_numpy(dtype=float), _groups_to_sets(df["cs"].to_numpy(dtype=int))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def rsparsepro_run(z, R, src, K=0):
    """RSparsePro on one locus (default settings). Returns (pip, credible_sets).

    RSparsePro has no sample-size argument; K = 0 keeps its default number of effects.
    """
    td = tempfile.mkdtemp(prefix="diffmap_rsp_")
    try:
        p = len(z)
        zf, lf, save = (os.path.join(td, x) for x in ("in.z", "in.ld", "out"))
        with open(zf, "w") as f:
            f.write("RSID\tZ\n")
            for i in range(p):
                f.write(f"v{i}\t{float(z[i]):.8g}\n")
        np.savetxt(lf, np.asarray(R, dtype=float), fmt="%.8g", delimiter=" ")
        cmd = [sys.executable, os.path.join(src, "src", "rsparsepro_ld.py"),
               "--z", zf, "--ld", lf, "--save", save]
        if K > 0:
            cmd += ["--K", str(K)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        import pandas as pd
        df = pd.read_csv(save + ".rsparsepro.txt", sep=r"\s+", engine="python")
        pip = df["PIP"].to_numpy(dtype=float) if "PIP" in df.columns else np.zeros(p)
        if len(pip) != p:
            pip = np.zeros(p)
        return pip, _groups_to_sets(df["cs"].fillna(0).to_numpy())
    finally:
        shutil.rmtree(td, ignore_errors=True)


def run_locus(method, z, R, n, rsparsepro_src=None, K=0):
    """Uniform entry point used by the comparison scripts. Returns PIPs only."""
    if method == "rsparsepro":
        return rsparsepro_run(z, R, rsparsepro_src, K)[0]
    if method == "susieR":
        return susieR_run(z, R, n)[0]
    raise ValueError(f"unknown method {method!r}")
