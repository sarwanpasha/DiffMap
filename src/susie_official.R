# susie_official.R -- run the canonical susieR::susie_rss on one locus.
#
# Usage:  Rscript susie_official.R  z.txt  R.txt  n  out.txt  [L]
#   z.txt   : one z-score per line (p lines)
#   R.txt   : p x p LD matrix, whitespace separated, one row per line
#   n       : GWAS sample size
#   out.txt : output, tab separated with columns  pip  cs
#             (cs = credible-set id, -1 if the variant is in no 95% credible set)
#   L       : maximum number of single effects (default 10)
#
# Called automatically by baselines.py::susieR_run. Tested with susieR 0.14.2.
# If susieR is installed in a personal library, set R_LIBS_USER before calling.
suppressMessages(library(susieR))

args <- commandArgs(trailingOnly = TRUE)
z  <- scan(args[1], quiet = TRUE)
p  <- length(z)
R  <- matrix(scan(args[2], quiet = TRUE), nrow = p, byrow = TRUE)
n  <- as.numeric(args[3])
L  <- if (length(args) >= 5) as.integer(args[5]) else 10L

# Default settings; external-reference LD occasionally makes residual-variance
# estimation fail, in which case we fall back to susieR's plain default call.
fit <- tryCatch(
  susie_rss(z = z, R = R, n = n, L = L, estimate_residual_variance = FALSE),
  error = function(e) susie_rss(z = z, R = R, n = n, L = L))

pip   <- susie_get_pip(fit)
cs    <- susie_get_cs(fit, Xcorr = R, coverage = 0.95, min_abs_corr = 0.5)$cs
cs_id <- rep(-1L, p)
if (length(cs) > 0) for (k in seq_along(cs)) cs_id[cs[[k]]] <- k

write.table(data.frame(pip = pip, cs = cs_id), args[4],
            row.names = FALSE, quote = FALSE, sep = "\t")
