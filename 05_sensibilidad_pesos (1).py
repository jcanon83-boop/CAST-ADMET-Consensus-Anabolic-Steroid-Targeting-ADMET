#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_sensibilidad_pesos.py — Sensitivity of the consensus framework to platform
weights (manuscript Section 2.3).

Samples 1,000 random weight vectors (Dirichlet(1,1,1,1)) over the four
target-fishing platforms (SwissTargetPrediction, FGP, PLATO, PHARMAPPER),
recomputes the consensus matrix, the Detectability (Det) and Endocrine
Impact (EI) indices and the four-class risk scheme for each replicate, and
quantifies:

  a) ranking stability      — Spearman rho vs the baseline ranking
  b) stealth-steroid stability — % of replicates in which each baseline
     Class III compound remains in Class III

Reference scenarios: equal weights (0.25 each) and single-platform-only
(one-hot) weightings.

Input : consensus_long.csv (same folder)
Output: FigS1_sensibilidad_pesos.png
        sensibilidad_pesos_claseIII.csv

Requires: Python >= 3.11, pandas, numpy, scipy, matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent

# --- platform weights used in the manuscript (Section 2.3) ---------------- #
W = {"fgp": 0.35, "plato": 0.30, "swiss": 0.20, "pharmapper": 0.15}
ORDER = ["swiss", "fgp", "plato", "pharmapper"]

# --- target panel for the Det / EI indices (Sections 2.4-2.5) ------------- #
UNIPROT = {
    "CYP3A4": "P08684", "CYP2C9": "P11712", "CYP2C19": "P33261",
    "CYP2D6": "P10635", "CYP17A1": "P05093", "CYP19A1": "P11511",
    "CYP11B1": "P15538", "CYP11B2": "P19099", "CYP51A1": "Q16850",
    "SRD5A1": "P18405", "SRD5A2": "P31213", "UGT2B7": "P16662",
    "HSD17B1": "P14061", "HSD17B2": "P37059", "HSD17B3": "P37058",
    "AR": "P10275", "SHBG": "P04278",
}
CYP_PANEL = ["CYP3A4", "CYP2C9", "CYP2C19", "CYP2D6", "CYP17A1",
             "CYP19A1", "CYP11B1", "CYP11B2", "CYP51A1"]          # 9 isoforms


def risk_class(det, ei):
    """Four-class scheme (Section 2.7)."""
    if ei >= 0.30:
        return "I" if det >= 0.35 else ("II" if det >= 0.15 else "III")
    return "IV" if det < 0.15 else None


def main(n_iter=1000, seed=42):
    cl = pd.read_csv(HERE / "consensus_long.csv")

    # per-platform score matrices (mol x uniprot), NaN = not reported
    mols = sorted(cl["mol"].unique())
    targets = sorted(cl["uniprot"].unique())
    mi = {m: i for i, m in enumerate(mols)}
    ti = {t: i for i, t in enumerate(targets)}
    M = {k: np.full((len(mols), len(targets)), np.nan) for k in W}
    for r in cl.itertuples():
        for k in W:
            v = getattr(r, k)
            if pd.notna(v):
                M[k][mi[r.mol], ti[r.uniprot]] = v

    tidx = {g: ti[UNIPROT[g]] for g in UNIPROT if UNIPROT[g] in ti}

    def det_ei(C):
        get = lambda g: C[:, tidx[g]] if g in tidx else np.zeros(C.shape[0])
        cyp = np.nanmean(np.column_stack(
            [np.nan_to_num(get(g)) for g in CYP_PANEL]), axis=1)
        srd = np.nanmax(np.column_stack(
            [np.nan_to_num(get("SRD5A1")), np.nan_to_num(get("SRD5A2"))]), axis=1)
        hsd = np.nanmean(np.column_stack(
            [np.nan_to_num(get(g)) for g in ("HSD17B1", "HSD17B2", "HSD17B3")]), axis=1)
        ugt = np.nan_to_num(get("UGT2B7")); aro = np.nan_to_num(get("CYP19A1"))
        ar = np.nan_to_num(get("AR"));      shbg = np.nan_to_num(get("SHBG"))
        det = 0.25 * cyp + 0.30 * srd + 0.20 * ugt + 0.15 * aro + 0.10 * hsd
        ei = 0.5 * ar + 0.3 * shbg + 0.2 * aro
        return det, ei

    def run_weights(w):
        num = np.zeros_like(M["swiss"]); den = np.zeros_like(M["swiss"])
        for k, wk in zip(ORDER, w):
            mask = ~np.isnan(M[k])
            num += np.where(mask, wk * np.nan_to_num(M[k]), 0)
            den += np.where(mask, wk, 0)
        with np.errstate(invalid="ignore"):
            return det_ei(num / den)

    # baseline (manuscript weights)
    det_b, ei_b = run_weights([W[k] for k in ORDER])
    base_iii = [m for m, d, e in zip(mols, det_b, ei_b)
                if risk_class(d, e) == "III"]
    print(f"Baseline Class III (n={len(base_iii)}): {base_iii}")

    # Dirichlet sampling
    rng = np.random.default_rng(seed)
    rho_det, rho_ei, iii_sets = [], [], []
    for w in rng.dirichlet(np.ones(4), size=n_iter):
        d, e = run_weights(w)
        rho_det.append(stats.spearmanr(d, det_b).statistic)
        rho_ei.append(stats.spearmanr(e, ei_b).statistic)
        iii_sets.append(frozenset(
            m for m, dd, ee in zip(mols, d, e) if risk_class(dd, ee) == "III"))
    rho_det = np.array(rho_det); rho_ei = np.array(rho_ei)

    print(f"Spearman Det: median {np.median(rho_det):.4f} "
          f"95% CI [{np.percentile(rho_det, 2.5):.4f}, "
          f"{np.percentile(rho_det, 97.5):.4f}]")
    print(f"Spearman EI:  median {np.median(rho_ei):.4f} "
          f"95% CI [{np.percentile(rho_ei, 2.5):.4f}, "
          f"{np.percentile(rho_ei, 97.5):.4f}]")

    stab = {m: 100 * np.mean([m in s for s in iii_sets]) for m in base_iii}
    res = (pd.DataFrame({"compound": list(stab), "stability_pct": list(stab.values())})
           .sort_values("stability_pct", ascending=False))
    res.to_csv(HERE / "sensibilidad_pesos_claseIII.csv", index=False)
    print(res.to_string(index=False))

    # reference scenarios
    for name, w in [("equal weights", [.25] * 4), ("Swiss only", [1, 0, 0, 0]),
                    ("FGP only", [0, 1, 0, 0]), ("PLATO only", [0, 0, 1, 0]),
                    ("PHARMAPPER only", [0, 0, 0, 1])]:
        d, e = run_weights(w)
        print(f"{name:17s} rho Det {stats.spearmanr(d, det_b).statistic:.3f}  "
              f"rho EI {stats.spearmanr(e, ei_b).statistic:.3f}")

    # ---- figure ---------------------------------------------------------- #
    C_DARK, C_MID = "#2E5E4E", "#8FC1B5"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    ax.hist(rho_det, bins=40, color=C_DARK, alpha=0.85,
            label=f"Detectability (median {np.median(rho_det):.3f})")
    ax.hist(rho_ei, bins=40, color=C_MID, alpha=0.75,
            label=f"Endocrine Impact (median {np.median(rho_ei):.3f})")
    d_eq, _ = run_weights([.25] * 4)
    d_sw, _ = run_weights([1, 0, 0, 0])
    d_fg, _ = run_weights([0, 1, 0, 0])
    for v, c, lab in [(stats.spearmanr(d_eq, det_b).statistic, "#4A5D8A", " equal"),
                      (stats.spearmanr(d_sw, det_b).statistic, "#8E6C3A", " Swiss only"),
                      (stats.spearmanr(d_fg, det_b).statistic, "#C0392B", " FGP only")]:
        ax.axvline(v, color=c, ls="--", lw=1.6)
        ax.text(v, ax.get_ylim()[1] * 0.97, lab, color=c, fontsize=8.5, va="top")
    ax.set_xlabel("Spearman ρ vs baseline ranking (1,000 random weightings)")
    ax.set_ylabel("Frequency")
    ax.set_title("a) Ranking stability under weight perturbation",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    names = sorted(stab, key=lambda m: stab[m])
    vals = [stab[m] for m in names]
    colors = [C_DARK if v >= 90 else (C_MID if v >= 65 else "#C0392B") for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(90, color="#888", ls="--", lw=1)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of weightings retaining Class III (stealth)")
    ax.set_title("b) Stealth-steroid stability (baseline Class III, n = 10)",
                 fontsize=11, fontweight="bold", loc="left")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Sensitivity of the consensus framework to platform weights "
                 "(Dirichlet sampling, n = 1,000)", fontsize=12.5, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(HERE / "FigS1_sensibilidad_pesos.png", dpi=220, bbox_inches="tight")
    print("Figure written:", HERE / "FigS1_sensibilidad_pesos.png")


if __name__ == "__main__":
    main()
