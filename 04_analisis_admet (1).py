#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_analisis_admet.py
====================
ADMETlab 3.0 analysis of the 68 WADA S1.1 anabolic-androgenic steroids
(supporting code for the manuscript "Computational anti-doping detection
of anabolic agents", Section 2.5, Tables 5-6, Figures 6-7).

The ADMETlab 3.0 export ('ADMET results.xlsx') lost every decimal
separator when it was written (locale issue): probabilities are stored as
long integers (e.g. 0.9438576102256776 -> 9438576102256776) and physical
descriptors as integers scaled by powers of ten (MW 29022 -> 290.22).
The first step of this script reconstructs the original values.

Reproduces, bit-exact:
  * Table 5  - mean toxicity probabilities by structural class
               (17alpha-alkylated n=31 vs non-alkylated n=37).
  * Table 6  - top-10 compounds by predicted human hepatotoxicity.
  * NR-AR statements: P(AR activation) >= 0.9 for 60/68 compounds;
    oxymetholone 0.54, prostanozol 0.46.
  * Figure 6 - toxicological heatmap (68 compounds x 11 endpoints,
               rows sorted by a composite toxicity score).
  * Figure 7 - ADMET overview panels (physicochemical space, PPB,
               HIA vs F30, mean risk by structural class).

Input :  admet_decoded_corregido.csv  (RECOMMENDED: curated table with the
         7 structure-verified ADMETlab 3.0 API re-queries applied and all
         columns on native scales)
         or 'ADMET results.xlsx'      (raw ADMETlab 3.0 export, 68 x 124;
         decimal reconstruction applied, see below)
         HORMONES_ANABOLIC_ALL_TARGETS(1).xlsx  (only for the optional
         cross-platform androgen-receptor agreement check)
Output:  admet_decoded.csv          (decoded table + annotations)
         fig6_toxicity_heatmap.png
         fig7_adme_panel.png

Usage:   python 04_analisis_admet.py [path/to/ADMET results.xlsx] \
                                    [path/to/HORMONES xlsx] [outdir]
Requires: pandas, numpy, matplotlib, openpyxl (Python >= 3.11)
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# 1. Decimal reconstruction                                           #
# ------------------------------------------------------------------ #

def dec_prob(x):
    """Rebuild a probability in [0, 1] stored as bare integer digits.

    The decimal point was removed, so the original value is the integer
    divided by 10**(number of digits).  Keeps the sign (logS is stored
    the same way but negative).
    """
    xi = int(round(float(x)))
    if xi == 0:
        return 0.0
    return xi / (10 ** len(str(abs(xi))))


def dec_scaled(x, n_int):
    """Rebuild a descriptor whose value has ``n_int`` integer digits.

    Examples: MW 29022 -> 290.22 (3 int digits);
              TPSA 4046 -> 40.46 (2 int digits);
              logP/logD/logS use n_int = 1.
    """
    xi = int(round(float(x)))
    s = str(abs(xi))
    val = xi / (10 ** (len(s) - n_int))
    return val


# probability-like endpoints (0-1) decoded with dec_prob
PROB_COLS = [
    "QED", "hia", "f20", "f30", "f50", "PPB", "BSEP",
    "hERG", "hERG-10um", "DILI", "Ames", "Carcinogenicity", "Respiratory",
    "H-HT", "Neurotoxicity-DI", "Hematotoxicity", "Nephrotoxicity-DI",
    "Genotoxicity", "NR-AR", "NR-AR-LBD", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "NR-AhR", "SR-p53",
]
# descriptors decoded with dec_scaled(x, 1): one integer digit
LOG_COLS = ["logS", "logP", "logD"]
# caco2 is log cm/s (one integer digit, negative): NOT a probability
CACO2_COLS = ["caco2"]

# toxicity endpoints compared in Table 5 / panel d
TABLE5_ENDPOINTS = ["H-HT", "DILI", "Nephrotoxicity-DI", "Carcinogenicity",
                    "Genotoxicity", "hERG", "BSEP"]
# composite toxicity score (Figure 6): mean of these 9 endpoints
COMPOSITE_ENDPOINTS = ["H-HT", "DILI", "Carcinogenicity", "Genotoxicity",
                       "Ames", "hERG", "Nephrotoxicity-DI",
                       "Neurotoxicity-DI", "Hematotoxicity"]
# columns shown in the Figure 6 heatmap (in this order)
FIG6_ENDPOINTS = ["H-HT", "DILI", "BSEP", "Carcinogenicity", "Genotoxicity",
                  "Ames", "hERG", "Nephrotoxicity-DI", "Neurotoxicity-DI",
                  "Hematotoxicity", "Respiratory"]
FIG6_LABELS = ["Human\nhepatotox.", "DILI", "BSEP\ninhibition",
               "Carcino-\ngenicity", "Genotoxi-\ncity", "Ames\n(mutag.)",
               "hERG", "Nephro-\ntoxicity", "Neuro-\ntoxicity",
               "Hemato-\ntoxicity", "Respiratory\ntox."]

# ------------------------------------------------------------------ #
# 2. 17alpha-alkylation annotation (structural curation, n = 31)      #
# ------------------------------------------------------------------ #
# Compounds curated as bearing a 17alpha-alkyl (methyl, ethyl or
# ethynyl) substituent.  This is the annotation used throughout the
# manuscript (Table 5, Figures 6-7); names as in the ADMETlab export.
ALKYLATED_17A = {
    "17α-Methylepithiostanol (Epistane)", "Bolasterone", "Calusterone",
    "Danazol", "Dehydrochlormethyltestosterone (Oral Turinabol)",
    "Desoxymethyltestosterone", "Ethylestrenol", "Fluoxymesterone",
    "Furazabol", "Gestrinone", "Mestanolone", "Metandienone (Dianabol)",
    "Methandriol", "Methasterone (Superdrol)", "Methyl-1-testosterone",
    "Methylclostebol", "Methyldienolone", "Methylnortestosterone",
    "Methyltestosterone", "Metribolone (Methyltrienolone)", "Mibolerone",
    "Norboletone", "Norclostebol", "Norethandrolone", "Oxandrolone",
    "Oxymesterone", "Oxymetholone", "Prostanozol", "Stanozolol",
    "Tetrahydrogestrinone (THG)", "Tibolone",
}

# colours (manuscript palette)
C_ALK = "#8B1A1A"      # dark red  -> 17alpha-alkylated
C_NON = "#2E86C1"      # blue      -> non-alkylated


def load_admet(path):
    """Read the ADMETlab export and rebuild all decimal values."""
    ad = pd.read_excel(path)
    ad["name"] = (ad["Compuesto"].astype(str)
                  .str.replace("*", "", regex=False).str.strip())
    out = ad[["name", "smiles"]].copy()
    for c in PROB_COLS:
        if c in ad.columns:
            out[c] = pd.to_numeric(ad[c], errors="coerce").map(dec_prob)
    for c in LOG_COLS:
        if c in ad.columns:
            out[c] = pd.to_numeric(ad[c], errors="coerce").map(
                lambda v: dec_scaled(v, 1))
    for c in CACO2_COLS:
        if c in ad.columns:
            out[c] = pd.to_numeric(ad[c], errors="coerce").map(
                lambda v: dec_scaled(v, 1))
    out["MW"] = pd.to_numeric(ad["MW"], errors="coerce").map(
        lambda v: dec_scaled(v, 3))
    out["TPSA"] = pd.to_numeric(ad["TPSA"], errors="coerce").map(
        lambda v: dec_scaled(v, 2))
    out["Lipinski"] = pd.to_numeric(ad["Lipinski"], errors="coerce")
    out["alkylated"] = out["name"].isin(ALKYLATED_17A)
    # composite toxicity score (mean of 9 endpoints; Figure 6 legend)
    out["composite_tox"] = out[COMPOSITE_ENDPOINTS].mean(axis=1)
    return out


def load_corrected_csv(path):
    """Load the curated, structure-verified table (admet_decoded_corregido.csv).

    Produced after the PubChem re-verification of the 68 canonical SMILES:
    7 compounds whose original export had been computed on wrong structures
    (danazol, epitestosterone, ethylestrenol, gestrinone, norethandrolone,
    testosterone, tibolone) were re-queried through the official ADMETlab 3.0
    API; the caco2 and PPB columns are on their native scales for all rows.
    Extra columns (BBB, OATP1B1/1B3, CYP3A4-sub/inh, t0.5, cl-plasma, Fu_pct)
    are included for the pharmacokinetic statements of Section 3.7.
    """
    out = pd.read_csv(path)
    out["alkylated"] = out["alkylated"].astype(bool)
    return out


# ------------------------------------------------------------------ #
# 3. Tables                                                           #
# ------------------------------------------------------------------ #

def table5(dec):
    """Mean toxicity probabilities by structural class.

    Individual probabilities are rounded to 2 decimals (as reported)
    before averaging.
    """
    rows = {}
    for col in TABLE5_ENDPOINTS:
        v = dec[col].round(2)
        rows[col] = (round(v[~dec["alkylated"]].mean(), 2),
                     round(v[dec["alkylated"]].mean(), 2))
    t5 = pd.DataFrame(rows, index=["Non-alkylated", "17a-alkylated"]).T
    t5.columns = [f"Non-17a-alkylated (n={(~dec['alkylated']).sum()})",
                  f"17a-alkylated (n={dec['alkylated'].sum()})"]
    return t5


def table6(dec, n=10):
    """Top-n compounds by predicted human hepatotoxicity (H-HT)."""
    cols = ["name", "H-HT", "DILI", "BSEP"]
    return (dec.sort_values("H-HT", ascending=False)[cols]
            .head(n).reset_index(drop=True))


# ------------------------------------------------------------------ #
# 4. Figures                                                          #
# ------------------------------------------------------------------ #

def short_label(name):
    """Compound label without the parenthetical synonym."""
    return re.sub(r"\s*\(.*?\)", "", name).strip()


def figure6(dec, out_png):
    """Toxicological heatmap: 68 compounds x 11 endpoints, rows sorted
    by the composite toxicity score; 17a-alkylated marked with '*'."""
    d = dec.sort_values("composite_tox", ascending=False).reset_index(drop=True)
    M = d[FIG6_ENDPOINTS].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 16.0), dpi=200)
    ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(FIG6_ENDPOINTS)))
    ax.set_xticklabels(FIG6_LABELS, fontsize=10)
    ax.xaxis.set_ticks_position("bottom")
    labels = [("* " + short_label(n)) if a else short_label(n)
              for n, a in zip(d["name"], d["alkylated"])]
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(labels, fontsize=9)
    for tick, a in zip(ax.get_yticklabels(), d["alkylated"]):
        if a:
            tick.set_color("#8B1A1A")
    ax.set_xticks(np.arange(-0.5, len(FIG6_ENDPOINTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(d), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Toxicological heatmap — 68 WADA-listed anabolic agents "
                 "(ADMETlab 3.0)\n(red = higher predicted probability; "
                 "* in dark red = 17α-alkylated)", fontsize=12, pad=14)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                               norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Predicted probability", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure7(dec, out_png):
    """Four-panel ADMET overview."""
    alk, non = dec[dec["alkylated"]], dec[~dec["alkylated"]]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 11.0), dpi=200)

    # (a) physicochemical space: logP vs TPSA
    ax = axes[0, 0]
    ax.scatter(alk["logP"], alk["TPSA"], s=45, color=C_ALK, alpha=0.85,
               label="17α-alkylated", zorder=3)
    ax.scatter(non["logP"], non["TPSA"], s=45, color=C_NON, alpha=0.85,
               label="non-alkylated", zorder=3)
    nviol = int(dec["Lipinski"].fillna(0).astype(int).ne(0).sum())
    ax.set_title(f"a) Physicochemical space (all pass Lipinski: "
                 f"{nviol} violations)", fontsize=12)
    ax.set_xlabel("logP (lipophilicity)")
    ax.set_ylabel("TPSA (Å²)")
    ax.legend(fontsize=9, framealpha=0.9)

    # (b) plasma protein binding distribution
    ax = axes[0, 1]
    ppb = dec["PPB"] * 100.0
    ax.hist(ppb, bins=np.arange(57.5, 102.5, 2.5), color="#5B7FBF",
            edgecolor="white", zorder=3)
    ax.axvline(90, color="red", ls="--", lw=1.4, zorder=4)
    ax.text(90.4, ax.get_ylim()[1] * 0.62, "90%", color="red", fontsize=9)
    ax.set_title(f"b) PPB: median {ppb.median():.1f}%  "
                 f"({(ppb >= 90).sum()} compounds ≥90%)", fontsize=12)
    ax.set_xlabel("Plasma protein binding, PPB (%)")
    ax.set_ylabel("No. of compounds")

    # (c) oral absorption: HIA vs bioavailability >= 30%
    ax = axes[1, 0]
    ax.scatter(alk["hia"], alk["f30"], s=45, color=C_ALK, alpha=0.85, zorder=3)
    ax.scatter(non["hia"], non["f30"], s=45, color=C_NON, alpha=0.85, zorder=3)
    ax.set_title("c) Oral absorption: HIA vs bioavailability", fontsize=12)
    ax.set_xlabel("Human intestinal absorption probability (HIA)")
    ax.set_ylabel("Bioavailability ≥30% probability (F30)")

    # (d) mean toxicity probabilities by structural class
    ax = axes[1, 1]
    eps = ["H-HT", "DILI", "Nephrotoxicity-DI", "Carcinogenicity",
           "Genotoxicity", "hERG"]
    labels = ["Hepatotox.", "DILI", "Nephrotox.", "Carcinog.",
              "Genotox.", "hERG"]
    m_non = [dec[c].round(2)[~dec["alkylated"]].mean() for c in eps]
    m_alk = [dec[c].round(2)[dec["alkylated"]].mean() for c in eps]
    x = np.arange(len(eps))
    ax.bar(x - 0.2, m_non, width=0.4, color=C_NON,
           label=f"Non-alkylated (n={(~dec['alkylated']).sum()})", zorder=3)
    ax.bar(x + 0.2, m_alk, width=0.4, color=C_ALK,
           label=f"17α-alkylated (n={dec['alkylated'].sum()})", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean predicted probability")
    ax.set_title("d) Mean risk: 17α-alkylated vs non-alkylated", fontsize=12)
    ax.legend(fontsize=9)

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
# 5. Cross-platform agreement at the androgen receptor                #
# ------------------------------------------------------------------ #

def _norm_name(s):
    """Minimal name canonicalization shared with 01_pipeline_consenso.py."""
    s = str(s).translate(str.maketrans({"ß": "β", "ɑ": "α"}))
    s = re.sub(r"\s*\(.*?\)", "", s)          # drop parenthetical synonyms
    return re.sub(r"\s+", " ", s).strip().lower()


def ar_agreement(dec, hormones_path):
    """Binary agreement between SwissTargetPrediction AR probability
    (>= 0.1) and ADMETlab NR-AR probability (>= 0.5) for the compounds
    with directly comparable names."""
    sw = pd.read_excel(hormones_path, sheet_name="SWISS_TARGET_PREDICTION",
                       header=3)
    sw["m"] = sw["Molecule"].map(_norm_name)
    sw["prob"] = pd.to_numeric(sw["Probability*"], errors="coerce")
    is_ar = (sw["Target"].astype(str).str.replace(" ", " ")
             .str.strip().str.lower().str.startswith("androgen receptor"))
    ar = sw[is_ar].groupby("m")["prob"].max()
    dec["m"] = dec["name"].map(_norm_name)
    both = dec[dec["m"].isin(ar.index)]
    agree = ((ar[both["m"]].values >= 0.1) == (both["NR-AR"].values >= 0.5))
    return len(both), int(agree.sum())


# ------------------------------------------------------------------ #
# main                                                                #
# ------------------------------------------------------------------ #

def main():
    admet_path = (sys.argv[1] if len(sys.argv) > 1
                  else "C:/Users/Mateo/Downloads/WADA-S1-target-fishing-ADMET-main/ADMET results.xlsx")
    horm_path = (sys.argv[2] if len(sys.argv) > 2
                 else "C:/Users/Mateo/Downloads/WADA-S1-target-fishing-ADMET-main/HORMONES_ANABOLIC_ALL_TARGETS(1).xlsx")
    outdir = Path(sys.argv[3] if len(sys.argv) > 3
                  else "C:/Users/Mateo/Downloads/WADA-S1-target-fishing-ADMET-main")

    if str(admet_path).endswith(".csv"):
        # corrected, structure-verified table (recommended; final manuscript)
        dec = load_corrected_csv(admet_path)
    else:
        dec = load_admet(admet_path)
    print(f"Compounds: {len(dec)}  |  17alpha-alkylated: "
          f"{dec['alkylated'].sum()}  non-alkylated: {(~dec['alkylated']).sum()}")
    dec.to_csv(outdir / "admet_decoded.csv", index=False)

    print("\n--- Table 5. Toxicity endpoints by structural class "
          "(mean predicted probabilities) ---")
    print(table5(dec).to_string())

    print("\n--- Table 6. Top-10 by human hepatotoxicity (H-HT) ---")
    t6 = table6(dec)
    print(t6.round(3).to_string(index=False))

    print("\n--- NR-AR (androgen receptor activation) ---")
    print(f"P(NR-AR) >= 0.9: {(dec['NR-AR'] >= 0.9).sum()}/68")
    for nm in ("Oxymetholone", "Prostanozol"):
        v = dec.loc[dec["name"] == nm, "NR-AR"].iloc[0]
        print(f"  {nm}: {v:.2f}")

    figure6(dec, outdir / "fig6_toxicity_heatmap.png")
    figure7(dec, outdir / "fig7_adme_panel.png")
    print("\nFigures written: fig6_toxicity_heatmap.png, fig7_adme_panel.png")

    if Path(horm_path).exists():
        n, k = ar_agreement(dec, horm_path)
        print(f"\n--- Cross-platform AR agreement ---")
        print(f"Comparable compounds: {n} | agreement: {k}/{n} "
              f"= {100*k/n:.1f}%  (manuscript: 60/62 = 96.8%)")


if __name__ == "__main__":
    main()
