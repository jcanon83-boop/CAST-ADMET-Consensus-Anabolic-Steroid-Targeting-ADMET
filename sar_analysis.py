"""
SAR analysis for Steroids journal submission.
Merges the structural classification (aas_structural_classification.csv) with
the per-compound ADMET/target-fishing data and produces:
  - sar_table.csv            : mean ± SD of each endpoint by structural class
  - sar_stats.csv            : Kruskal-Wallis p-values + Holm-corrected pairwise MWU
  - fig_sar_boxplots.png     : boxplots of key endpoints by class
  - fig_sar_flags.png        : endpoints as a function of structural flags (17a-alkyl, 19-nor, D1, hetero, halo)

Expected input (edit the path below). One row per compound. Column names are
flexible; map them in COLMAP. 'compound' must contain the compound name.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ------------------- CONFIGURATION -------------------
DATA_PATH   = "aas_data.csv"          # <- your per-compound data (ADMETlab + DI/EII)
CLASS_PATH  = "aas_structural_classification.csv"
OUTDIR      = "."

# Map YOUR column names -> canonical names used here
COLMAP = {
    "compound":        "compound",   # e.g. "compound", "name", "Name"
    "h_ht":            "h_ht",       # human hepatotoxicity probability
    "dili":            "dili",
    "bsep":            "bsep",       # BSEP inhibition probability
    "carcinogenicity": "carcinogenicity",
    "genotoxicity":    "genotoxicity",
    "ames":            "ames",
    "herg":            "herg",
    "nephrotoxicity":  "nephrotoxicity",
    "nr_ar":           "nr_ar",      # ADMETlab NR-AR probability
    "ar":              "ar_cons",    # consensus AR (target-fishing layer)
    "di":              "di",         # Detectability Index
    "eii":             "eii",        # Endocrine Impact Index
}

ENDPOINTS = ["h_ht", "dili", "bsep", "carcinogenicity",
             "nephrotoxicity", "herg", "nr_ar", "di", "eii"]

# ------------------- LOAD -------------------
data  = pd.read_csv(DATA_PATH)
classif = pd.read_csv(CLASS_PATH)

data = data.rename(columns={v: k for k, v in COLMAP.items() if v in data.columns})
data["compound_l"] = (data["compound"].str.lower()
                      .str.replace(r"[^a-z0-9]", "", regex=True))
classif["compound_l"] = (classif["compound"].str.lower()
                         .str.replace(r"[^a-z0-9]", "", regex=True))

merged = classif.merge(data, on="compound_l", how="outer", indicator=True,
                       suffixes=("", "_data"))
unmatched = merged.loc[merged["_merge"] != "both", ["compound", "compound_data", "_merge"]]
if len(unmatched):
    print("WARNING - unmatched rows (fix names or add to classification):")
    print(unmatched.to_string(index=False))
merged = merged.loc[merged["_merge"] == "both"].copy()
print(f"Merged {len(merged)} / {len(data)} compounds")

FLAG_COLS = ["a17", "nor19", "d1", "d9_11", "hetero", "halo", "cmod", "o7"]

# ------------------- 1. TABLE BY PRIMARY CLASS -------------------
rows = []
for ep in ENDPOINTS:
    for cls, g in merged.groupby("primary_class"):
        v = pd.to_numeric(g[ep], errors="coerce").dropna()
        if len(v) >= 2:
            rows.append({"endpoint": ep, "class": cls, "n": len(v),
                         "mean": v.mean(), "sd": v.std()})
sar = pd.DataFrame(rows).pivot(index="endpoint", columns="class",
                               values="mean").round(3)
n_tab = pd.DataFrame(rows).pivot(index="endpoint", columns="class", values="n")
sar.to_csv(f"{OUTDIR}/sar_table.csv")
print("\nMean by structural class:\n", sar)

# ------------------- 2. KRUSKAL-WALLIS + HOLM-CORRECTED PAIRWISE -------------------
def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        prev = max(val, prev)
        adj[idx] = prev
    return adj

stat_rows = []
for ep in ENDPOINTS:
    groups = [pd.to_numeric(g[ep], errors="coerce").dropna()
              for _, g in merged.groupby("primary_class") if len(g) >= 2]
    if len(groups) < 2:
        continue
    H, p_kw = stats.kruskal(*groups)
    stat_rows.append({"endpoint": ep, "test": "Kruskal-Wallis",
                      "stat": round(H, 3), "p": p_kw})
    # pairwise Mann-Whitney with Holm correction
    labels = [cls for cls, g in merged.groupby("primary_class") if len(g) >= 2]
    pairs, pvals = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = pd.to_numeric(merged.loc[merged.primary_class == labels[i], ep],
                              errors="coerce").dropna()
            b = pd.to_numeric(merged.loc[merged.primary_class == labels[j], ep],
                              errors="coerce").dropna()
            if len(a) >= 2 and len(b) >= 2:
                _, p = stats.mannwhitneyu(a, b)
                pairs.append(f"{labels[i]} vs {labels[j]}")
                pvals.append(p)
    if pvals:
        padj = holm(pvals)
        for pair, p, pa in zip(pairs, pvals, padj):
            stat_rows.append({"endpoint": ep, "test": "MWU Holm",
                              "stat": pair, "p": p, "p_adj": pa})
pd.DataFrame(stat_rows).to_csv(f"{OUTDIR}/sar_stats.csv", index=False)
print("\nStats saved -> sar_stats.csv")

# ------------------- 3. FIGURE: BOXPLOTS BY CLASS -------------------
order = merged.groupby("primary_class")["h_ht"].median().sort_values(ascending=False).index
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, ep in zip(axes.ravel(), ENDPOINTS):
    d = [(pd.to_numeric(g[ep], errors="coerce").dropna(), cls)
         for cls, g in merged.groupby("primary_class")]
    d = sorted(d, key=lambda t: -t[0].median())
    ax.boxplot([t[0] for t in d], tick_labels=[t[1] for t in d], showfliers=False)
    ax.set_title(ep); ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(-0.02, 1.02)
fig.suptitle("ADMET / target-fishing endpoints by primary structural class")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig_sar_boxplots.png", dpi=300)

# ------------------- 4. FIGURE: BY STRUCTURAL FLAG -------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, ep in zip(axes.ravel(), ENDPOINTS):
    pos = pd.to_numeric(merged.loc[merged["a17"] == 1, ep], errors="coerce").dropna()
    neg = pd.to_numeric(merged.loc[merged["a17"] == 0, ep], errors="coerce").dropna()
    ax.boxplot([neg, pos], tick_labels=["non-17α-alkyl", "17α-alkyl"], showfliers=False)
    _, p = stats.mannwhitneyu(neg, pos)
    ax.set_title(f"{ep}\n17α-alkyl effect, MWU p = {p:.4f}")
    ax.set_ylim(-0.02, 1.02)
fig.suptitle("Effect of 17α-alkylation across endpoints")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig_sar_flags.png", dpi=300)
print("Figures saved.")
