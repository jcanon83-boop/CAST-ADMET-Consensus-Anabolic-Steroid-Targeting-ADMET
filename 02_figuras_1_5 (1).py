#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_figuras_1_5.py
=================
Regenerates Figures 1-5 of the manuscript from the outputs of
01_pipeline_consenso.py (run that script first).

  Figure 1  fig1_clustering.png    Hierarchical clustering of 67 AAS
                                   (Ward / Euclidean, top-132 targets by
                                   maximum consensus; 3 clusters)
  Figure 2  fig2_scatter.png       Detectability vs Endocrine Impact,
                                   coloured by risk class
  Figure 3  fig3_enzymes.png       Heatmap of metabolic-enzyme consensus
  Figure 4  fig4_corrections.png   SwissTargetPrediction-only vs consensus
                                   detectability, 12 largest corrections
  Figure 5  fig5_consensus_boxplot.png
                                   Consensus distribution by number of
                                   contributing platforms (1-4)

Requires: pandas, numpy, matplotlib, scipy.  Python >= 3.11.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
MATRIX = "consensus_matrix.csv"      # rows = compounds, cols = UniProt IDs
LONG   = "consensus_long.csv"
INDEX  = "indices_clases.csv"
GENES  = "uniprot_genes.csv"

matrix = pd.read_csv(MATRIX, index_col=0)
long   = pd.read_csv(LONG)
idx    = pd.read_csv(INDEX, index_col=0)
genes  = pd.read_csv(GENES, index_col=0).iloc[:, 0].to_dict()

UNIPROT = {   # pathway enzymes (Sections 2.4-2.5)
    "CYP3A4": "P08684", "CYP2C9": "P11712", "CYP2C19": "P33261",
    "CYP2D6": "P10635", "CYP17A1": "P05093", "CYP19A1": "P11511",
    "CYP11B1": "P15538", "CYP11B2": "P19099", "CYP51A1": "Q16850",
    "SRD5A1": "P18405", "SRD5A2": "P31213", "UGT2B7": "P16662",
    "HSD17B1": "P14061", "HSD17B2": "P37059", "HSD17B3": "P37058",
    "AR": "P10275", "SHBG": "P04278",
}

# manuscript palette
CLUSTER_COLORS = {1: "#2E5E4E", 2: "#8E6C3A", 3: "#4A5D8A"}
CLASS_COLORS = {"I": "#C0392B", "II": "#E67E22", "III": "#7D3C98", "IV": "#17A2B8"}

sym = lambda u: genes.get(u, u)

# ---------------------------------------------------------------------------
# Figure 1 - hierarchical clustering
# ---------------------------------------------------------------------------
X = matrix.fillna(0.0)
top132 = X.max().sort_values(ascending=False).index[:132]   # top targets by max consensus
Xc = X[top132]
Z = linkage(pdist(Xc.values), method="ward")
labels = pd.Series(fcluster(Z, t=3, criterion="maxclust"), index=Xc.index)

# order clusters as in the manuscript: 1 = designer/17alpha (n=16),
# 2 = DHT derivatives (n=8), 3 = remaining (n=43)
sizes = labels.value_counts()
renum = {sizes.index[sizes == 16][0]: 1,
         sizes.index[sizes == 8][0]: 2,
         sizes.index[sizes == 43][0]: 3}
labels = labels.map(renum)

fig, ax = plt.subplots(figsize=(7.5, 13))
dn = dendrogram(Z, ax=ax, orientation="right", labels=list(Xc.index),
                leaf_font_size=7.5, color_threshold=0,
                above_threshold_color="#9AA8A2")
for tick, mol in zip(ax.get_yticklabels(), dn["ivl"]):
    tick.set_color(CLUSTER_COLORS[labels[mol]])
ax.set_xlabel("Ward distance (Euclidean)", fontsize=10)
ax.set_title("Hierarchical clustering of 67 AAS by consensus target profiles",
             fontsize=12)
handles = [Patch(facecolor=CLUSTER_COLORS[k],
                 label=f"Cluster {k} (n={int((labels == k).sum())})")
           for k in (1, 2, 3)]
ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig1_clustering.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 - Detectability vs Endocrine Impact scatter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 6.2))
for klass in ["I", "II", "III", "IV"]:
    sub = idx[idx["Class"] == klass]
    ax.scatter(sub["Det"], sub["EI"], s=42, color=CLASS_COLORS[klass],
               edgecolor="white", linewidth=0.6, alpha=0.9,
               label=f"Class {klass} (n={len(sub)})", zorder=3)
ax.axvline(0.35, ls="--", lw=1, color="#8a9a90")
ax.axvline(0.15, ls="--", lw=1, color="#8a9a90")
ax.axhline(0.30, ls="--", lw=1, color="#8a9a90")
for mol in ["Trenbolone", "Metribolone", "Fluoxymesterone", "Tibolone",
            "Oxymetholone", "Furazabol"]:
    if mol in idx.index:
        ax.annotate(mol, (idx.loc[mol, "Det"], idx.loc[mol, "EI"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=7.5,
                    color="#37474F")
ax.set_xlabel("Detectability Index", fontsize=11)
ax.set_ylabel("Endocrine Impact Index", fontsize=11)
ax.set_title("Detectability vs endocrine impact of 67 AAS", fontsize=12)
ax.legend(fontsize=9, loc="lower right", frameon=True)
ax.grid(alpha=0.25, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig2_scatter.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 - metabolic-enzyme consensus heatmap
# ---------------------------------------------------------------------------
ENZ = ["CYP3A4", "CYP2C9", "CYP2C19", "CYP2D6", "CYP17A1", "CYP19A1",
       "CYP11B1", "CYP11B2", "CYP51A1", "SRD5A1", "SRD5A2", "UGT2B7",
       "HSD17B1", "HSD17B2", "HSD17B3"]
H = matrix[[UNIPROT[g] for g in ENZ]].copy()
H.columns = ENZ
H = H.loc[idx["Det"].sort_values(ascending=False).index]   # sort by detectability

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("teal",
        ["#FFFFFF", "#D8E8E2", "#8FC1B5", "#2E5E4E"])
fig, ax = plt.subplots(figsize=(8.6, 11))
im = ax.imshow(H.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
ax.set_xticks(range(len(ENZ))); ax.set_xticklabels(ENZ, rotation=45, ha="left",
                                                  fontsize=8.5)
ax.xaxis.set_ticks_position("top")
ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index, fontsize=6.4)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-.5, len(ENZ), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(H), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.3)
ax.tick_params(which="minor", length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cb.set_label("Consensus score", fontsize=9)
ax.set_title("Metabolic-enzyme consensus scores (compounds sorted by Detectability)",
             fontsize=11, pad=52)
fig.tight_layout()
fig.savefig("fig3_enzymes.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4 - SwissTargetPrediction-only vs consensus detectability (top 12)
# ---------------------------------------------------------------------------
top12 = idx["diff"].sort_values(ascending=False).head(12).index
d = idx.loc[top12].iloc[::-1]
y = np.arange(len(d))
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.barh(y, d["Det"], height=0.62, color="#2E5E4E", alpha=0.92,
        label="Consensus detectability")
ax.barh(y, d["Det_swiss"], height=0.62, color="#9FB8AE",
        label="SwissTargetPrediction only")
for i, (dv, sv) in enumerate(zip(d["Det"], d["Det_swiss"])):
    ax.text(dv + 0.004, i, f"{dv:.3f}", va="center", fontsize=8, color="#2E5E4E")
    ax.text(sv + 0.004, i - 0.28, f"{sv:.3f}", va="center", fontsize=7,
            color="#5a6b62")
ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=9)
ax.set_xlabel("Detectability Index", fontsize=11)
ax.set_title("SwissTargetPrediction-only vs consensus detectability - "
             "12 largest corrections", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig4_corrections.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5 - consensus distribution by number of contributing platforms
# ---------------------------------------------------------------------------
groups = [1, 2, 3, 4]
data = [long.loc[long["n"] == g, "consensus"].values for g in groups]
rng = np.random.default_rng(42)
box_cols = ["#DCE6EC", "#B8D4CE", "#8FC1B5", "#5DA493"]
edge_col = "#2E5E4E"

fig, ax = plt.subplots(figsize=(8.5, 6))
bp = ax.boxplot(data, positions=groups, widths=0.52, patch_artist=True,
                showfliers=False,
                medianprops=dict(color="#B03A2E", lw=2.2),
                boxprops=dict(color=edge_col, lw=1.2),
                whiskerprops=dict(color=edge_col, lw=1.2),
                capprops=dict(color=edge_col, lw=1.2))
for patch, c in zip(bp["boxes"], box_cols):
    patch.set_facecolor(c)
for g, dd in zip(groups, data):
    dd = dd if len(dd) <= 400 else rng.choice(dd, 400, replace=False)
    ax.scatter(np.full(len(dd), g) + rng.uniform(-0.16, 0.16, len(dd)), dd,
               s=7, color="#37474F", alpha=0.25 if len(dd) > 100 else 0.45,
               zorder=3, edgecolors="none")
    med = np.median(dd)
    ax.annotate(f"median = {med:.3f}", (g, med), textcoords="offset points",
                xytext=(30, -4), fontsize=8.5, color="#B03A2E",
                fontweight="bold")
    ax.annotate(f"n = {len(dd):,}", (g, -0.14), ha="center", fontsize=8.5,
                color="#5a6b62")
ax.set_xticks(groups)
ax.set_xticklabels([f"{g} platform{'s' if g > 1 else ''}" for g in groups],
                   fontsize=10)
ax.set_xlabel("Number of contributing platforms", fontsize=11)
ax.set_ylabel("Consensus score", fontsize=11)
ax.set_ylim(-0.18, 1.05)
ax.set_title("Consensus score distribution by number of contributing platforms",
             fontsize=12)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig5_consensus_boxplot.png", dpi=200)
plt.close(fig)

print("Figures 1-5 written: fig1_clustering.png, fig2_scatter.png, "
      "fig3_enzymes.png, fig4_corrections.png, fig5_consensus_boxplot.png")
print("Clusters:", labels.value_counts().sort_index().to_dict())
print("Top-12 corrections:", list(top12))
