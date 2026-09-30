#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_figura1_alternativa.py
=========================
Alternative version of Figure 1 (clustermap layout): the 67 AAS are
hierarchically clustered exactly as in the manuscript Figure 1
(Ward / Euclidean on the top-132 targets by maximum consensus score),
but compounds and targets are displayed as a dual-dendrogram heatmap
with annotation strips for cluster membership and risk class.

  Rows    : 67 compounds (Ward dendrogram, left)
  Columns : 27 most informative targets - top 24 by variance of the
            consensus scores plus AR, CYP19A1 and UGT2B7
            (Ward dendrogram, top; gene symbols)
  Strips  : cluster (1/2/3) and risk class (I-IV)
  Labels  : compound names on the right, coloured by cluster

Input : consensus_matrix.csv, indices_clases.csv, uniprot_genes.csv
        (outputs of 01_pipeline_consenso.py - run that script first)
Output: figura1_alternativa_clustermap.png

Requires: pandas, numpy, matplotlib, scipy.  Python >= 3.11.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import (linkage, dendrogram, fcluster,
                                     leaves_list)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
matrix = pd.read_csv("consensus_matrix.csv", index_col=0)
idx    = pd.read_csv("indices_clases.csv", index_col=0)
genes  = pd.read_csv("uniprot_genes.csv", index_col=0).iloc[:, 0].to_dict()

sym = lambda u: genes.get(u, u)

CLUSTER_COLORS = {1: "#2E5E4E", 2: "#8E6C3A", 3: "#4A5D8A"}
CLASS_COLORS = {"I": "#C0392B", "II": "#E67E22", "III": "#7D3C98",
                "IV": "#17A2B8"}
TEAL = LinearSegmentedColormap.from_list(
    "teal", ["#FFFFFF", "#D8E8E2", "#8FC1B5", "#2E5E4E"])
GREY = "#8A938F"

# ---------------------------------------------------------------------------
# Row clustering - identical to manuscript Figure 1 (script 02)
# ---------------------------------------------------------------------------
X = matrix.fillna(0.0)
top132 = X.max().sort_values(ascending=False).index[:132]
Xc = X[top132]
Z = linkage(pdist(Xc.values), method="ward")
labels = pd.Series(fcluster(Z, t=3, criterion="maxclust"), index=Xc.index)
sizes = labels.value_counts()
renum = {sizes.index[sizes == 16][0]: 1,
         sizes.index[sizes == 8][0]: 2,
         sizes.index[sizes == 43][0]: 3}
labels = labels.map(renum)

# ---------------------------------------------------------------------------
# Column selection: 24 most variable targets + AR, CYP19A1, UGT2B7 (27 total)
# The explicit UniProt list below is the selection shown in the figure.
# ---------------------------------------------------------------------------
_SELECTED = ["SHBG", "AR", "NR3C1", "SERPINA6", "CYP19A1", "G6PD", "HSD17B3",
             "GPBAR1", "UGT2B7", "NR1I2", "FABP1", "SRD5A2", "SRD5A1",
             "SIGMAR1", "CYP17A1", "NR3C2", "PGR", "CHRM2", "SLC6A2",
             "NPC1L1", "ESR1", "ADORA3", "SLC6A3", "CYP2C9", "PTPN11",
             "CYP51A1", "PDE4D"]
_sym2u = {v: k for k, v in genes.items()}
feat = [_sym2u[s] for s in _SELECTED]            # 27 targets (as published)

H = Xc[feat]
Zc = linkage(pdist(H.T.values), method="ward")
col_leaves = leaves_list(Zc)
H = H.iloc[:, col_leaves]

# display order: scipy plots leaf 0 at the BOTTOM for orientation="left",
# whereas imshow row 0 is at the TOP -> reverse the leaf order
disp_order = [Xc.index[i] for i in leaves_list(Z)][::-1]
H = H.loc[disp_order]

# ---------------------------------------------------------------------------
# Layout: 2 x 4 grid (row dendrogram | cluster strip | class strip | heatmap)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13.6, 13.2), dpi=220)
gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 0.16, 0.16, 5.2],
                      height_ratios=[1.1, 8.0],
                      left=0.025, right=0.74, top=0.94, bottom=0.055,
                      wspace=0.02, hspace=0.09)

# -- column dendrogram (top, over the heatmap) ------------------------------
ax_c = fig.add_subplot(gs[0, 3])
dn_c = dendrogram(Zc, ax=ax_c, orientation="top", no_labels=True,
                  color_threshold=0, above_threshold_color=GREY)
for ln in ax_c.lines:
    ln.set_linewidth(0.9)
ax_c.axis("off")

# -- row dendrogram (left) ---------------------------------------------------
ax_r = fig.add_subplot(gs[1, 0])
dendrogram(Z, ax=ax_r, orientation="left", no_labels=True,
           color_threshold=0, above_threshold_color=GREY)
for ln in ax_r.lines:
    ln.set_linewidth(0.9)
ax_r.axis("off")

# -- annotation strips -------------------------------------------------------
clu_vec = np.array([[labels[m]] for m in disp_order])
cls_vec = np.array([[{"I": 1, "II": 2, "III": 3, "IV": 4}[idx.loc[m, "Class"]]
                     for m in disp_order]]).T
cmap_clu = ListedColormap([CLUSTER_COLORS[1], CLUSTER_COLORS[2],
                           CLUSTER_COLORS[3]])
cmap_cls = ListedColormap([CLASS_COLORS[k] for k in ("I", "II", "III", "IV")])
for col, vec, cm, kmax in ((1, clu_vec, cmap_clu, 3), (2, cls_vec, cmap_cls, 4)):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(vec, aspect="auto", cmap=cm, vmin=0.5, vmax=kmax + 0.5,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# -- heatmap -----------------------------------------------------------------
ax_h = fig.add_subplot(gs[1, 3])
im = ax_h.imshow(H.values, aspect="auto", cmap=TEAL, vmin=0, vmax=1,
                 interpolation="nearest")
ax_h.set_xticks(range(H.shape[1]))           # gene symbols on top
ax_h.set_xticklabels([sym(u) for u in H.columns], rotation=75, ha="left",
                     fontsize=8.5)
ax_h.xaxis.set_ticks_position("top")
ax_h.set_yticks(range(len(H)))
ax_h.set_yticklabels(H.index, fontsize=7.0)
ax_h.yaxis.tick_right()
ax_h.tick_params(length=0)
for tick, mol in zip(ax_h.get_yticklabels(), disp_order):
    tick.set_color(CLUSTER_COLORS[labels[mol]])
ax_h.set_xticks(np.arange(-.5, H.shape[1], 1), minor=True)
ax_h.set_yticks(np.arange(-.5, len(H), 1), minor=True)
ax_h.grid(which="minor", color="white", linewidth=0.35)
ax_h.tick_params(which="minor", length=0)
for s in ax_h.spines.values():
    s.set_visible(False)

# -- colour bar --------------------------------------------------------------
cax = fig.add_axes([0.905, 0.68, 0.009, 0.18])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Consensus score", fontsize=10)
cb.ax.tick_params(labelsize=8)

# -- legends -----------------------------------------------------------------
n_clu = labels.value_counts().sort_index()
h_clu = [Patch(facecolor=CLUSTER_COLORS[k], label=f"Cluster {k} (n={n_clu[k]})")
         for k in (1, 2, 3)]
leg1 = fig.legend(handles=h_clu, loc="upper left", bbox_to_anchor=(0.885, 0.60),
                  fontsize=9, title="Cluster", title_fontsize=10,
                  frameon=False)
n_cls = idx["Class"].value_counts()
h_cls = [Patch(facecolor=CLASS_COLORS[k], label=f"Class {k} (n={n_cls[k]})")
         for k in ("I", "II", "III") if k in n_cls]
fig.legend(handles=h_cls, loc="upper left", bbox_to_anchor=(0.885, 0.47),
           fontsize=9, title="Risk class", title_fontsize=10,
           frameon=False)

# -- title and footnote ------------------------------------------------------
fig.text(0.30, 0.965,
         "Hierarchical clustering of 67 AAS by consensus target profile",
         fontsize=15, fontweight="bold", ha="center")
fig.text(0.30, 0.018,
         "Rows: compounds (Ward, Euclidean, 132-target consensus matrix). "
         "Columns: 27 most informative\n"
         "targets (top 24 by variance + AR, CYP19A1, UGT2B7), Ward-clustered. "
         "Strips: cluster and risk class.",
         fontsize=8.5, color="#5a6b62", ha="center")

fig.savefig("figura1_alternativa_clustermap.png", dpi=220)
plt.close(fig)
print("Alternative Figure 1 written: figura1_alternativa_clustermap.png")
print("Clusters:", n_clu.to_dict(), "| features:", [sym(u) for u in H.columns])
