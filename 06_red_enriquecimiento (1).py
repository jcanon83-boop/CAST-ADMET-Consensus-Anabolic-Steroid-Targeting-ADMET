#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_red_enriquecimiento.py — Protein-protein interaction network and functional
enrichment of the 132-target consensus set (STRING v12, Homo sapiens).

The 132 targets are selected exactly as in script 02 (highest maximum
consensus score across the 67 AAS). The script queries the STRING API for:

  1) the PPI network among the 132 proteins (confidence >= 0.4) and its
     evidence-flavoured image  -> FigS2_red_STRING_132_dianas.png
  2) functional enrichment (GO Biological Process, KEGG, Reactome)
     -> enriquecimiento_STRING_132.csv  (full table, 2,547 terms)
     -> FigS3_enriquecimiento_funcional.png  (dot plot of key terms)

Requires: Python >= 3.11, pandas, numpy, matplotlib, requests
          (internet access to https://string-db.org)
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
STRING = "https://string-db.org/api"
CALLER = "aas_consensus_manuscript"

# curated key terms for the dot plot (category, description)
SEL_TERMS = [
    ("Process", "Steroid metabolic process"),
    ("Process", "Response to xenobiotic stimulus"),
    ("Process", "Cellular response to steroid hormone stimulus"),
    ("Process", "Hormone-mediated signaling pathway"),
    ("KEGG", "Steroid hormone biosynthesis"),
    ("KEGG", "Estrogen signaling pathway"),
    ("KEGG", "Progesterone-mediated oocyte maturation"),
    ("KEGG", "Neuroactive ligand-receptor interaction"),
    ("KEGG", "Ovarian steroidogenesis"),
    ("RCTM", "Metabolism of steroids"),
    ("RCTM", "Nuclear Receptor transcription pathway"),
    ("RCTM", "Metabolism of steroid hormones"),
]
CAT_LABEL = {"Process": "GO Biological Process", "KEGG": "KEGG", "RCTM": "Reactome"}


def main():
    cl = pd.read_csv(HERE / "consensus_long.csv")
    genes = pd.read_csv(HERE / "uniprot_genes.csv", index_col=0).iloc[:, 0].to_dict()

    # top-132 targets by maximum consensus (same criterion as script 02)
    mat = cl.pivot_table(index="mol", columns="uniprot", values="consensus")
    top132 = mat.max().sort_values(ascending=False).index[:132].tolist()
    symbols = [genes[u] for u in top132]
    ids = "%0d".join(symbols)
    print(f"{len(symbols)} targets submitted to STRING")

    # 1) network edges + image ------------------------------------------------
    net = requests.get(f"{STRING}/tsv/network",
                       params={"identifiers": ids, "species": 9606,
                               "required_score": 400, "caller_identity": CALLER},
                       timeout=120)
    edges = pd.read_csv(io.StringIO(net.text), sep="\t")
    print(f"network: {len(edges)} edges, "
          f"{pd.concat([edges.preferredName_A, edges.preferredName_B]).nunique()} nodes")

    img = requests.get(f"{STRING}/image/network",
                       params={"identifiers": ids, "species": 9606,
                               "required_score": 400, "network_flavor": "evidence",
                               "caller_identity": CALLER}, timeout=180)
    img.raise_for_status()
    (HERE / "FigS2_red_STRING_132_dianas.png").write_bytes(img.content)
    print("FigS2 written")

    # 2) enrichment -----------------------------------------------------------
    enr = requests.get(f"{STRING}/tsv/enrichment",
                       params={"identifiers": ids, "species": 9606,
                               "caller_identity": CALLER}, timeout=120)
    enr_df = pd.read_csv(io.StringIO(enr.text), sep="\t")
    enr_df[["category", "term", "description", "number_of_genes",
            "number_of_genes_in_background", "p_value", "fdr",
            "preferredNames"]].to_csv(HERE / "enriquecimiento_STRING_132.csv",
                                      index=False)
    print(f"enrichment table written ({len(enr_df)} terms)")

    rows = []
    for cat, desc in SEL_TERMS:
        m = enr_df[(enr_df.category == cat) & (enr_df.description == desc)]
        if len(m):
            r = m.iloc[0]
            rows.append(dict(category=CAT_LABEL[cat], description=desc,
                             n=r.number_of_genes, fdr=r.fdr))
    dot = pd.DataFrame(rows).sort_values("fdr").reset_index(drop=True)
    print(dot.to_string(index=False))

    # 3) dot plot -------------------------------------------------------------
    cat_colors = {"GO Biological Process": "#2E5E4E", "KEGG": "#8E6C3A",
                  "Reactome": "#4A5D8A"}
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    y = np.arange(len(dot))[::-1]
    for yi, (_, r) in zip(y, dot.iterrows()):
        ax.scatter(-np.log10(r.fdr), yi, s=40 + r["n"] * 22,
                   color=cat_colors[r.category], alpha=0.85,
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(-np.log10(r.fdr) + 0.35, yi, f"n={r['n']}", va="center",
                fontsize=8, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels(dot["description"], fontsize=10)
    ax.set_xlabel("−log₁₀(FDR)", fontsize=11)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                      markersize=10, label=l) for l, c in cat_colors.items()]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9.5)
    ax.set_title("Functional enrichment of the 132-target consensus set "
                 "(STRING v12, Homo sapiens)", fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "FigS3_enriquecimiento_funcional.png", dpi=220,
                bbox_inches="tight")
    print("FigS3 written")


if __name__ == "__main__":
    main()
