#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_pipeline_consenso.py
=======================
Consensus target-fishing pipeline for 67 anabolic-androgenic steroids (AAS).

Supporting Information for:
"Consensus target fishing and ADMET profiling of anabolic-androgenic steroids"
(manuscript submitted to Forensic Science International).

Implements Sections 2.2-2.7 of the manuscript:
  * Parsing of the four target-fishing platforms (SwissTargetPrediction, FGP,
    PLATO, PHARMAPPER) from the raw workbook.
  * Molecule-name canonicalisation (aliases, Greek-letter lookalikes,
    whitespace artefacts of the platform exports).
  * UniProt ID as the unified target key (target names of PLATO/PHARMAPPER
    are mapped through the SwissTargetPrediction nomenclature, exact matching
    after punctuation-insensitive normalisation).
  * Per-platform score normalisation and weighted consensus.
  * Detectability Index, Endocrine Impact Index and four-class risk
    classification.

Input : HORMONES_ANABOLIC_ALL_TARGETS.xlsx  (one sheet per platform)
Output: consensus_matrix.csv      (67 compounds x 573 targets, weighted consensus)
        consensus_long.csv        (molecule, UniProt, per-platform scores, n, consensus)
        indices_clases.csv        (Det, EI, Det_swiss, discrepancy, risk class)

Validation checkpoints reproduced by this script (Section 3.2-3.3):
  n platforms per interaction: {1: 3737, 2: 2196, 3: 579, 4: 188}
  Trenbolone       Det 0.235 (Swiss-only 0.096)
  Metribolone      Det 0.202 (Swiss-only 0.065)
  Methylclostebol  Det 0.226 (Swiss-only 0.069)
  7-Keto-DHEA      Det 0.294 (Swiss-only 0.117)
  Det range 0.064-0.504, mean 0.242 +/- 0.091; EI mean 0.716 +/- 0.147
  Risk classes: I = 8, II = 49, III = 10, IV = 0

Requires: pandas, numpy, openpyxl.  Python >= 3.11.
"""
import os
import re
import sys
import numpy as np
import pandas as pd

INPUT_XLSX = sys.argv[1] if len(sys.argv) > 1 else "HORMONES_ANABOLIC_ALL_TARGETS.xlsx"
INPUT_XLSX = sys.argv[1] if len(sys.argv) > 1 else "HORMONES_ANABOLIC_ALL_TARGETS.xlsx"

if not os.path.exists(INPUT_XLSX):
    raise FileNotFoundError(
        f"No se encontró el archivo: {INPUT_XLSX}\n"
        f"Verifica que esté en la carpeta correcta o pasa la ruta completa.\n"
        f"Ejemplo:\npython 01_pipeline_consenso.py \"C:\\Users\\Mateo\\Downloads\\WADA-S1-target-fishing-ADMET-main\\HORMONES_ANABOLIC_ALL_TARGETS.xlsx\""
    )

# ---------------------------------------------------------------------------
# 1. Molecule-name canonicalisation
# ---------------------------------------------------------------------------
# The platform exports contain: trailing/duplicate whitespace, Latin
# lookalikes of Greek letters (sharp-s 'ß' for beta, script-a 'ɑ' for alpha)
# and platform-specific synonyms.  All names are mapped to a single canonical
# form (proper Greek letters, as used in the manuscript).

GREEK_FIX = {"ß": "β", "ɑ": "α"}          # Latin lookalikes -> Greek letters

CANON = {  # lower-cased key -> canonical display name
    "17-epi-dihydrotestosterone": "Epi-dihydrotestosterone",
    "epi-dihydrotestosterone": "Epi-dihydrotestosterone",
    "17α-methylepithiostanol": "17α-Methylepithiostanol",
    "17α-methylepithiostanol (epistane)": "17α-Methylepithiostanol",
    "19-nortestosterone": "Nandrolone",
    "nandrolone": "Nandrolone",
    "androst-4-ene-3,11,17-trione": "Androst-4-ene-3,11,17-trione",
    "androst-4-ene-3,11,17- trione": "Androst-4-ene-3,11,17-trione",
}


def norm_mol(name):
    """Return the canonical molecule name."""
    s = re.sub(r"\s+", " ", str(name)).strip()
    for a, b in GREEK_FIX.items():
        s = s.replace(a, b)
    return CANON.get(s.lower(), s)


def norm_target(name):
    """Normalise a target name for exact cross-platform matching.

    Lowercase, strip, collapse whitespace and neutralise punctuation
    (hyphens, commas, brackets) so that, e.g., PLATO 'Matrix
    metalloproteinase-2' matches SwissTargetPrediction 'Matrix
    metalloproteinase 2'.  Matching is strictly exact after this
    normalisation (no fuzzy matching).
    """
    s = re.sub(r"\s+", " ", str(name)).strip().lower()
    s = re.sub(r"[-,\[\]]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# 2. Parse the four platform sheets
# ---------------------------------------------------------------------------
def load_platforms(path):
    xls = pd.ExcelFile(path)

    # SwissTargetPrediction: header on row 4 (index 3)
    sw = pd.read_excel(xls, "SWISS_TARGET_PREDICTION", header=3).dropna(how="all", axis=1)
    sw.columns = ["Molecule", "Target", "Common", "Uniprot", "ChEMBL",
                  "Class", "Prob", "Actives"]
    sw = sw.dropna(subset=["Molecule", "Uniprot"])

    # FGP: header on row 2 (index 1); five algorithms with "class (prob)" cells
    fg = pd.read_excel(xls, "FGP", header=1).dropna(how="all", axis=1)
    fg.columns = ["Molecule", "TID", "Uniprot", "DTREE", "GM", "KNN", "RF", "SVM"]
    fg = fg.dropna(subset=["Molecule", "Uniprot"])

    # PLATO: header on row 5 (index 4); several organism rows per target
    pl = pd.read_excel(xls, "PLATO", header=4).dropna(how="all", axis=1)
    pl.columns = ["Molecule", "Target", "Organism", "Score", "Reliable"]
    pl = pl.dropna(subset=["Molecule", "Target"])

    # PHARMAPPER: header on row 5 (index 4)
    ph = pd.read_excel(xls, "PHARMAPPER", header=4).dropna(how="all", axis=1)
    ph.columns = ["Molecule", "Rank", "Target", "Nfeat", "Fit", "NormFit", "z"]
    ph = ph.dropna(subset=["Molecule", "Target"])

    for df in (sw, fg, pl, ph):
        df["mol"] = df["Molecule"].map(norm_mol)
    return sw, fg, pl, ph


# ---------------------------------------------------------------------------
# 3. Per-platform normalised scores, keyed by (molecule, UniProt)
# ---------------------------------------------------------------------------
FGP_ALGOS = ["DTREE", "GM", "KNN", "RF", "SVM"]


def fgp_prob(cell):
    """Extract the probability from an FGP 'class (prob)' cell, e.g. '1 (0.93)'."""
    m = re.match(r"\s*\d\s*\(([\d.]+)\)", str(cell))
    return float(m.group(1)) if m else np.nan


def build_scores(sw, fg, pl, ph):
    # SwissTargetPrediction: probability in [0, 1], used directly
    sw["swiss"] = sw["Prob"].clip(0, 1).astype(float)
    sw_s = sw.set_index(["mol", "Uniprot"])["swiss"]

    # FGP: ensemble score = mean probability of the five algorithms
    fg["fgp"] = fg[FGP_ALGOS].map(fgp_prob).mean(axis=1)
    fg_s = fg.set_index(["mol", "Uniprot"])["fgp"]

    # PLATO: min-max scale bioactivity to [0, 1] over the dataset range,
    # keep the maximum across organism rows per (molecule, target)
    pl["plato_raw"] = pd.to_numeric(pl["Score"], errors="coerce")
    pl["plato"] = ((pl["plato_raw"] - pl["plato_raw"].min())
                   / (pl["plato_raw"].max() - pl["plato_raw"].min()))

    # PHARMAPPER: sigmoid transform of the z'-score
    ph["pharmapper"] = 1.0 / (1.0 + np.exp(-pd.to_numeric(ph["z"], errors="coerce")))

    # UniProt mapping for PLATO / PHARMAPPER through the Swiss nomenclature
    name2unip = {}
    for _, r in sw.iterrows():
        name2unip.setdefault(norm_target(r["Target"]), str(r["Uniprot"]).strip())
    pl["uniprot"] = pl["Target"].map(lambda t: name2unip.get(norm_target(t)))
    ph["uniprot"] = ph["Target"].map(lambda t: name2unip.get(norm_target(t)))

    pl_s = pl.dropna(subset=["uniprot"]).groupby(["mol", "uniprot"])["plato"].max()
    ph_s = ph.dropna(subset=["uniprot"]).groupby(["mol", "uniprot"])["pharmapper"].max()
    return sw_s, fg_s, pl_s, ph_s


# ---------------------------------------------------------------------------
# 4. Weighted consensus (Section 2.3)
# ---------------------------------------------------------------------------
WEIGHTS = {"swiss": 0.20, "fgp": 0.35, "plato": 0.30, "pharmapper": 0.15}


def consensus_table(sw_s, fg_s, pl_s, ph_s):
    """Long table over SwissTargetPrediction (molecule, UniProt) pairs.

    n = number of contributing platforms; weights are rescaled over the
    available subset (Eq. in Section 2.3).
    """
    c = pd.DataFrame(index=sw_s.index)
    c["swiss"] = sw_s
    c["fgp"] = fg_s.reindex(c.index)
    c["plato"] = pl_s.reindex(c.index)
    c["pharmapper"] = ph_s.reindex(c.index)
    c["n"] = c[list(WEIGHTS)].notna().sum(axis=1)

    def wmean(row):
        num = sum(row[k] * w for k, w in WEIGHTS.items() if pd.notna(row[k]))
        den = sum(w for k, w in WEIGHTS.items() if pd.notna(row[k]))
        return num / den

    c["consensus"] = c.apply(wmean, axis=1)
    c.index = c.index.set_names(["mol", "uniprot"])
    return c


# ---------------------------------------------------------------------------
# 5. Detectability and Endocrine Impact indices (Sections 2.4-2.5)
# ---------------------------------------------------------------------------
UNIPROT = {
    "CYP3A4": "P08684", "CYP2C9": "P11712", "CYP2C19": "P33261",
    "CYP2D6": "P10635", "CYP17A1": "P05093", "CYP19A1": "P11511",
    "CYP11B1": "P15538", "CYP11B2": "P19099", "CYP51A1": "Q16850",
    "SRD5A1": "P18405", "SRD5A2": "P31213", "UGT2B7": "P16662",
    "HSD17B1": "P14061", "HSD17B2": "P37059", "HSD17B3": "P37058",
    "AR": "P10275", "SHBG": "P04278",
}
CYP_PANEL = ["CYP3A4", "CYP2C9", "CYP2C19", "CYP2D6", "CYP17A1",
             "CYP19A1", "CYP11B1", "CYP11B2", "CYP51A1"]   # 9 isoforms


def _components(scores):
    """Pathway components from a (mol, uniprot) -> score Series.

    Missing interactions count as 0 (no predicted metabolism):
    CYP     = mean over the 9 CYP isoforms (missing isoforms = 0)
    SRD5A   = max(SRD5A1, SRD5A2)
    UGT2B7  = consensus of UDP-glucuronosyltransferase 2B7
    CYP19A1 = consensus of aromatase
    HSD17B  = mean over HSD17B1/2/3 (missing isoforms = 0)
    """
    def val(gene):
        return scores.get(UNIPROT[gene], np.nan)

    cyp = np.nanmean([val(g) if pd.notna(val(g)) else 0.0 for g in CYP_PANEL])
    srd = np.nanmax([val("SRD5A1") if pd.notna(val("SRD5A1")) else 0.0,
                     val("SRD5A2") if pd.notna(val("SRD5A2")) else 0.0])
    hsd = np.nanmean([val(g) if pd.notna(val(g)) else 0.0
                      for g in ("HSD17B1", "HSD17B2", "HSD17B3")])
    ugt = val("UGT2B7"); ugt = ugt if pd.notna(ugt) else 0.0
    aro = val("CYP19A1"); aro = aro if pd.notna(aro) else 0.0
    ar = val("AR");       ar = ar if pd.notna(ar) else 0.0
    shbg = val("SHBG");   shbg = shbg if pd.notna(shbg) else 0.0
    return dict(CYP=cyp, SRD5A=srd, UGT2B7=ugt, CYP19A1=aro,
                HSD17B=hsd, AR=ar, SHBG=shbg)


def detectability(c):
    return (0.25 * c["CYP"] + 0.30 * c["SRD5A"] + 0.20 * c["UGT2B7"]
            + 0.15 * c["CYP19A1"] + 0.10 * c["HSD17B"])


def endocrine_impact(c):
    return 0.5 * c["AR"] + 0.3 * c["SHBG"] + 0.2 * c["CYP19A1"]


def risk_class(det, ei):
    """Section 2.7: four-class scheme from the two indices."""
    if ei >= 0.30:
        if det >= 0.35:
            return "I"     # high detectability, high endocrine impact
        if det >= 0.15:
            return "II"    # moderate risk
        return "III"       # stealth, endocrine-positive
    if det < 0.15:
        return "IV"        # stealth, endocrine-negative
    return np.nan


def build_indices(cons):
    mols = sorted(cons.index.get_level_values("mol").unique())
    rows = {}
    for mol in mols:
        s_cons = cons["consensus"].xs(mol, level="mol")
        s_swis = cons["swiss"].xs(mol, level="mol")
        cc = _components(s_cons)
        cs = _components(s_swis)
        det, det_sw = detectability(cc), detectability(cs)
        ei = endocrine_impact(cc)
        rows[mol] = dict(**{f"cons_{k}": v for k, v in cc.items()},
                         Det=det, EI=ei, Det_swiss=det_sw,
                         diff=det - det_sw, Class=risk_class(det, ei))
    return pd.DataFrame(rows).T.astype(float, errors="ignore")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    sw, fg, pl, ph = load_platforms(INPUT_XLSX)
    sw_s, fg_s, pl_s, ph_s = build_scores(sw, fg, pl, ph)
    cons = consensus_table(sw_s, fg_s, pl_s, ph_s)
    idx = build_indices(cons)

    # ---- outputs ----
    matrix = cons["consensus"].unstack("uniprot")          # 67 x 573
    matrix.to_csv("consensus_matrix.csv")
    cons.reset_index().to_csv("consensus_long.csv", index=False)
    idx.to_csv("indices_clases.csv")

    # UniProt -> gene-symbol map (used by the figure scripts)
    genes = (sw.assign(Uniprot=sw["Uniprot"].astype(str).str.strip())
               .drop_duplicates("Uniprot")
               .set_index("Uniprot")["Common"])
    genes.to_csv("uniprot_genes.csv")

    # ---- validation report ----
    print("Interactions per number of platforms:",
          cons["n"].value_counts().sort_index().to_dict())
    print(f"Consensus matrix: {matrix.shape[0]} compounds x {matrix.shape[1]} targets")
    print(f"Det  range {idx['Det'].min():.3f}-{idx['Det'].max():.3f}, "
          f"mean {idx['Det'].mean():.3f} +/- {idx['Det'].std():.3f}")
    print(f"EI   mean {idx['EI'].mean():.3f} +/- {idx['EI'].std():.3f}")
    print("Risk classes:", idx["Class"].value_counts().to_dict())
    for m in ["Trenbolone", "Metribolone", "Methylclostebol", "7-Keto-DHEA"]:
        print(f"  {m:16s} Det = {idx.loc[m, 'Det']:.3f} "
              f"(Swiss-only {idx.loc[m, 'Det_swiss']:.3f})")


if __name__ == "__main__":
    main()
