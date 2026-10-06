#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_validacion_externa_tox21.py
==============================

Validación externa de las predicciones de diana (receptor de andrógenos, AR)
frente a los datos experimentales de cribado de alto rendimiento de Tox21:

  - PubChem BioAssay AID 743053 (Tox21 AR agonist, summary)
  - PubChem BioAssay AID 743063 (Tox21 AR antagonist, summary)

Un agente se considera experimentalmente AR-activo si es "Active" en cualquiera
de los dos ensayos (definición de "binder" del consorcio CoMPARA; Mansouri et
al. 2020, Environ Health Perspect 128:027002; Kleinstreuer et al. 2017, Chem
Res Toxicol 30:946-964).

Entradas (mismo directorio):
  - smiles_68_corregidos.csv   (id, compuesto, smiles, smiles_corregido)
  - consensus_long.csv         (mol, uniprot, swiss, fgp, plato, pharmapper, n, consensus)

Salida:
  - external_validation_tox21.csv  y estadísticos por pantalla.

Notas:
  - Epi-dihydrotestosterone se corrige a su estereoisómero resuelto
    (5β-androstan-17β-ol-3-one, PubChem CID 11302), que no está en la librería
    Tox21; el SMILES depositado colapsa al registro de androstanolone.
  - Metandienone está cribada en Tox21 pero no forma parte de la matriz de
    consenso de 67 compuestos (sin puntuaciones).
"""

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

AID_AGONIST = 743053
AID_ANTAGONIST = 743063
AR_UNIPROT = "P10275"

# Corrección estereoquímica a priori (ver docstring)
EPIDHT_CID = 11302
EPIDHT_INCHIKEY = "NVKAWKQGWWIWPM-MISPCMORSA-N"


def pubchem_get(url, retries=5, timeout=60):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError("PubChem no respondió: " + url)


def smiles_to_cid_inchikey(smi):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
           + urllib.parse.quote(smi, safe="") + "/property/InChIKey/JSON")
    d = pubchem_get(url)["PropertyTable"]["Properties"][0]
    return d["CID"], d["InChIKey"]


def assay_cid_set(aid, ctype):
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/cids/JSON"
           f"?cids_type={ctype}")
    d = pubchem_get(url)
    return set(d["InformationList"]["Information"][0].get("CID", []))


def assay_concise(aid):
    """Tabla 'concise' del ensayo (incluye potencia AC50 en µM)."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/CSV"
    import io
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def main():
    sm = pd.read_csv("smiles_68_corregidos.csv")
    cl = pd.read_csv("consensus_long.csv")

    # 1) SMILES definitivo (corregido si existe) -> CID + InChIKey
    def pick(row):
        sc = str(row["smiles_corregido"])
        return sc if sc not in ("False", "nan", "") else row["smiles"]

    recs = []
    for _, row in sm.iterrows():
        cid, ikey = smiles_to_cid_inchikey(pick(row))
        recs.append({"compound": row["compuesto"], "pubchem_cid": cid,
                     "inchikey": ikey})
        time.sleep(0.25)
    ids = pd.DataFrame(recs)

    # corrección estereoquímica documentada
    m = ids["compound"] == "Epi-dihydrotestosterone"
    ids.loc[m, ["pubchem_cid", "inchikey"]] = [EPIDHT_CID, EPIDHT_INCHIKEY]

    # 2) Etiquetas experimentales Tox21 AR
    ag_a = assay_cid_set(AID_AGONIST, "active")
    ag_i = assay_cid_set(AID_AGONIST, "inactive")
    an_a = assay_cid_set(AID_ANTAGONIST, "active")
    an_i = assay_cid_set(AID_ANTAGONIST, "inactive")

    def call(cid, a, i):
        return "Active" if cid in a else ("Inactive" if cid in i else "NT")

    ids["tox21_ar_agonist_aid743053"] = ids["pubchem_cid"].apply(lambda c: call(c, ag_a, ag_i))
    ids["tox21_ar_antagonist_aid743063"] = ids["pubchem_cid"].apply(lambda c: call(c, an_a, an_i))
    tested = ids["tox21_ar_agonist_aid743053"].ne("NT") | ids["tox21_ar_antagonist_aid743063"].ne("NT")
    active = ids["pubchem_cid"].isin(ag_a | an_a)
    ids["tox21_ar_active"] = np.where(tested, active, np.nan)

    # 3) Potencia agonista (AC50 mediana de las filas activas)
    conc = assay_concise(AID_AGONIST)
    pot = (conc[(conc["CID"].isin(ids["pubchem_cid"])) &
                (conc["Activity Outcome"] == "Active")]
           .groupby("CID")["Activity Value [uM]"].median())
    ids["agonist_ac50_uM_median"] = ids["pubchem_cid"].map(pot)

    # 4) Unir con puntuaciones AR del consenso y rango de AR
    ar = cl[cl["uniprot"] == AR_UNIPROT].copy()
    ids["base"] = ids["compound"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
    ids = ids.merge(ar[["mol", "swiss", "consensus"]],
                    left_on="base", right_on="mol", how="left")

    cm_wide = (cl.pivot_table(index="mol", columns="uniprot",
                              values="consensus", aggfunc="first"))
    ranks = {}
    for mol, row in cm_wide.iterrows():
        if AR_UNIPROT in row and pd.notna(row[AR_UNIPROT]):
            ranks[mol] = int((row > row[AR_UNIPROT]).sum() + 1)
    ids["ar_rank_by_consensus"] = ids["mol"].map(ranks)

    ids = ids.rename(columns={"swiss": "swisstargetprediction_ar_prob",
                              "consensus": "consensus_ar_score"})
    out = ids[tested.values].drop(columns=["base", "mol"]).sort_values("compound")
    out.to_csv("external_validation_tox21.csv", index=False)

    # 5) Estadísticos del benchmark
    t = ids[tested.values].copy()
    tt = t.dropna(subset=["consensus_ar_score"])
    print(f"Cribados en Tox21 AR: {len(t)}/{len(ids)}; activos: {int(t['tox21_ar_active'].sum())}")
    print(f"Con predicción de consenso: {len(tt)}; rango {tt['consensus_ar_score'].min():.2f}-"
          f"{tt['consensus_ar_score'].max():.2f}; mediana {tt['consensus_ar_score'].median():.2f}")
    print(f"Rango mediano de AR: {tt['ar_rank_by_consensus'].median():.0f}; "
          f"top-5: {(tt['ar_rank_by_consensus'] <= 5).sum()}/{len(tt)}; "
          f"top-10: {(tt['ar_rank_by_consensus'] <= 10).sum()}/{len(tt)}")
    miss = tt[tt["swisstargetprediction_ar_prob"] < 0.5]
    print("SwissTP < 0.5 recuperados por consenso:", miss["compound"].tolist())
    tc = tt.dropna(subset=["agonist_ac50_uM_median"])
    rho, p = stats.spearmanr(tc["consensus_ar_score"],
                             -np.log10(tc["agonist_ac50_uM_median"] * 1e-6))
    print(f"Spearman consenso vs pAC50: rho={rho:.2f}, p={p:.2f} (n={len(tc)})")


if __name__ == "__main__":
    main()
