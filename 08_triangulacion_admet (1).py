#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_triangulacion_admet.py
=========================
Actividad complementaria 4 — Triangulación inter-plataforma de endpoints
toxicológicos clave (ADMETlab 3.0 vs admetSAR 3.0 vs pkCSM).

Entradas:
  - admet_decoded_corregido.csv   Predicciones ADMETlab 3.0 (probabilidades),
                                  con las 7 filas de estructuras erróneas en el
                                  archivo fuente original (testosterona,
                                  epitestosterona, danazol, gestrinona,
                                  noretandrolona, tibolona, etilestrenol)
                                  regeneradas mediante la API oficial de
                                  ADMETlab 3.0 (/api/single/admet) usando los
                                  SMILES verificados en PubChem.
  - admetsar_out/admetsar_XX.txt   TSV de 121 endpoints de admetSAR 3.0
                                   (probabilidades; binarizado a 0.5).
  - pkcsm_out/pkcsm_XX.html  Páginas de resultados de pkCSM (modo toxicity;
                             llamadas categóricas Yes/No).
  - smiles_68_corregidos.csv Lista maestra id/compuesto/SMILES (con las 7
                             estructuras corregidas vía PubChem).

Endpoints armonizados:
  DILI/hepatotoxicidad : ADMETlab DILI | admetSAR DILI | pkCSM Hepatotoxicity
  hERG                 : ADMETlab hERG-10um | admetSAR hERG_10uM |
                         pkCSM hERG I / hERG II
  BSEP                 : ADMETlab BSEP | admetSAR BSEP_inhibitor
  Ames                 : ADMETlab Ames | admetSAR Ames | pkCSM AMES toxicity
  Carcinogenicidad     : ADMETlab Carcinogenicity | admetSAR
                         Rodents_carcinogenicity

Nota: el análisis se realiza con n = 68 y estructuras ya corregidas
(ver admetlab3_correccion_7compuestos.csv para el detalle antes/después).

Salidas:
  - triangulacion_admet.csv        Tabla por compuesto (68 filas).
  - figura_triangulacion_admet.png Figura de 3 paneles.

Uso:
  python3 08_triangulacion_admet.py [dir_datos] [dir_salida]
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '.'
ADL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'admet_decoded_corregido.csv')

WRONG_SMI_IDS = {25, 32, 33, 37, 54, 64, 66}

PK_ENDPOINTS = ['AMES toxicity', 'Max. tolerated dose (human)',
                'hERG I inhibitor', 'hERG II inhibitor',
                'Oral Rat Acute Toxicity (LD50)',
                'Oral Rat Chronic Toxicity (LOAEL)', 'Hepatotoxicity',
                'Skin Sensitisation']


def clean(name):
    return re.sub(r'\s*\(.*$', '', str(name)).replace('*', '').strip().lower()


def parse_pkcsm(path):
    h = open(path, errors='ignore').read()
    d = {}
    for ep in PK_ENDPOINTS:
        m = re.search(re.escape(ep) +
                      r'</td>\s*<td><b>(?:<b>)?\s*([^<]+?)\s*(?:</b>)?\s*</td>',
                      h)
        d['pkcsm_' + ep] = m.group(1).strip() if m else None
    return d


def parse_admetsar(path):
    t = pd.read_csv(path, sep='\t')
    cols = ['DILI', 'BSEP_inhibitor', 'hERG_1uM', 'hERG_10uM', 'hERG_30uM',
            'Ames', 'Mouse_carcinogenicity', 'Rat_carcinogenicity',
            'Rodents_carcinogenicity']
    return {'as3_' + c: float(t.iloc[0][c]) for c in cols}


def fleiss_kappa(table):
    N, k = table.shape
    n = table.sum(axis=1)[0]
    p = table.sum(axis=0) / (N * n)
    Pbar = ((table ** 2).sum(axis=1) - n).sum() / (N * n * (n - 1))
    Pe = (p ** 2).sum()
    return (Pbar - Pe) / (1 - Pe)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    smi = pd.read_csv(os.path.join(DATA_DIR, 'smiles_68_corregidos.csv'))

    rows = []
    for _, r in smi.iterrows():
        cid = int(r.id)
        d = {'id': cid, 'compuesto': r.compuesto}
        d.update(parse_pkcsm(os.path.join(
            DATA_DIR, 'pkcsm_out', f'pkcsm_{cid:02d}.html')))
        d.update(parse_admetsar(os.path.join(
            DATA_DIR, 'admetsar_out', f'admetsar_{cid:02d}.txt')))
        rows.append(d)
    tox = pd.DataFrame(rows)
    tox['key'] = tox['compuesto'].map(clean)
    tox.loc[tox.key.str.contains(r'androst-4-ene-3$', na=False),
            'key'] = 'androst-4-ene-3,11,17-trione'

    adl = pd.read_csv(ADL_CSV)
    adl['key'] = adl['name'].map(clean)
    tox = tox.merge(adl[['key', 'BSEP', 'hERG', 'hERG-10um', 'DILI', 'Ames',
                         'Carcinogenicity', 'alkylated']], on='key')
    tox['estructura_validada'] = ~tox['id'].isin(WRONG_SMI_IDS)

    # Binarización (umbral 0.5 para probabilidades; Yes/No para pkCSM)
    for c in ['BSEP', 'hERG', 'hERG-10um', 'DILI', 'Ames', 'Carcinogenicity']:
        tox['adl_' + c + '_bin'] = (tox[c] >= 0.5).astype(int)
    for c in ['DILI', 'BSEP_inhibitor', 'hERG_10uM', 'Ames',
              'Rodents_carcinogenicity']:
        tox['as3_' + c + '_bin'] = (tox['as3_' + c] >= 0.5).astype(int)
    for c in ['AMES toxicity', 'hERG I inhibitor', 'hERG II inhibitor',
              'Hepatotoxicity']:
        tox['pk_' + c + '_bin'] = (tox['pkcsm_' + c] == 'Yes').astype(int)

    # Tabla final + consenso DILI
    out = pd.DataFrame({
        'id': tox.id, 'compuesto': tox.compuesto,
        'estructura_validada': tox.estructura_validada,
        'ADMETlab3_DILI': tox.adl_DILI_bin,
        'admetSAR3_DILI_prob': tox.as3_DILI.round(3),
        'admetSAR3_DILI': tox.as3_DILI_bin,
        'pkCSM_hepatotox': tox['pk_Hepatotoxicity_bin'],
        'ADMETlab3_hERG10uM': tox['adl_hERG-10um_bin'],
        'admetSAR3_hERG10uM_prob': tox.as3_hERG_10uM.round(3),
        'admetSAR3_hERG10uM': tox.as3_hERG_10uM_bin,
        'pkCSM_hERG_I': tox['pk_hERG I inhibitor_bin'],
        'pkCSM_hERG_II': tox['pk_hERG II inhibitor_bin'],
        'ADMETlab3_BSEP': tox.adl_BSEP_bin,
        'admetSAR3_BSEP_prob': tox.as3_BSEP_inhibitor.round(3),
        'admetSAR3_BSEP': tox.as3_BSEP_inhibitor_bin,
        'ADMETlab3_Ames': tox.adl_Ames_bin,
        'admetSAR3_Ames_prob': tox.as3_Ames.round(3),
        'admetSAR3_Ames': tox.as3_Ames_bin,
        'pkCSM_Ames': tox['pk_AMES toxicity_bin'],
        'ADMETlab3_carcino': tox.adl_Carcinogenicity_bin,
        'admetSAR3_carcino_prob': tox.as3_Rodents_carcinogenicity.round(3),
        'admetSAR3_carcino': tox.as3_Rodents_carcinogenicity_bin,
    })
    out['DILI_consenso_n'] = out[['ADMETlab3_DILI', 'admetSAR3_DILI',
                                  'pkCSM_hepatotox']].sum(axis=1)
    out['Ames_consenso_n'] = out[['ADMETlab3_Ames', 'admetSAR3_Ames',
                                  'pkCSM_Ames']].sum(axis=1)
    out.to_csv(os.path.join(OUT_DIR, 'triangulacion_admet.csv'), index=False)

    # ---- Concordancia (n = 68, estructuras corregidas) ---------------------
    ok = tox.copy()
    print(f'n = {len(ok)}')

    def conc(cols, label):
        sub = ok[cols].astype(int)
        agree = (sub.nunique(axis=1) == 1).mean()
        ks = {}
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = sub[cols[i]], sub[cols[j]]
                ks[f'{cols[i]}~{cols[j]}'] = (
                    round(cohen_kappa_score(a, b), 3)
                    if a.nunique() > 1 and b.nunique() > 1 else None)
        fk = None
        if len(cols) >= 3:
            tab = np.column_stack([(sub == 0).sum(axis=1),
                                   (sub == 1).sum(axis=1)])
            fk = round(fleiss_kappa(tab), 3)
        print(f'{label}: acuerdo={agree:.3f} Fleiss={fk} kappas={ks}')

    conc(['adl_DILI_bin', 'as3_DILI_bin', 'pk_Hepatotoxicity_bin'], 'DILI')
    conc(['adl_Ames_bin', 'as3_Ames_bin', 'pk_AMES toxicity_bin'], 'Ames')
    conc(['adl_hERG-10um_bin', 'as3_hERG_10uM_bin',
          'pk_hERG I inhibitor_bin'], 'hERG-I')
    conc(['adl_hERG-10um_bin', 'as3_hERG_10uM_bin',
          'pk_hERG II inhibitor_bin'], 'hERG-II')
    conc(['adl_BSEP_bin', 'as3_BSEP_inhibitor_bin'], 'BSEP')
    conc(['adl_Carcinogenicity_bin', 'as3_Rodents_carcinogenicity_bin'],
         'Carcino')

    # Correlaciones continuas de probabilidades
    for a, b in [('DILI', 'as3_DILI'), ('hERG-10um', 'as3_hERG_10uM'),
                 ('BSEP', 'as3_BSEP_inhibitor'), ('Ames', 'as3_Ames'),
                 ('Carcinogenicity', 'as3_Rodents_carcinogenicity')]:
        rho, p = stats.spearmanr(ok[a], ok[b])
        print(f'Spearman {a}: rho={rho:+.3f} p={p:.4f}')

    # Enriquecimiento de 17α-alquilación
    for col in ['adl_DILI_bin', 'as3_DILI_bin', 'pk_Hepatotoxicity_bin']:
        tab = pd.crosstab(ok[col], ok['alkylated'])
        odds, p = fisher_exact(tab)
        print(f'{col}: OR={odds:.2f}, p={p:.4f}')

    # ---- Figura ------------------------------------------------------------
    endpoints = ['DILI /\nhepatotox.', 'hERG\n(10 µM)', 'BSEP', 'Ames',
                 'Carcino-\ngenicity']
    adl_r = [ok.adl_DILI_bin.mean(), ok['adl_hERG-10um_bin'].mean(),
             ok.adl_BSEP_bin.mean(), ok.adl_Ames_bin.mean(),
             ok.adl_Carcinogenicity_bin.mean()]
    as3_r = [ok.as3_DILI_bin.mean(), ok.as3_hERG_10uM_bin.mean(),
             ok.as3_BSEP_inhibitor_bin.mean(), ok.as3_Ames_bin.mean(),
             ok.as3_Rodents_carcinogenicity_bin.mean()]
    pk_r = [ok['pk_Hepatotoxicity_bin'].mean(),
            ok['pk_hERG I inhibitor_bin'].mean(), np.nan,
            ok['pk_AMES toxicity_bin'].mean(), np.nan]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
    ax = axes[0]
    x = np.arange(len(endpoints))
    w = 0.27
    ax.bar(x - w, np.array(adl_r) * 100, w, label='ADMETlab 3.0',
           color='#1f77b4', edgecolor='k', lw=0.5)
    ax.bar(x, np.array(as3_r) * 100, w, label='admetSAR 3.0',
           color='#ff7f0e', edgecolor='k', lw=0.5)
    ax.bar(x + w, np.array([0 if np.isnan(v) else v for v in pk_r]) * 100,
           w, label='pkCSM', color='#2ca02c', edgecolor='k', lw=0.5)
    for i, v in enumerate(pk_r):
        if np.isnan(v):
            ax.text(i + w, 2, 'n/a', ha='center', fontsize=8, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(['DILI', 'hERG', 'BSEP', 'Ames', 'Carc.'],
                       fontsize=8)
    ax.set_ylabel('Positive calls (%)')
    ax.set_title('A  Positive-call rates by platform', loc='left',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.14), ncol=3, handlelength=1.2,
              columnspacing=0.9)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]

    def _k(a, b):
        sa, sb = ok[a].astype(int), ok[b].astype(int)
        if sa.nunique() < 2 or sb.nunique() < 2:
            return np.nan
        return cohen_kappa_score(sa, sb)

    data_k = np.array([
        [_k('adl_DILI_bin', 'as3_DILI_bin'),
         _k('adl_DILI_bin', 'pk_Hepatotoxicity_bin'),
         _k('as3_DILI_bin', 'pk_Hepatotoxicity_bin')],
        [_k('adl_hERG-10um_bin', 'as3_hERG_10uM_bin'),
         _k('adl_hERG-10um_bin', 'pk_hERG I inhibitor_bin'),
         _k('as3_hERG_10uM_bin', 'pk_hERG I inhibitor_bin')],
        [_k('adl_hERG-10um_bin', 'as3_hERG_10uM_bin'),
         _k('adl_hERG-10um_bin', 'pk_hERG II inhibitor_bin'),
         _k('as3_hERG_10uM_bin', 'pk_hERG II inhibitor_bin')],
        [_k('adl_BSEP_bin', 'as3_BSEP_inhibitor_bin'), np.nan, np.nan],
        [_k('adl_Ames_bin', 'as3_Ames_bin'),
         _k('adl_Ames_bin', 'pk_AMES toxicity_bin'),
         _k('as3_Ames_bin', 'pk_AMES toxicity_bin')],
        [_k('adl_Carcinogenicity_bin', 'as3_Rodents_carcinogenicity_bin'),
         np.nan, np.nan]])
    row_lbl = ['DILI/hepatotox.', 'hERG (pkCSM I)', 'hERG (pkCSM II)',
               'BSEP', 'Ames', 'Carcinogenicity']
    pairs_lbl = ['ADL–\naSAR', 'ADL–\npkCSM', 'aSAR–\npkCSM']
    im = ax.imshow(np.ma.masked_invalid(data_k), cmap='RdYlGn',
                   vmin=-0.4, vmax=0.6, aspect='auto')
    ax.set_xticks(range(3))
    ax.set_xticklabels(pairs_lbl, fontsize=7.5)
    ax.set_yticks(range(6))
    ax.set_yticklabels(row_lbl, fontsize=7.5)
    for i in range(6):
        for j in range(3):
            v = data_k[i, j]
            ax.text(j, i, 'undef' if np.isnan(v) else f'{v:.2f}',
                    ha='center', va='center', fontsize=8)
    ax.set_title("B  Pairwise Cohen's κ", loc='left', fontweight='bold')
    cb = plt.colorbar(im, ax=ax, shrink=0.85, label="Cohen's κ")
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Cohen's κ", fontsize=8)

    ax = axes[2]
    plats = ['ADMETlab\n3.0', 'admetSAR\n3.0', 'pkCSM']
    dl, asd = ok.adl_DILI_bin, ok.as3_DILI_bin
    pkh = ok['pk_Hepatotoxicity_bin']
    pct = [ok[dl == 1]['alkylated'].mean() * 100,
           ok[asd == 1]['alkylated'].mean() * 100,
           ok[pkh == 1]['alkylated'].mean() * 100]
    ns = [f"{int(ok[dl == 1]['alkylated'].sum())}/{int(dl.sum())}",
          f"{int(ok[asd == 1]['alkylated'].sum())}/{int(asd.sum())}",
          f"{int(ok[pkh == 1]['alkylated'].sum())}/{int(pkh.sum())}"]
    ps = []
    for col in ['adl_DILI_bin', 'as3_DILI_bin', 'pk_Hepatotoxicity_bin']:
        _, _p = fisher_exact(pd.crosstab(ok[col], ok['alkylated']))
        ps.append(f'p = {_p:.3f}' + ('' if _p < 0.05 else ' (n.s.)'))
    bars = ax.bar(plats, pct, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  edgecolor='k', lw=0.6, width=0.55)
    for b, p_, n_ in zip(bars, ps, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f'{n_}\n{p_}', ha='center', fontsize=7.5)
    base = ok['alkylated'].mean() * 100
    ax.axhline(base, ls='--', c='gray', lw=1)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylabel('17α-alkylated among DILI+ (%)')
    ax.set_ylim(0, 118)
    ax.set_title('C  17α-alkylation enrichment (Fisher)', loc='left',
                 fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figura_triangulacion_admet.png'),
                dpi=300, bbox_inches='tight')
    print('Figura guardada en', OUT_DIR)


if __name__ == '__main__':
    main()
