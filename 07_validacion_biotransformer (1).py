#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_validacion_biotransformer.py
===============================
Actividad complementaria 1 — Validación retrospectiva de la predicción de
metabolitos con BioTransformer 3.0 y correlación con el índice Det.

Entradas (rutas relativas configurables abajo):
  - results_csv/met_<id>.csv      Salidas de BioTransformer 3.0 en modo
                                  monomolecular (-k pred -b allHuman -s 1),
                                  una por compuesto. Para los 7 compuestos
                                  cuyas estructuras fueron corregidas
                                  (testosterona, epitestosterona, danazol,
                                  gestrinona, noretandrolona, tibolona,
                                  etilestrenol) se usaron los SMILES
                                  verificados en PubChem.
  - results_csv2/met2_<id>.csv    Corridas a 2 pasos (-s 2) para los 12
                                  compuestos del conjunto de validación.
  - results_sdf/chunk*.sdf        Salidas del modo multiThread con base de
                                  datos MetXBioDB activada (useDatabase=true).
  - inputFolder/aasbatch.csv      Lista maestra: ID,Nombre,SMILES,...
  - inputFolder/inputSmilesBatch_*.csv  Lotes del modo multiThread (para
                                  mapear cada SDF a su compuesto).
  - indices_clases.csv            Índices de consenso (Det, EI, clase).

Salidas:
  - validacion_biotransformer.csv     Tabla de validación (24 metabolitos
                                      documentados x 12 compuestos).
  - metabolitos_predichos_det.csv     Recuento de metabolitos por compuesto
                                      junto a Det/EI/clase.
  - figura_validacion_metabolitos.png Figura resumen (3 paneles).

Criterios de procesado:
  * Se descartan artefactos de cosustrato con < 14 átomos de carbono.
  * Deduplicación por compuesto según InChIKey.
  * Fase II: reacción de tipo glucurono-/sulfo-/metil-/acetil-conjugación.
  * Coincidencia "exacta": InChIKey completo (27 caracteres) idéntico al de
    la referencia (PubChem/HMDB). Coincidencia "conectividad": solo el primer
    bloque (14 caracteres) coincide (estereoquímica no resuelta).

Uso:
  python3 07_validacion_biotransformer.py [dir_biotransformer] [dir_salida]
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BT_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '.'
IDX_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'indices_clases.csv')

PHASE2_RXN = re.compile(r'glycosyl|sulfat|methylat|acetyl|glutathion|conjugat', re.I)

# Compuestos cuyas salidas multiThread (SDF) se descartan por haberse
# generado con estructuras incorrectas en el archivo fuente original
# (se sustituyeron por corridas monomoleculares con SMILES de PubChem).
WRONG_SMI_IDS = {64, 32, 25, 37, 54, 66, 33}

# ---------------------------------------------------------------------------
# Conjunto de validación retrospectiva: 24 metabolitos marcadores documentados
# por WADA / literatura para 12 AAS bien estudiados.
# InChIKey verificados contra PubChem, HMDB, Wikidata y NCATS-GSRS.
# ---------------------------------------------------------------------------
VERIFIED = {
    ('Testosterone', 'Androsterone'): 'QGXBDMJGAMFCBF-HLUDHZFRSA-N',
    ('Testosterone', 'Etiocholanolone'): 'QGXBDMJGAMFCBF-BNSUEQOYSA-N',
    ('Testosterone', '6β-Hydroxytestosterone'): 'XSEGWEUVSZRCBC-ZVBLRVHNSA-N',
    ('Testosterone', 'Testosterone glucuronide'): 'NIKZPECGCSUSBV-HMAFJQTKSA-N',
    ('Testosterone', 'Androstenedione'): 'AEMFNILZOJDQLW-QAGGRKNESA-N',
    ('Nandrolone', '19-Norandrosterone'): 'UOUIARGWRPHDBX-CQZDKXCPSA-N',
    ('Nandrolone', '19-Noretiocholanolone'): 'UOUIARGWRPHDBX-DHMVHTBWSA-N',
    ('Stanozolol', "3'-Hydroxystanozolol"): 'SWPAIUOYLTYQKK-YEZTZDHTSA-N',
    ('Stanozolol', '4β-Hydroxystanozolol'): 'OCUSYXNRARMJHS-SUVJOWDWSA-N',
    ('Stanozolol', '16β-Hydroxystanozolol'): 'IZGBPAAEPVNBGA-BWPSUJIGSA-N',
    ('Methandienone', '6β-Hydroxymethandienone'): 'OBCJFTMGLMNCTJ-INIPNLRTSA-N',
    ('Methandienone', '17-Epimethandienone'): 'XWALNWXLMVGSFR-MPRNQXESSA-N',
    ('Boldenone', 'Androsta-1,4-diene-3,17-dione (boldione)'): 'LUJVUUWNAPIQQI-QAGGRKNESA-N',
    ('Trenbolone', 'Epitrenbolone'): 'MEHHPFQKXOUFFV-XDNAFOTISA-N',
    ('Trenbolone', 'Trendione'): 'KBSXJBBFQODDTQ-RYRKJORJSA-N',
    ('Oxandrolone', '17-Epioxandrolone'): 'QSLJIVKCVHQPLV-IWLBWFBKSA-N',
    ('Mesterolone', '1α-Methyl-androsterone'): 'UBSQZFBBZMBPFF-XHSSWRBBNA-N',
    ('Prasterone (DHEA)', 'Androsterone'): 'QGXBDMJGAMFCBF-HLUDHZFRSA-N',
    ('Prasterone (DHEA)', 'Etiocholanolone'): 'QGXBDMJGAMFCBF-BNSUEQOYSA-N',
    ('Prasterone (DHEA)', '5-Androstenediol'): 'QADHLRWLCPCEKT-LOVVWNRFSA-N',
    ('Prasterone (DHEA)', 'DHEA sulfate'): 'CZWCKYRVOZZJNM-USOAJAOKSA-N',
    ('Methyltestosterone', '17α-Methyl-5α-androstan-3α,17β-diol'): 'QGKQXZFZOIQFBI-XSWYFRFISA-N',
    ('Metenolone', '3α-Hydroxy-1-methylene-5α-androstan-17-one'): 'YSEVFKWFUGTGAQ-MCMLKMLFNA-N',
    ('Fluoxymesterone', '6β-Hydroxyfluoxymesterone'): 'SAMKMXPGYNTUIZ-SIZAIIQISA-N',
}
PARENT_ID = {'Testosterone': 64, 'Nandrolone': 51, 'Stanozolol': 62,
             'Methandienone': 40, 'Boldenone': 21, 'Trenbolone': 67,
             'Oxandrolone': 56, 'Mesterolone': 39, 'Prasterone (DHEA)': 59,
             'Methyltestosterone': 48, 'Metenolone': 41, 'Fluoxymesterone': 34}

# Anotación mecanística de los metabolitos no recuperados
MULTISTEP = {'Androsterone', 'Etiocholanolone', '19-Norandrosterone',
             '19-Noretiocholanolone', '1α-Methyl-androsterone',
             '3α-Hydroxy-1-methylene-5α-androstan-17-one',
             '17α-Methyl-5α-androstan-3α,17β-diol', '17-Epimethandienone',
             '17-Epioxandrolone', 'Epitrenbolone'}
CYP_ONLY = {'6β-Hydroxytestosterone', "3'-Hydroxystanozolol",
            '4β-Hydroxystanozolol', '16β-Hydroxystanozolol',
            '6β-Hydroxymethandienone', '6β-Hydroxyfluoxymesterone'}


def carbon_count(formula):
    m = re.search(r'C(\d*)', str(formula))
    return (int(m.group(1)) if m.group(1) else 1) if m else 0


def dedup(records):
    seen, out = set(), []
    for m in records:
        if m['inchikey'] not in seen:
            seen.add(m['inchikey'])
            out.append(m)
    return out


def load_bt_csv(path):
    """Carga una salida CSV de BioTransformer (modo monomolecular)."""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, on_bad_lines='skip')
    out = []
    for _, r in df.iterrows():
        ik = str(r.get('InChIKey', ''))
        if not ik or ik == 'nan':
            continue
        if carbon_count(r.get('Molecular formula', '')) < 14:
            continue  # artefactos de cosustrato
        rxn = str(r.get('Reaction', ''))
        out.append({'inchikey': ik, 'conn': ik.split('-')[0],
                    'reaction': rxn,
                    'phase2': bool(PHASE2_RXN.search(rxn))})
    return out


def parse_sdf(path):
    rows = []
    for mol in open(path, errors='ignore').read().split('$$$$'):
        if not mol.strip():
            continue
        d = {mm.group(1): mm.group(2).strip()
             for mm in re.finditer(r'> <([^>]+)>\r?\n([^\n]*)', mol)}
        if d:
            rows.append(d)
    return rows


def load_names():
    names = {}
    with open(os.path.join(BT_DIR, 'inputFolder', 'aasbatch.csv'),
              errors='ignore') as fh:
        for line in fh:
            p = line.strip().split(',')
            if len(p) >= 2 and p[0].strip().isdigit():
                names[int(p[0])] = p[1].strip()
    return names


def load_sdf_sets(names):
    """Mapea cada SDF del modo multiThread a su compuesto de origen."""
    name2id = {v: k for k, v in names.items()}
    chunk_row2cid = {}
    for cf in glob.glob(os.path.join(BT_DIR, 'inputFolder',
                                     'inputSmilesBatch_*.csv')):
        chunk = re.search(r'_(\d+)\.csv', cf).group(1)
        for line in open(cf, errors='ignore'):
            p = line.strip().split(',')
            if len(p) >= 2 and p[0].isdigit():
                cid = name2id.get(p[1])
                if cid is not None:
                    chunk_row2cid[(chunk, p[0])] = cid
    sdf_met = {cid: [] for cid in names}
    for sf in glob.glob(os.path.join(BT_DIR, 'results_sdf', '*.sdf')):
        m = re.search(r'chunk(\d+)_molouput_(\d+)\.sdf', sf)
        if not m:
            continue
        cid = chunk_row2cid.get((m.group(1), m.group(2)))
        if cid is None or cid in WRONG_SMI_IDS:
            continue
        for d in parse_sdf(sf):
            ik = d.get('InChIKey', '')
            if ik and carbon_count(d.get('Molecular_formula', '')) >= 14:
                rxn = d.get('Reaction', '')
                sdf_met[cid].append({'inchikey': ik, 'conn': ik.split('-')[0],
                                     'reaction': rxn,
                                     'phase2': bool(PHASE2_RXN.search(rxn))})
    return sdf_met


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = load_names()
    sdf_met = load_sdf_sets(names)

    # --- Conjunto combinado (validación): s1 + s2 + SDF ---------------------
    combined, uniform = {}, {}
    for cid in names:
        s1 = load_bt_csv(os.path.join(BT_DIR, 'results_csv',
                                      f'met_{cid}.csv'))
        s2 = load_bt_csv(os.path.join(BT_DIR, 'results_csv2',
                                      f'met2_{cid}.csv'))
        sdf = sdf_met[cid] if cid not in WRONG_SMI_IDS else []
        combined[cid] = dedup(s1 + s2 + sdf)
        uniform[cid] = dedup(s1 + sdf)  # profundidad 1, todos los compuestos

    # --- Tabla de validación retrospectiva ---------------------------------
    rows = []
    for (parent, met), ik in VERIFIED.items():
        preds = combined[PARENT_ID[parent]]
        iks = {p['inchikey'] for p in preds}
        conns = {p['conn'] for p in preds}
        if ik in iks:
            match = 'Exacta'
        elif ik.split('-')[0] in conns:
            match = 'Conectividad'
        else:
            match = 'No recuperado'
        lim = ''
        if match == 'No recuperado':
            if met in MULTISTEP:
                lim = '≥2 pasos metabólicos'
            elif met in CYP_ONLY:
                lim = 'CYP (módulo sin cobertura para esteroides)'
        rows.append({'Compuesto': parent,
                     'Metabolito documentado (WADA)': met,
                     'InChIKey de referencia': ik,
                     'Coincidencia': match,
                     'Limitación probable': lim})
    val = pd.DataFrame(rows)
    val.to_csv(os.path.join(OUT_DIR, 'validacion_biotransformer.csv'),
               index=False)
    print(val['Coincidencia'].value_counts())

    # --- Recuento uniforme + correlación con Det/EI --------------------------
    ucounts = pd.DataFrame({
        'id': list(names),
        'compuesto': [names[c] for c in names],
        'n_metabolitos': [len(uniform[c]) for c in names],
        'n_fase2': [sum(p['phase2'] for p in uniform[c]) for c in names]})
    ucounts['n_fase1'] = ucounts['n_metabolitos'] - ucounts['n_fase2']

    idx = pd.read_csv(IDX_CSV).rename(columns={'Unnamed: 0': 'compuesto'})
    clean = lambda s: re.sub(r'\s*\(.*$', '', str(s)).strip()
    idx['key'] = idx.compuesto.map(clean)
    ucounts['key'] = ucounts.compuesto.map(clean)
    ucounts.loc[ucounts.key == 'Androst-4-ene-3',
                'key'] = 'Androst-4-ene-3,11,17-trione'
    m = ucounts.merge(idx.drop(columns=['compuesto']), on='key', how='left')
    m[['id', 'compuesto', 'n_metabolitos', 'n_fase1', 'n_fase2',
       'Det', 'EI', 'Class']].to_csv(
        os.path.join(OUT_DIR, 'metabolitos_predichos_det.csv'), index=False)

    d = m.dropna(subset=['Det'])
    rho_d, p_d = stats.spearmanr(d['n_metabolitos'], d['Det'])
    rho_e, p_e = stats.spearmanr(d['n_metabolitos'], d['EI'])
    print(f'n={len(d)} | Det: rho={rho_d:.3f} (p={p_d:.4f}) | '
          f'EI: rho={rho_e:.3f} (p={p_e:.4f})')

    # --- Figura (journal double-column width ~9.5 in -> fonts >=6 pt) ----
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))

    ax = axes[0]
    cats = ['Exact', 'Connectivity', 'Not\nrecovered']
    vc = val['Coincidencia'].value_counts()
    vals = [vc.get('Exacta', 0), vc.get('Conectividad', 0),
            vc.get('No recuperado', 0)]
    bars = ax.bar(cats, vals, color=['#2ca02c', '#ffbf00', '#d62728'],
                  edgecolor='black', linewidth=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25,
                f'{v}\n({v / len(val) * 100:.0f}%)', ha='center', fontsize=9)
    ax.set_ylabel(f'Documented marker metabolites (n = {len(val)})')
    ax.set_title('A  Retrospective validation', loc='left', fontweight='bold')
    ax.set_ylim(0, max(vals) + 2.5)
    ax.spines[['top', 'right']].set_visible(False)

    cls_colors = {'I': '#d62728', 'II': '#1f77b4', 'III': '#2ca02c'}
    for ax, (xcol, rho, pval, xlabel, panel) in zip(
            axes[1:],
            [('Det', rho_d, p_d, 'Det index (consensus)',
              'B  Metabolite count vs Det'),
             ('EI', rho_e, p_e, 'EI index (consensus)',
              'C  Metabolite count vs EI')]):
        for cls, g in d.groupby('Class'):
            ax.scatter(g[xcol], g['n_metabolitos'], s=42, alpha=0.85,
                       color=cls_colors[cls], edgecolor='black',
                       linewidth=0.5, label=f'Class {cls}')
        z = np.polyfit(d[xcol], d['n_metabolitos'], 1)
        xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
        ax.plot(xs, np.polyval(z, xs), '--', color='gray', lw=1.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Predicted metabolites (BioTransformer 3.0)')
        ax.set_title(panel, loc='left', fontweight='bold')
        ptxt = f'p = {pval:.3f}' + (' (n.s.)' if pval >= 0.05 else '')
        ax.text(0.03, 0.97, f'Spearman ρ = {rho:.2f}\n{ptxt}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.35', fc='#f0f0f0',
                          ec='gray', lw=0.6))
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8.5, loc='upper right')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figura_validacion_metabolitos.png'),
                dpi=300, bbox_inches='tight')
    print('Figura guardada en', OUT_DIR)


if __name__ == '__main__':
    main()
