# Consensus target fishing and ADMET profiling of anabolic-androgenic steroids

Code and data accompanying the manuscript:

> **Integrating multi-algorithm consensus target fishing and in silico ADMET profiling of anabolic-androgenic steroids: a computational polypharmacology framework for proactive anti-doping detection**
> A. Malagón, D. Cañón — *Forensic Toxicology* (submitted)

The repository reproduces all tables and figures of the manuscript and its Online Resource 1 (ESM): a two-layer in silico framework combining (i) multi-algorithm consensus target fishing across four platforms (SwissTargetPrediction, FGP, PLATO, PHARMAPPER) and (ii) ADMET/toxicity profiling with ADMETlab 3.0, with retrospective metabolite validation (BioTransformer 3.0) and cross-platform toxicity triangulation (pkCSM, admetSAR 3.0) for 67–68 WADA S1 anabolic agents.

## Repository structure

```
├── code/                              # analysis scripts (run in order 01 → 09)
│   ├── 01_pipeline_consenso.py
│   ├── 02_figuras_1_5.py
│   ├── 04_analisis_admet.py
│   ├── 05_sensibilidad_pesos.py
│   ├── 06_red_enriquecimiento.py
│   ├── 07_validacion_biotransformer.py
│   ├── 08_triangulacion_admet.py
│   ├── 09_validacion_externa_tox21.py
│   └── sar_analysis.py
├── data/                              # input and result datasets (CSV)
└── README.md
```

## Requirements

- Python ≥ 3.11
- Dependencies (see `requirements.txt`):

```bash
pip install -r requirements.txt
```

- **Optional (script 07 only):** [BioTransformer 3.0](https://bitbucket.org/wishartlab/biotransformer/) standalone JAR. The repository already contains the BioTransformer outputs and the derived validation table, so running the JAR is only needed to regenerate metabolite predictions from scratch.
- **Script 09 requires internet access** (queries the PubChem PUG-REST API). The resulting `external_validation_tox21.csv` is already included, so the script is only needed to regenerate the benchmark.

## Execution order

| Step | Script | Main inputs | Main outputs | Reproduces |
|------|--------|-------------|--------------|------------|
| 1 | `01_pipeline_consenso.py` | `HORMONES_ANABOLIC_ALL_TARGETS.xlsx` (raw exports of the 4 platforms, one sheet each) | `consensus_matrix.csv`, `consensus_long.csv`, `indices_clases.csv` | §2.2–2.5, §3.1–3.6; Tables 1–4 |
| 2 | `02_figuras_1_5.py` | outputs of step 1 | Figures 1–5 (PNG, 300 dpi) | Figures 1–5 |
| 3 | `04_analisis_admet.py` | `admet_decoded_corregido.csv` ⚠️ corrected master table | Tables 5–6 values, Figures 6–7 | §3.7–3.9; Tables 5–6; Figures 6–7 |
| 4 | `05_sensibilidad_pesos.py` | `consensus_long.csv` | `sensibilidad_pesos_claseIII.csv`, Fig. S1 | Figure S1 |
| 5 | `06_red_enriquecimiento.py` | `consensus_long.csv`, `uniprot_genes.csv` | `enriquecimiento_STRING_132.csv`, Figs. S2–S3 | Figures S2–S3 |
| 6 | `07_validacion_biotransformer.py` | `inputFolder/`, `results_csv/`, `results_csv2/`, `results_sdf/` (from `raw_platform_outputs.zip`), `indices_clases.csv`, `smiles_68_corregidos.csv` | `validacion_biotransformer.csv` (Table S1), `metabolitos_predichos_det.csv`, Figure 8 | §3.11; Figure 8; Table S1 |
| 7 | `08_triangulacion_admet.py` | `admet_decoded_corregido.csv`, `smiles_68_corregidos.csv`, `pkcsm_out/`, `admetsar_out/` (from `raw_platform_outputs.zip`) | `triangulacion_admet.csv` (Table S3), Figure 9 | §3.12; Figure 9; Table S3 |
| 8 | `sar_analysis.py` | `aas_data.csv`, `aas_structural_classification.csv` | SAR statistics by structural class | supports §3.8 |
| 9 | `09_validacion_externa_tox21.py` 🌐 | `smiles_68_corregidos.csv`, `consensus_long.csv`; PubChem PUG-REST (AIDs 743053/743063) | `external_validation_tox21.csv` (Table S4) | §3.10; Table S4 |

## Data files

| File | Description |
|------|-------------|
| `consensus_matrix.csv` | Consensus scores, 67 compounds × 573 UniProt targets |
| `consensus_long.csv` | Long format: per-platform scores and consensus for 6,700 interactions |
| `indices_clases.csv` | Detectability Index, Endocrine Impact Index and risk class (I–IV) per compound |
| `uniprot_genes.csv` | UniProt → gene symbol mapping for the 573 targets |
| `aas_structural_classification.csv` | Structural annotation (17α-alkylated, 19-nor, etc.) per compound |
| `smiles_68_corregidos.csv` | Master list of the 68 verified canonical SMILES (7 structure-corrected) |
| `admet_decoded_corregido.csv` | **Definitive ADMETlab 3.0 master table (68 × 47 endpoints)**, with the 7 structure-corrected compounds re-queried through the official API |
| `admetlab3_correccion_7compuestos.csv` | Before/after comparison for the 7 corrected compounds (**Table S2**) |
| `validacion_biotransformer.csv` | Retrospective validation vs. 24 documented WADA urinary markers (**Table S1**) |
| `metabolitos_predichos_det.csv` | Predicted metabolite counts per compound vs. DI/EII (Figure 8b–c) |
| `triangulacion_admet.csv` | Per-compound toxicity calls across ADMETlab 3.0 / pkCSM / admetSAR 3.0 (**Table S3**) |
| `external_validation_tox21.csv` | External benchmark vs. Tox21 AR agonist/antagonist qHTS data (PubChem AIDs 743053/743063): experimental calls, agonist AC50, SwissTP probability, consensus AR score and AR rank for the 19 screened agents (**Table S4**) |
| `enriquecimiento_STRING_132.csv` | STRING functional enrichment of the 132 prioritized targets (Figure S3) |
| `sensibilidad_pesos_claseIII.csv` | Consensus-weight sensitivity for Class III steroids (Figure S1) |
| `aas_data.csv` | Merged per-compound table (ADMET endpoints + DI/EII + consensus AR) used by `sar_analysis.py` |

### Raw platform outputs

`raw_platform_outputs.zip` (unzip in the repository root) contains the raw third-party outputs parsed by scripts 07 and 08:

- `inputFolder/` — BioTransformer batch inputs (`aasbatch.csv`, `inputSmilesBatch_*.csv`)
- `results_csv/`, `results_csv2/`, `results_sdf/` — BioTransformer 3.0 outputs (1- and 2-iteration runs)
- `pkcsm_out/` — 68 pkCSM toxicity result pages (`pkcsm_XX.html`)
- `admetsar_out/` — 68 admetSAR 3.0 result tables (`admetsar_XX.txt`)
- `admetlab3_out/` — ADMETlab 3.0 API JSON responses for the 7 structure-corrected compounds (provenance of Table S2)

> ⚠️ `admet_decoded.csv` (if present) is the **superseded** version containing 7 compounds with erroneous SMILES; it is kept only for provenance. Always use `admet_decoded_corregido.csv`.

## Notes

- Prediction probabilities were binarized at 0.5 unless otherwise stated; extreme values (≈0 or ≈1) should be read as rankings, not absolute risks (see §4.6 of the manuscript).
- ADMETlab 3.0, pkCSM and admetSAR 3.0 predictions were collected January–September 2026; web-server model updates may shift individual probabilities.

## License

MIT (code). The datasets are released under CC-BY-4.0.

## Contact

Andrés Malagón — edmalagon@uan.edu.co
Universidad Antonio Nariño, Bogotá, Colombia
