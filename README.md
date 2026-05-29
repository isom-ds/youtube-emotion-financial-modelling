# youtube-emotion-financial-modelling
> Modelling the impact of YouTube emotion and topic signals on asset returns during the 2025 US Tariff crisis.

## Abstract

Social media discourse during financial crises carries affective signals that may precede or accompany market movements. This repository implements a pipeline for extracting Plutchik emotion signals and 18 financial/social topic labels from YouTube transcripts and comments during the 2025 US Tariff crisis, then models their relationship to asset returns. The work contributes an affective decision-support lens to financial markets by demonstrating how online emotional dynamics co-vary with market volatility.

## Research Context

- **Thesis:** *Epidemiology of Online Emotions* (Kok-Shun, 2026)
- **Chapter:** Chapter 7 — Affective State Indices for Decision-Making
- **Contribution type:** Artefact (financial decision-support system)
- **Associated paper:** "Flaring Emotions, Hot Topics and Volatile Markets: Modelling the Impacts of the 2025 US Tariffs on Social Media Perspectives and Asset Returns," ACIS 2025

## Methods

- Plutchik 8-emotion model (anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
- Valence categorisation (positive / negative); surprise treated as informational novelty
- OpenAI batch API for large-scale emotion and topic extraction
- 18-topic classification (Conspiracy and Historical Events, Disaster and Resilience, Disputes and Opinions, Economic and Financial Matters, Education and Knowledge, Environmental and Infrastructure, Health and Social Welfare, Labor and Workforce, Leadership and Governance, Legal and Regulatory Framework, Miscellaneous and Other, Power and Influence Dynamics, Psychological and Emotional Aspects, Quality of Life and Standards, Security and Conflict, Social and Cultural Aspects, Sports and Recreation, Trust/Credibility/Communication)
- Financial time-series modelling of asset returns (ARX, VARX baselines and emotion-augmented models)

## Datasets

| Dataset | Description | Access |
|---------|-------------|--------|
| YouTube Transcripts & Comments | US Tariff discourse (2025), stored as Parquet by topic | Collected |
| Financial market data | Asset return time series for modelling (`data/financial_returns_and_controls.*`) | Collected |

## Repository Structure

```
youtube-emotion-financial-modelling/
├── modules/
│   ├── ai/       # GPT-based emotion and topic extraction
│   ├── data/     # Data loading and preprocessing utilities
│   ├── finance/  # Financial modelling (ARX, VARX)
│   └── index/    # Polarity index aggregation
├── openai_wrapper/
│   ├── batch_process.py  # OpenAI batch API wrapper
│   └── utils.py
├── utils/
│   ├── formatters.py
│   ├── gcs.py            # Google Cloud Storage utilities
│   └── requests.py
├── data/                 # Parquet and JSON data files
├── constants.py          # Plutchik emotions, valence sets, topic list
├── keys.py
├── main.py               # Full pipeline entry point
├── 100_data_apism_yt_finance.ipynb
├── 110_data_finance_clean.ipynb
├── 200_ai_topic_emotion.ipynb
├── 210_ai_extraction.ipynb
├── 220_ai_topic_refinement.ipynb
├── 230_combine_reshape.ipynb
├── 300_index_module.ipynb
├── 400_index_comparison_epi.ipynb
├── 401_index_comparison_signed_epi.ipynb
├── 402_index_comparison_ccd.ipynb
├── 403_index_comparison_surprise.ipynb
├── 404_index_comparison_intensity_split.ipynb
├── 405_index_comparison_ei.ipynb
├── 410_fm_baseline_arx.ipynb
├── 411_fm_baseline_varx.ipynb
├── 412_fm_arx_emotion.ipynb
├── 413_fm_varx_emotion.ipynb
├── 500_fm_model_summary.ipynb
├── 600_insights_module.ipynb
└── pyproject.toml
```

## Requirements & Setup

**Stack:** Python 3.12, OpenAI API (batch processing), Pandas, Polars, PyArrow, GCS credentials (optional).

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key-here
```

## Usage

Run the full pipeline via `main.py`, or execute notebooks individually for specific analyses:

```
100_data_apism_yt_finance   → YouTube data ingestion
110_data_finance_clean      → financial data cleaning
200_ai_topic_emotion        → topic and emotion labelling
210_ai_extraction           → structured AI extraction (batch API)
220_ai_topic_refinement     → topic refinement loop
230_combine_reshape         → data combining and reshaping
300_index_module            → polarity index computation
400–405_index_comparison_*  → index variant comparisons
410–413_fm_*                → financial modelling (ARX/VARX baselines and emotion-augmented)
500_fm_model_summary        → model results summary
600_insights_module         → visualisation and insights
```

Data is stored as Parquet files under `data/`.

## References

B. V. Kok-Shun, J. Chan, G. Peko, and D. Sundaram, "Flaring Emotions, Hot Topics and Volatile Markets: Modelling the Impacts of the 2025 US Tariffs on Social Media Perspectives and Asset Returns," in *ACIS 2025 Proceedings*, 2025 [Online]. Available: https://aisel.aisnet.org/acis2025/263.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{P9_kok-shun_flaring_2025,
  title     = {Flaring {Emotions}, {Hot} {Topics} and {Volatile} {Markets}: {Modelling} the {Impacts} of the 2025 {US} {Tariffs} on {Social} {Media} {Perspectives} and {Asset} {Returns}},
  booktitle = {{ACIS} 2025 {Proceedings}},
  author    = {Kok-Shun, Brice Valentin and Chan, Johnny and Peko, Gabrielle and Sundaram, David},
  year      = {2025},
  url      = {https://aisel.aisnet.org/acis2025/263},
}
```

</details>
