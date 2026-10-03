# Finished Bar Yield Strength Prediction

Machine learning-based prediction of finished bar yield strength (YS) from steel chemistry,
**Reheating Furnace (RHF)** parameters and **rolling mill / water-box** parameters.

**Plant:** IISCO Steel Plant (ISP), Burnpur

## Problem
Yield strength of finished bar is only known after rolling, via destructive lab tests, so
off-spec strength is detected late. RHF/mill settings (soaking temperature, zone temperatures,
water-box flow & pressure, mill speed) are set from operator experience, not a validated model.

## Objective
Build a data-driven model that predicts finished bar YS from grade/chemistry, RHF and mill
parameters for real-time property prediction and parameter optimisation, reducing trials,
downgrades and rejections.

## Deliverables
| # | Deliverable | Where |
|---|-------------|-------|
| 1 | Validated regression/ML model for YS within an agreed error band, benchmarked vs lab data | `ysp/models.py`, `reports/` |
| 2 | Ranking of RHF & mill parameters by influence on YS | `ysp/importance.py` |
| 3 | Recommended operating windows by grade (water-box pressure/flow, mill speed) to meet target YS/UTS | `ysp/optimize.py` |
| 4 | Reduction in rejection / optimisation of mill speed | `ysp/optimize.py` |
| 5 | Working prototype | `app/` (Streamlit) |

## Build phases
1. **Scaffold + data** – project layout, config, physics-informed synthetic data generator (stand-in until plant data arrives)
2. EDA, cleaning & feature engineering
3. Model training, validation & benchmarking
4. Parameter influence ranking (sensitivity / feature importance)
5. Operating-window optimisation by grade
6. Prototype app + documentation

## Quick start
```bash
pip install -r requirements.txt
python -m ysp.data_generator      # writes data/raw/process_data.csv
python -m ysp.preprocess          # clean + engineer features -> data/processed/
python -m ysp.models               # train, CV, benchmark -> models/, reports/model_metrics.json
python -m ysp.importance           # parameter ranking -> reports/parameter_ranking.md
python -m ysp.eda                 # figures in reports/figures, reports/eda_summary.md
pytest
```

## Using real plant data
Drop a CSV with the same columns as `data/raw/process_data.csv` (see `ysp/config.py` for
feature groups) in `data/raw/`. The synthetic file is clearly a stand-in; metallurgical
conclusions should only be drawn from models trained on real data.
