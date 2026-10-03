# Finished Bar Yield Strength Prediction

Machine learning-based prediction of finished bar yield strength (YS) from steel chemistry,
**Reheating Furnace (RHF)** parameters and **rolling mill / water-box** parameters.

**Plant:** IISCO Steel Plant (ISP), Burnpur

## Problem
Yield strength of finished bar is only confirmed by destructive lab testing after rolling, so
off-spec strength is detected late. RHF/mill settings (soaking temperature, zone temperatures,
water-box flow & pressure, mill speed) are largely set from operator experience rather than a
validated model.

## Objective
Build a data-driven model that predicts finished bar YS from grade/chemistry, RHF and mill
parameters, enabling real-time property prediction and parameter optimisation to reduce trials,
downgrades and rejections.

## Status of data
> **No plant data is available yet.** `data/raw/process_data.csv` is produced by a
> physics-informed synthetic generator (`ysp/data_generator.py`) with the schema the real data
> should have. All numbers in `reports/` come from this stand-in and demonstrate the pipeline,
> not ISP metallurgy. Replace the CSV with real data (same columns, see `ysp/config.py`),
> re-run the commands below, and revisit the error band and spec limits with plant metallurgists.

## Deliverables
| # | Deliverable | Where |
|---|-------------|-------|
| 1 | Regression/ML model for YS within an agreed error band, benchmarked vs lab data | `ysp/models.py`, `reports/model_metrics.json` |
| 2 | Ranking of RHF & mill parameters by influence on YS | `ysp/importance.py`, `reports/parameter_ranking.md` |
| 3 | Recommended operating windows by grade (water-box pressure/flow, mill speed) | `ysp/optimize.py`, `reports/operating_windows.md` |
| 4 | Reduction in rejection and mill-speed optimisation | `reports/optimisation_summary.json` |
| 5 | Working prototype | `app/streamlit_app.py` |

## Quick start
```bash
pip install -r requirements.txt
python -m ysp.data_generator      # (or supply real data) -> data/raw/process_data.csv
python -m ysp.preprocess          # clean + engineer features -> data/processed/
python -m ysp.eda                 # figures in reports/figures, reports/eda_summary.md
python -m ysp.models              # train, grouped CV, holdout benchmark -> models/, reports/
python -m ysp.importance          # parameter ranking
python -m ysp.optimize            # operating windows + off-spec estimate (also trains the UTS model)
streamlit run app/streamlit_app.py
pytest
```

## Method
1. **Cleaning** – stuck-at-zero sensors and out-of-range values become missing, rows without a lab
   result are dropped, remaining gaps are filled with grade+diameter medians.
2. **Features** – raw chemistry/RHF/mill/water-box signals plus physically motivated terms
   (quench index, 1/speed dwell, surface-to-volume, soak-to-finish drop, microalloy sum, ...).
3. **Validation** – chronological holdout of the latest 20% of heats (no heat shared with
   training); model selection by 5-fold cross-validation grouped by heat.
   Models: ridge, random forest, gradient boosting. Error band: ±15 MPa (`ERROR_BAND_MPA`).
4. **Influence** – permutation importance and +1 std sensitivity on a raw-parameter model.
5. **Optimisation** – per heat, grid search of water-box pressure, flow and mill speed within the
   range the line has run for that bar size, requiring YS inside the grade band and UTS/YS above
   the minimum with a safety margin; fastest feasible speed wins. Windows are P10-P90 of the
   per-heat optima.

## Results on synthetic data (illustrative)
| Metric | Value |
|---|---|
| Holdout MAE / RMSE | 6.7 / 8.5 MPa |
| Holdout R² | 0.944 |
| Predictions within ±15 MPa of lab | 92.5% |
| Expected off-spec (current -> recommended) | 21.6% -> 7.0% |
| Mean mill speed | +5% |

## Layout
```
ysp/          config, data_generator, preprocess, eda, models, importance, optimize, predict
app/          Streamlit prototype
tests/        pytest suite
data/raw/     input data        data/processed/  generated (gitignored)
models/       trained models (gitignored; regenerate with python -m ysp.models)
reports/      metrics, rankings, windows, figures
```

## Limitations / next steps
- Replace synthetic data with real ISP heat/process/lab records and retrain; expect lower accuracy.
- Confirm spec limits (`GRADES`), error band and sensor ranges with the plant.
- Off-spec reduction is a model-based estimate; validate with controlled mill trials.
- Recommendations assume chemistry, furnace and finishing temperature are fixed per heat.
