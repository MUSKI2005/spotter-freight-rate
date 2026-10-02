# Spotter AI — Freight Rate Prediction

Machine learning pipeline for predicting posted freight rates from historical load data.

## 1. Project Overview

This project develops a regression pipeline for predicting `posted_rate` for unseen freight loads.

The development dataset contains 48,000 labeled historical loads, while the final validation dataset contains 12,000 unlabeled loads. Because the prediction task is forward-looking, model development uses chronological validation rather than a random train/test split.

The final solution uses a `CatBoostRegressor` trained on a log-transformed target:

`log1p(posted_rate)`

Predictions are converted back to the original dollar scale using:

`expm1(prediction)`

---

## 2. Repository Structure

```text
spotter-freight/
│
├── data/
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
│
├── plots/
│   ├── plot1.png
│   └── plots_combined.png
│
├── reports/
│   ├── dev_fold_metrics.csv
│   ├── dev_summary.csv
│   ├── frozen_config.json
│   ├── october_metrics.csv
│   └── october_predictions.csv
│
├── scorer_results/
│   └── candidate_december.png
│
├── eda.ipynb
├── train_test_notebook.ipynb
├── main.py
├── score.py
├── requirements.txt
├── validation_predictions.csv
├── december_predictions.csv
└── README.md