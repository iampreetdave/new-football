# New Football Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6C35?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

> An end-to-end football match prediction system using ensemble ML models with a custom confidence scoring method.

## About

A comprehensive football prediction pipeline that fetches live match data from the FootyStats API, engineers 40+ features per match, trains ensemble models (XGBoost, Gradient Boosting, Random Forest), and generates Over/Under 2.5 and Moneyline predictions with confidence scores. Uses the "Fottyy Confidence Method" (60% base probability + 40% margin) for calibrated prediction confidence. Designed for automated daily predictions across multiple leagues.

## Tech Stack

- **Language:** Python 3
- **ML:** XGBoost (GPU), Gradient Boosting, Random Forest
- **Data:** Pandas, NumPy, scikit-learn
- **API:** FootyStats API
- **Serialization:** joblib

## Features

- **Live data fetching** — pulls today's matches and team stats from the FootyStats API
- **Rich feature engineering** — xG, PPG, Elo ratings, form, H2H, shots accuracy, dangerous attacks, market odds
- **Ensemble predictions** — weighted average of XGBoost, Gradient Boosting, and Random Forest with market probability blending
- **Dual prediction targets** — Over/Under 2.5 goals and Moneyline (Home/Away win)
- **CTMCL calculation** — custom Consensus Total Market Goals Line from O/U odds
- **Confidence scoring** — Fottyy method: 60% base confidence + 40% margin confidence
- **GPU acceleration** — XGBoost configured for CUDA when available
- **Automated pipeline** — fetch → feature extraction → model training → inference → CSV output
- **Match mapping** — maps matches across different data sources for validation

## Getting Started

### Prerequisites

- Python 3.8+
- FootyStats API key (set as `FOOTYSTATSAPI` environment variable)

### Installation

```bash
git clone https://github.com/iampreetdave/new-football.git
cd new-football
pip install -r requirements.txt
```

### Run

**1. Train models on historical data:**

```bash
python genrate_models.py
```

**2. Fetch today's live matches and extract features:**

```bash
python fetch_data.py
```

**3. Generate predictions:**

```bash
python genrate_predictions.py
```

## How It Works

1. **Data Collection:** `fetch_data.py` reads `live.csv` (today's matches), calls the FootyStats API for team and league stats, and engineers 40+ features per match
2. **Model Training:** `genrate_models.py` loads historical data (`top.csv`), computes rolling features (Elo, form, xG averages), and trains XGBoost/GB/RF ensembles for O/U 2.5 and Moneyline
3. **Inference:** `genrate_predictions.py` loads pre-trained models from `models/`, transforms live features, and generates weighted ensemble predictions blended with market odds
4. **Confidence:** Each prediction gets a confidence score via the Fottyy method — 60% from the max predicted probability + 40% from the margin between predicted classes
5. **Output:** Results saved to `predictions_output.csv` with match details, predictions, odds, and confidence levels

## Project Structure

```
new-football/
├── fetch_data.py              # API data fetcher & feature extractor
├── genrate_models.py          # Model training pipeline
├── genrate_predictions.py     # Inference pipeline
├── match_mapping.py           # Match ID mapping across sources
├── validate_predictions.py    # Prediction validation & grading
├── v3_over_under.py           # Over/Under analysis variant
├── v3_ou_grade.py             # O/U grading utility
├── save_predictions.py        # Prediction persistence
├── new_save.py                # Updated save logic
├── winbetsID.py               # WinBets ID mapping
├── requirements.txt           # Python dependencies
├── models/                    # Trained model files (.pkl)
├── live.csv                   # Today's match input
├── top.csv                    # Historical training data
├── extracted_features_complete.csv
├── predictions_output.csv
└── README.md
```

## License

This project is licensed under the [MIT License](LICENSE).
