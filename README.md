# ML-Driven Equity Alpha & Market-Neutral Portfolio

## Goal
Generate robust cross-sectional alpha factors with ML and allocate to a market-neutral, risk-controlled portfolio with realistic costs and constraints.

## Pipeline
1. Data ingest → cleaning → bias controls
2. Factor engineering → standardize/winsorize → neutralize
3. Labeling (forward returns) → purged CV
4. Model training (LightGBM baseline) → IC/RankIC eval
5. Portfolio construction (beta/net=0, sector caps) → backtest

## Quickstart
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- python -m src.cli.run_backtest

## Experiments
All parameters live in configs/*.yaml. Copy and edit an experiment file under experiments/ and run the CLI.

## Data
Place vendor files in data/raw (see configs/data.yaml). Avoid committing large or proprietary data.
