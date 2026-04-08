# Causal-Weighted Machine Learning for Generalizable Prediction of Renal Replacement Therapy in Acute Kidney Injury

This repository contains the source code for the paper:

> **Causal-Weighted Machine Learning for Generalizable Prediction of Renal Replacement Therapy in Acute Kidney Injury**
>
> Submitted to *IEEE Journal of Biomedical and Health Informatics (JBHI)*

## Overview

We develop and externally validate a leakage-safe, regularized stacking pipeline for predicting renal replacement therapy (RRT) initiation within 48 hours in critically ill patients with KDIGO Stage 3 acute kidney injury (AKI). The model integrates causal-weighted propensity score features with a LightGBM + XGBoost + RandomForest stacking ensemble and achieves:

| Metric | Internal Test (MIMIC-IV) | External (eICU) |
|--------|------------------------:|----------------:|
| AUC-ROC | 0.872 | 0.802 |
| Accuracy | 0.877 | 0.829 |
| Brier Score | 0.096 | 0.123 |

## Data Sources

- **MIMIC-IV** (v2.2): Beth Israel Deaconess Medical Center, Boston, 2008–2019. Accessed via [PhysioNet](https://physionet.org/content/mimiciv/).
- **eICU-CRD** (v2.0): 200+ US hospitals, 2014–2015. Accessed via [PhysioNet](https://physionet.org/content/eicu-crd/).

> **Note:** Raw data is not included. You must obtain access through PhysioNet and load the data into a local PostgreSQL database.

## Repository Structure

```
code/
├── preprocessing/           # Data extraction and preprocessing
│   ├── mimic/               #   MIMIC-IV preprocessing (R + SQL)
│   │   ├── config/          #     Database connection config
│   │   ├── scripts/         #     Preprocessing scripts
│   │   └── sql/             #     Schema fixes for local MIMIC-IV
│   ├── eicu/                #   eICU-CRD preprocessing (R + Python)
│   │   ├── config/          #     Database connection config
│   │   └── scripts/         #     Feature extraction & alignment
│   └── data_split/          #   Train/Val/Test split (7:2:1) + imputation
│
├── models/                  # Model training
│   ├── lightgbm/            #   LightGBM, XGBoost, Stacking models (Python)
│   ├── dwols/               #   dWOLS causal treatment policy (R)
│   └── joint/               #   Joint LightGBM + dWOLS pipeline
│
├── experiments/             # Reproducibility experiments
│   ├── ablation/            #   Ablation studies (feature type, timepoint, etc.)
│   ├── analysis/            #   Baseline comparison, learning curves, etc.
│   └── validation/          #   External validation & domain adaptation
│
├── web_app/                 # Prototype CDSS web application
│   ├── backend/             #   FastAPI REST API
│   └── frontend/            #   Streamlit + HTML/CSS interface
│
├── visualization/           # Publication-quality figure generation
├── clinical_analysis/       # Subgroup analysis, calibration, bootstrap CI
├── scripts/                 # Fusion pipeline & paper figure scripts
├── .env.example             # Environment variable template
└── .gitignore
```

## Requirements

### R (≥ 4.1)
- `RPostgreSQL`, `mice`, `dplyr`, `data.table`, `survival`

### Python (≥ 3.9)
- `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `xgboost`
- `shap`, `matplotlib`, `seaborn`
- `fastapi`, `uvicorn`, `streamlit` (for web app)

Install Python dependencies:
```bash
pip install numpy pandas scikit-learn lightgbm xgboost shap matplotlib seaborn
pip install fastapi uvicorn streamlit  # for web app only
```

## Setup

### 1. Database Configuration

Copy `.env.example` and fill in your PostgreSQL credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
MIMIC_DB_NAME=MIMIC
EICU_DB_NAME=eICU
```

### 2. Data Preprocessing

```bash
# 1. Fix MIMIC-IV schema (run SQL files in order)
psql -d MIMIC -f preprocessing/mimic/sql/setup_mimiciv_alias_schemas.sql
psql -d MIMIC -f preprocessing/mimic/sql/fix_inputevents_mapping.sql
psql -d MIMIC -f preprocessing/mimic/sql/fix_procedureevents_mapping.sql

# 2. Run MIMIC preprocessing
Rscript preprocessing/mimic/scripts/run_preprocessing.R

# 3. Run eICU preprocessing
Rscript preprocessing/eicu/scripts/run_preprocessing.R

# 4. Split and impute
Rscript preprocessing/data_split/split_preprocessed_data.R
```

### 3. Model Training

```bash
# Train stacking model (main result)
python models/lightgbm/train_stacked_optimized.py

# Train dWOLS causal policy (optional)
Rscript models/dwols/train_dwols_with_uo.R
```

### 4. Experiments

```bash
# Run all ablation studies
python experiments/ablation/run_all_ablations.py

# External validation
python experiments/validation/improved_external_validation.py
```

### 5. Web Application (Optional)

```bash
# Start backend API
cd web_app/backend && uvicorn app:app --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
cd web_app/frontend && streamlit run streamlit_app.py
```

## Key Design Decisions

- **Leakage prevention**: All features use only pre-prediction-time information. The decision threshold is tuned on the validation split only; test and external sets are never used for fitting.
- **Causal weighting**: Propensity scores (ps_k1, ps_k2) are computed via logistic regression on training data only, then applied as stabilized IPW features.
- **Stacking architecture**: LightGBM + XGBoost + RandomForest base learners with Logistic Regression meta-learner (2-fold CV for meta-feature generation).
- **Anti-overfitting**: Constrained tree depth (max_depth=4–5), limited estimators (80–100), L1/L2 penalties, and class balancing.

## Citation

If you use this code, please cite:

```bibtex
@article{liao2026causal,
  title={Causal-Weighted Machine Learning for Generalizable Prediction of 
         Renal Replacement Therapy in Acute Kidney Injury},
  author={Liao, Zhaoyu and Chen, Ziyi and Wang, Siyuan and Du, Qiang 
          and Liao, Xiaozhu and Huang, Qiuchen},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026}
}
```

## Ethics

This study uses de-identified public datasets (MIMIC-IV and eICU-CRD) accessed through PhysioNet credentialed data use agreements. All authors completed CITI Program training for human subjects research.

## License

This project is licensed under the MIT License.
