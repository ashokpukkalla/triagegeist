# Triagegeist: Multi-Modal Undertriage Detection

**AI-powered emergency triage decision support using calibrated ensemble learning.**

Competition entry for the [Triagegeist Hackathon](https://www.kaggle.com/competitions/triagegeist) — Laitinen-Fredriksson Foundation, 2026.

---

## Overview

This project builds a multi-modal undertriage detection system for emergency departments. It predicts ESI (Emergency Severity Index) acuity levels from structured patient data, free-text chief complaints, and clinical history, with a focus on catching dangerous undertriage — when critically ill patients are assigned lower acuity than warranted.

## Key Features

- **4-Model Calibrated Ensemble**: LightGBM + XGBoost + MLP Neural Network + Logistic Regression with F1-weighted soft voting
- **Missingness-as-Clinical-Signal** (Novel): Information-theoretic analysis of which vitals are *not* recorded — formalizing the clinical observation that missing data in EDs is not random
- **Sentence-BERT NLP**: 384-dimensional dense semantic embeddings of chief complaint text
- **Undertriage Safety Net**: Binary high-sensitivity detector (≥95% recall) for ESI 1-2 patients
- **~490 Multi-Modal Features**: Vitals + clinical flags + computed scores + demographics + comorbidities + missingness + NLP
- **SHAP Explainability**: Global and local feature importance by clinical category
- **Bootstrap Confidence Intervals**: 95% CIs on all reported metrics
- **Demographic Fairness Audit**: Across age, sex, language, and insurance type

## Repository Structure

```
triagegeist/
├── README.md                  # This file
├── WRITEUP.md                 # Competition writeup
├── cover_image.png            # 560x280 cover image
├── requirements.txt           # Python dependencies
├── build_notebook.py          # Notebook builder script
├── notebook.ipynb             # Generated competition notebook
└── data/                      # Competition data (not included)
    ├── train.csv
    ├── test.csv
    ├── chief_complaints.csv
    ├── patient_history.csv
    └── sample_submission.csv
```

## Setup & Reproduction

### On Kaggle (Recommended)
The notebook is designed to run end-to-end on Kaggle with GPU/CPU:

1. Go to the [competition page](https://www.kaggle.com/competitions/triagegeist)
2. Open the [notebook](https://www.kaggle.com/code/ashokpukkalla/triagegeist-multi-modal-undertriage-detection)
3. Click "Copy & Edit" → Run All

### Local Setup
```bash
git clone https://github.com/ashokpukkalla/triagegeist.git
cd triagegeist
pip install -r requirements.txt

# Download competition data from Kaggle
kaggle competitions download -c triagegeist -p data/

# Build and run the notebook
python build_notebook.py
jupyter notebook notebook.ipynb
```

## Requirements

- Python 3.10+
- lightgbm >= 4.0
- xgboost >= 2.0
- scikit-learn >= 1.3
- sentence-transformers >= 2.2
- shap >= 0.42
- scipy >= 1.10
- matplotlib >= 3.7
- seaborn >= 0.12
- pandas >= 2.0
- numpy >= 1.24

## Methodology

### Data Integration
Three data sources merged on `patient_id`:
- Structured vitals and demographics (train.csv)
- Free-text chief complaints (chief_complaints.csv)
- Binary comorbidity flags (patient_history.csv)

### Feature Engineering (~490 features)
| Category | Count | Source |
|----------|-------|--------|
| Raw Vitals | 10 | train.csv |
| Clinical Flags | 18 | Computed (expert thresholds) |
| Computed Scores | 6 | MEWS, Shock Index, NEWS2, MAP |
| Demographics & Context | ~25 | Temporal, arrival, mental status |
| Comorbidity Profile | ~30 | History flags + combinations |
| Missingness Signature | 11 | Entropy + per-vital + interactions |
| NLP Embeddings | 384 | Sentence-BERT |
| Vital Interactions | 3 | Cross-feature products |

### Novel: Missingness Entropy
Shannon entropy of binary missingness vector captures documentation completeness as a clinical signal. Combined with ambulance-missing interaction to detect anomalous incomplete workups.

### Ensemble Architecture
Four models combined via probability-weighted soft voting (weights ∝ macro-F1):
1. LightGBM (histogram boosting, native missing handling)
2. XGBoost (complementary regularization)
3. MLP Neural Net (256→128→64, captures non-linear interactions)
4. Logistic Regression (calibrated linear baseline)

### Clinical Safety
- Undertriage Safety Net: Binary ESI 1-2 detector at ≥95% sensitivity
- Fairness Audit: No demographic disparities >3pp
- SHAP Explainability: Per-patient feature attribution

## Results

### Synthetic Data (Competition Dataset)
Cross-validated on 80,000 synthetic patients with 3-fold stratified CV:
- Accuracy: 99.97% | F1 (macro): 99.91% | QWK: 0.9998

### Real-World Validation (NHAMCS 2022)
Applied identical pipeline to 10,207 real CDC emergency department visits:

| Model | Accuracy | F1 (macro) | QWK |
|-------|----------|------------|-----|
| LightGBM | 53.95% | 0.261 | 0.188 |
| XGBoost | 54.14% | 0.258 | 0.196 |
| MLP | 53.96% | 0.252 | 0.215 |
| LogReg | 29.64% | 0.243 | 0.235 |
| **Ensemble** | **54.85%** | **0.280** | **0.239** |

**Key finding**: 45.1 percentage point accuracy gap between synthetic and real data, confirming synthetic over-specification. Real-world QWK of 0.239 is below published ESI inter-rater reliability (0.60-0.80).

### Ablation Study (NHAMCS)
| Removed | Features | Accuracy | Delta |
|---------|----------|----------|-------|
| Full model | 72 | 45.99% | — |
| − Vitals | 65 | 43.63% | −2.36% |
| − Demographics | 61 | 44.29% | −1.69% |
| − Comorbidities | 49 | 44.50% | −1.49% |
| − Missingness | 62 | 45.01% | −0.98% |

Vitals most impactful; missingness features contribute ~1pp — validating documentation patterns as genuine clinical signal.

See the [notebook](https://www.kaggle.com/code/ashokpukkalla/triagegeist-multi-modal-undertriage-detection) for full results with confidence intervals.

## Limitations

1. Synthetic dataset — contains over-specified features that inflate accuracy
2. No outcome validation (no 30-day mortality data)
3. No temporal dynamics (arrival-time features only)
4. Language bias in NLP embeddings
5. Single-site — external validation required

## License

This project is submitted as part of the Triagegeist competition. Code is provided for reproducibility.

## Citation

```
@misc{pukkalla2026triagegeist,
  title={Triagegeist: Multi-Modal Undertriage Detection via Calibrated Ensemble Learning},
  author={Pukkalla, Ashok},
  year={2026},
  note={Kaggle Competition Entry}
}
```

## References

1. Gilboy, N., et al. (2012). ESI: A Triage Tool for ED Care, Version 4. AHRQ.
2. Farrohknia, N., et al. (2011). ED triage scales: A systematic review. Scandinavian J. Trauma.
3. Obermeyer, Z., et al. (2019). Dissecting racial bias in health algorithms. Science.
4. Reimers, N. & Gurevych, I. (2019). Sentence-BERT. EMNLP.
