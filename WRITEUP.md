# Triagegeist: Multi-Modal Undertriage Detection via Calibrated Ensemble Learning

**Author**: Ashok Pukkalla  
**Track**: Triagegeist: AI in Emergency Triage  
**Notebook**: [Triagegeist: Multi-Modal Undertriage Detection](https://www.kaggle.com/code/ashokpukkalla/triagegeist-multi-modal-undertriage-detection)  
**Project Link**: [GitHub Repository](https://github.com/ashokpukkalla/triagegeist)

---

## Clinical Problem Statement

Among ~130 million annual U.S. ED visits (CDC NHAMCS 2022), triage nurses assign ESI acuity levels under extreme cognitive load with incomplete information. Undertriage — assigning lower acuity than warranted — delays care and contributes to preventable mortality, with published rates of 5–20% (Farrohknia et al., 2011) and inter-rater reliability of κ = 0.60–0.80 (Gilboy et al., 2012). Elderly patients and atypical presenters are disproportionately affected (Obermeyer et al., 2019).

We frame triage as an **undertriage safety problem**: a calibrated decision-support system that catches high-acuity patients at risk of dangerous delay, with transparent, explainable predictions clinicians can trust.

---

## Methodology

### Data Integration

We integrate all three competition sources on patient_id:

- **train.csv** (80,000 patients): 40 structured features — vitals, demographics, temporal patterns, arrival context, clinical scores (NEWS2, shock index, MAP)
- **chief_complaints.csv** (100,000 records): Free-text narratives providing clinical context that structured fields cannot capture
- **patient_history.csv** (100,000 records): 25 binary comorbidity flags plus medication counts and prior ED utilization

### Five Novel Components

**1. Four-Model Calibrated Ensemble.** LightGBM, XGBoost, MLP (256-128-64), and L2-penalized Logistic Regression combined via F1-weighted soft voting. This diversity-driven approach produces well-calibrated probabilities critical for threshold-based clinical workflows.

**2. Missingness-as-Clinical-Signal (Novel).** In real EDs, data is not Missing At Random — a triage nurse evaluating a sprained ankle rarely measures SpO2 or GCS. We formalize this with per-vital binary indicators, total missing count, missingness entropy (Shannon entropy of the binary missingness vector), and an ambulance-missing interaction. Kruskal-Wallis testing confirms strong correlation between missing vital count and ESI (p < 0.001).

**3. Sentence-BERT NLP Pipeline.** Chief complaints encoded via all-MiniLM-L6-v2 (384-dimensional embeddings) capture symptom severity, anatomical specificity, and urgency cues that bag-of-words misses. TF-IDF fallback included for environments without sentence-transformers.

**4. Undertriage Safety Net.** Binary classifier (ESI 1-2 vs. 3-5) with threshold tuned for ≥95% sensitivity. Deliberately accepts lower specificity to minimize missed critical patients.

**5. Interactive Clinical Decision Support Demo.** The notebook includes a live triage prediction interface where clinicians adjust patient parameters and instantly see predicted ESI, confidence, probability distribution, and safety-net alerts — demonstrating real-time deployment potential.

### Feature Engineering (~490 features)

| Category | Count | Source |
|----------|-------|--------|
| Raw Vitals + Clinical Flags | 28 | train.csv + computed |
| Computed Scores | 6 | Computed |
| Demographics/Context | ~25 | train.csv |
| Comorbidity Profile | ~30 | patient_history.csv |
| Missingness Signature | 11 | All sources |
| NLP Embeddings | 384 | chief_complaints.csv |
| Vital Interactions | 3 | Computed |

SHAP TreeExplainer provides global and local explanations, grouped by clinical category. 95% bootstrap CIs (1000 resamples) and McNemar's tests validate statistical significance.

---

## Results

### Synthetic Data (3-fold Stratified CV)

| Metric | Value |
|--------|-------|
| Accuracy | 99.97% |
| F1 (macro) | 99.91% |
| QWK | 0.9998 |

Near-perfect performance is expected: synthetic datasets preserve statistical relationships too faithfully. This motivates real-world validation.

### Real-World Validation: NHAMCS 2022

We apply the **identical pipeline** to 10,207 real ED visits from CDC NHAMCS 2022. Key constraint: no free-text chief complaints available, so the model operates on 72 structured features only.

| Model | Accuracy | F1 (macro) | QWK |
|-------|----------|------------|-----|
| LightGBM | 53.95% | 0.261 | 0.188 |
| XGBoost | 54.14% | 0.258 | 0.196 |
| MLP | 53.96% | 0.252 | 0.215 |
| LR | 29.64% | 0.243 | 0.235 |
| **Ensemble** | **54.85%** | **0.280** | **0.239** |

The ensemble consistently outperforms all individual models, validating diversity-driven combination.

**Undertriage on real data**: 20.27% overall rate (2,069/10,207), with 82.8% of ESI 2 patients misclassified as ESI 3 — highlighting that without NLP signals, the model cannot distinguish "emergent" from "urgent."

### Ablation Study (NHAMCS)

| Removed | Δ Accuracy |
|---------|------------|
| Vitals | **−2.36%** |
| Demographics | −1.69% |
| Comorbidities | −1.49% |
| Missingness | −0.98% |
| Interactions | −0.75% |
| Clinical Flags | −0.27% |

Vitals are most impactful. Missingness features contribute ~1 pp — validating that documentation patterns carry genuine clinical signal on real data.

### Performance Gap

| Dataset | N | Accuracy | QWK |
|---------|---|----------|-----|
| Synthetic | 80,000 | 99.97% | 0.9998 |
| NHAMCS (Real) | 10,207 | 54.85% | 0.2388 |

The **45.1-point accuracy gap** quantifies synthetic over-specification. Real-world QWK of 0.239 falls below published inter-rater reliability (κ = 0.60–0.80), proving that structured features alone are insufficient for accurate triage.

---

## Limitations

1. **Synthetic over-specification**: 45.1 pp accuracy gap confirms the competition dataset contains deterministic feature-target encodings. Real-world performance is the honest benchmark.
2. **NLP dependency**: 82.8% ESI 2→3 misclassification rate on NHAMCS (no free-text) proves chief complaint semantics are essential for distinguishing emergent from urgent patients.
3. **No outcome validation**: Without mortality/ICU/readmission data, clinical impact remains theoretical.
4. **Class imbalance**: NHAMCS ESI distribution (52% ESI 3) causes majority-class bias.
5. **Language bias**: Sentence-BERT trained primarily on English text.

---

## Reproducibility

- Single Kaggle notebook runs end-to-end (27.7 min runtime)
- [GitHub repository](https://github.com/ashokpukkalla/triagegeist): complete source, documentation, requirements
- Interactive clinical demo embedded in notebook
- Random seed 42 for all stochastic operations
- External validation: CDC NHAMCS 2022 (Kaggle dataset radhikaaaaaaa/nhamcs2022)

---

## Clinical Implications

Multi-modal data is non-negotiable: structured EHR fields alone achieve 54.85% accuracy — chief complaint NLP must be integrated. At this accuracy, the system serves as a **safety net for undertriage detection**, not a replacement for triage nurses. The 45-point synthetic-to-real gap is a cautionary result for all competition-based clinical ML — real-world validation on external datasets must become standard practice.

---

## References

1. Gilboy, N., et al. (2012). ESI: A Triage Tool for Emergency Department Care, Version 4. AHRQ.
2. Farrohknia, N., et al. (2011). ED triage scales: A systematic review. Scand. J. Trauma.
3. Levin, S., et al. (2018). ML-based electronic triage. Ann. Emerg. Med.
4. Lundberg, S. & Lee, S.I. (2017). Interpreting model predictions. NeurIPS.
5. Klug, M., et al. (2020). Gradient boosting for predicting early mortality in the ED. Acad. Emerg. Med.
6. Obermeyer, Z., et al. (2019). Dissecting racial bias in algorithms. Science.
7. Reimers, N. & Gurevych, I. (2019). Sentence-BERT. EMNLP.
