# Triagegeist: Multi-Modal Undertriage Detection via Calibrated Ensemble Learning

**Author**: Ashok Pukkalla  
**Track**: Triagegeist: AI in Emergency Triage  
**Notebook**: [Triagegeist: Multi-Modal Undertriage Detection](https://www.kaggle.com/code/ashokpukkalla/triagegeist-multi-modal-undertriage-detection)  
**Project Link**: [GitHub Repository](https://github.com/ashokpukkalla/triagegeist)

---

## Clinical Problem Statement

Emergency department triage determines the order and urgency of patient care. Among approximately 130 million annual U.S. ED visits (CDC NHAMCS 2022), triage nurses assign Emergency Severity Index (ESI) acuity levels 1 through 5 under extreme cognitive load, with incomplete information, in environments that are chronically understaffed.

The consequences of triage errors are clinically asymmetric. Overtriage (assigning higher acuity than warranted) wastes resources but preserves patient safety. Undertriage (assigning lower acuity) delays care, causes adverse outcomes, and contributes to preventable mortality. Published literature documents undertriage rates of 5 to 20 percent (Farrohknia et al., 2011), with ESI inter-rater reliability measured at kappa = 0.60 to 0.80 (Gilboy et al., 2012). Elderly patients, atypical presenters, and linguistic minorities are disproportionately affected (Obermeyer et al., 2019).

This asymmetry defines our entire approach. Rather than treating triage as a balanced five-class classification problem, we frame it as an undertriage safety problem. We build a calibrated decision-support system that catches high-acuity patients at risk of dangerous delay, while providing transparent, explainable predictions that clinicians can trust.

---

## Methodology

### Data Integration

We integrate all three competition data sources on patient_id to create a comprehensive multi-modal patient representation:

- **train.csv** (80,000 patients): 40 structured features including vitals (heart rate, blood pressure, SpO2, temperature, respiratory rate, GCS), demographics (age, sex), temporal patterns (arrival hour, day, season), arrival context (mode, transport origin, mental status), and pre-computed clinical scores (NEWS2, shock index, MAP).
- **chief_complaints.csv** (100,000 records): Free-text chief complaint narratives providing rich clinical context that structured fields cannot capture. A patient reporting "worst headache of my life" carries different clinical urgency than "headache for 3 days."
- **patient_history.csv** (100,000 records): 25 binary comorbidity flags (heart failure, COPD, diabetes, immunosuppression, malignancy, etc.) plus medication counts and prior ED utilization.

### Five Novel Components

**1. Four-Model Calibrated Ensemble**

We combine four complementary learning paradigms via probability-weighted soft voting:

- LightGBM (histogram-based gradient boosting): fast training, native missing value handling, strong regularization
- XGBoost (exact/approximate gradient boosting): different regularization approach provides ensemble diversity
- MLP Neural Network (256-128-64 architecture): captures non-linear feature interactions that tree-based models may miss
- Logistic Regression (L2-penalized, multinomial): well-calibrated linear baseline and diversity contributor

Ensemble weights are proportional to each model's macro-F1 score on 3-fold stratified cross-validation. This soft voting approach produces better-calibrated probabilities than hard voting, which is critical for threshold-based clinical workflows where a physician needs to know not just the predicted ESI level but how confident the system is.

**2. Missingness-as-Clinical-Signal (Novel)**

Our key methodological innovation exploits a clinical observation that machine learning pipelines typically discard: in real EDs, data is not Missing At Random. A triage nurse evaluating a sprained ankle does not typically measure SpO2, GCS, or respiratory rate. The absence of a measurement is itself a clinical judgment about patient acuity.

We formalize this with three complementary features:
- Per-vital binary missingness indicators (8 flags)
- Total missing vital count
- Missingness entropy: the Shannon entropy of the binary missingness vector. High entropy indicates selective documentation (some vitals taken, others skipped); zero entropy indicates either a complete workup or no vitals at all.

Additionally, the ambulance-missing interaction captures an anomalous pattern: patients arriving by ambulance almost always receive complete vital sign workups. When an ambulance patient has missing vitals, something unusual occurred during transport, and this anomaly is predictive.

Statistical validation shows a strong correlation between missing vital count and ESI level (Kruskal-Wallis p < 0.001), confirming that missingness is a legitimate predictive signal, not noise.

**3. Sentence-BERT NLP Pipeline**

Chief complaints are encoded using the all-MiniLM-L6-v2 sentence transformer, producing 384-dimensional dense semantic embeddings. This approach captures three critical clinical dimensions that bag-of-words methods miss:

- Symptom severity: "mild headache" versus "worst headache of my life" (thunderclap headache suggesting subarachnoid hemorrhage)
- Anatomical specificity: "chest pain" versus "sharp left-sided chest pain on inspiration" (pleuritic versus cardiac)
- Clinical urgency cues: "can't breathe" versus "short of breath with exertion" (acute versus chronic)

A TF-IDF fallback (300 features, sublinear TF, bigrams) is implemented for environments without sentence-transformers. Discriminative term analysis confirms that high-acuity complaints cluster around cardiac, neurological, and respiratory terminology while low-acuity clusters around musculoskeletal and minor injury terms.

**4. Undertriage Safety Net**

A binary classifier (ESI 1-2 versus ESI 3-5) with the decision threshold tuned for 95 percent or greater sensitivity. This operates as a clinical "second opinion": when the safety net fires, it signals the triage nurse that a patient may be more acutely ill than assigned. We deliberately accept lower specificity (more false alarms) to minimize missed critical patients, consistent with clinical risk management principles.

**5. Interactive Clinical Decision Support Interface (Novel)**

The notebook includes an interactive ipywidgets-based triage prediction interface that demonstrates how the system would function in a real clinical environment. Clinicians can adjust patient parameters (heart rate, blood pressure, SpO2, temperature, respiratory rate, GCS, age, demographics) via sliders and dropdown menus, and instantly see:
- The predicted ESI level with confidence percentage
- The full class probability distribution with color-coded bars
- Safety net alerts when high-acuity probability exceeds 30%

This interactive demonstration bridges the gap between offline model evaluation and real-time clinical decision support, showing how a trained ensemble could be deployed as a bedside second-opinion tool. The auto-prediction on default values provides immediate visual feedback on model behavior.

### Feature Engineering (~490 features)

| Category | Count | Source | Innovation |
|----------|-------|--------|------------|
| Raw Vitals | 10 | train.csv | All available physiological measurements |
| Clinical Flags | 18 | Computed | Expert thresholds: tachycardia (>100), hypoxia (<94%), hypothermia (<35°C) |
| Computed Scores | 6 | Computed + dataset | MEWS, Shock Index, NEWS2, MAP, pulse pressure, BMI |
| Demographics and Context | ~25 | train.csv | Cyclical temporal encoding, transport origin risk, mental status |
| Comorbidity Profile | ~30 | patient_history.csv | 25 flags + burden score + cardiopulmonary/immunocompromised combinations |
| Missingness Signature | 11 | All sources | Entropy + per-vital flags + ambulance interaction |
| NLP Embeddings | 384 | chief_complaints.csv | Sentence-BERT dense semantic vectors |
| Vital Interactions | 3 | Computed | HR x SBP, age x HR, SpO2 x RR interaction terms |

### Explainability

SHAP TreeExplainer provides both global feature importance (which categories drive predictions overall) and local explanations (why a specific ESI was assigned to a specific patient). Feature importance is grouped by clinical category (Vitals, NLP, Comorbidities, Missingness, Demographics) to communicate results in terms that clinicians understand, not raw feature names.

### Statistical Rigor

- 95% bootstrap confidence intervals (1000 resamples) on all reported metrics
- McNemar's test with continuity correction to verify ensemble improvement over individual models is statistically significant
- Failure mode analysis identifying high-confidence errors, ESI boundary confusion patterns, and undertriage risk factors

---

## Results

### Synthetic Data: Multi-Class ESI Prediction (3-fold Stratified CV)

On the competition's synthetic dataset (80,000 patients), the 4-model ensemble achieves near-perfect performance:

| Metric | Value |
|--------|-------|
| Accuracy | 99.97% |
| F1 (macro) | 99.91% |
| QWK | 0.9998 |

McNemar's tests confirm statistically significant ensemble improvement over individual models. The safety net binary classifier achieves ROC AUC > 0.999 with 95%+ sensitivity at acceptable specificity.

However, this performance is suspiciously high. When all models achieve >99.9% accuracy, the dataset likely contains feature combinations that deterministically encode the target — a well-known property of synthetic data generators that preserve statistical relationships too faithfully. This motivates our real-world validation.

### Real-World Validation: NHAMCS 2022 Clinical Data

To stress-test our methodology on genuine clinical data, we apply the identical pipeline — same feature engineering, same model architectures, same hyperparameters — to the CDC's National Hospital Ambulatory Medical Care Survey (NHAMCS) 2022 dataset: 10,207 real emergency department visits with physician-assigned ESI triage levels.

**Key constraint**: NHAMCS contains no free-text chief complaints, so the NLP component (384 features) is unavailable. The model operates on 72 structured features only, versus 494 in the synthetic pipeline.

#### Individual Model and Ensemble Results (NHAMCS)

| Model | Accuracy | F1 (macro) | QWK |
|-------|----------|------------|-----|
| LightGBM | 53.95% | 0.261 | 0.188 |
| XGBoost | 54.14% | 0.258 | 0.196 |
| MLP | 53.96% | 0.252 | 0.215 |
| Logistic Regression | 29.64% | 0.243 | 0.235 |
| **Ensemble** | **54.85%** | **0.280** | **0.239** |

The ensemble consistently outperforms all individual models, validating the diversity benefit of multi-model combination even on real data where overall performance is modest.

#### Per-Class Accuracy (NHAMCS)

| ESI Level | Accuracy | Support |
|-----------|----------|---------|
| ESI 1 (Resuscitation) | 10.8% | 139 |
| ESI 2 (Emergent) | 11.4% | 1,595 |
| ESI 3 (Urgent) | 87.2% | 5,340 |
| ESI 4 (Less Urgent) | 26.3% | 2,826 |
| ESI 5 (Non-Urgent) | 0.3% | 307 |

The model strongly biases toward ESI 3 (the majority class comprising 52% of visits), achieving 87% recall there but severely under-detecting extremes. This mirrors the clinical reality of ESI distribution in U.S. emergency departments.

#### Performance Comparison: Synthetic vs. Real

| Dataset | N | Accuracy | F1 (macro) | QWK |
|---------|---|----------|------------|-----|
| Synthetic (Competition) | 80,000 | 99.97% | 99.91% | 0.9998 |
| NHAMCS 2022 (Real) | 10,207 | 54.85% | 28.00% | 0.2388 |

The **45.1 percentage point accuracy gap** quantifies the degree of synthetic over-specification. The real-world QWK of 0.239 is below published ESI inter-rater reliability (κ = 0.60–0.80, Gilboy et al., 2012), suggesting that structured features alone — without free-text chief complaints and direct clinical observation — are insufficient for accurate triage prediction on real data.

#### Undertriage on Real Data

- Overall undertriage rate: **20.27%** (2,069 / 10,207 patients)
- Dangerous undertriage (≥2 levels): **1.83%** (187 / 10,207)
- ESI 2 patients misclassified as ESI 3: **1,320 patients (82.8%)**
- ESI 3 patients misclassified as ESI 4: **537 patients (10.1%)**

The 82.8% misclassification rate for ESI 2→3 is clinically alarming and highlights the challenge: without NLP signals (the patient's own words describing symptom severity), the model cannot distinguish "emergent" from "urgent" using structured vitals alone.

### Ablation Study: Feature Group Contributions (NHAMCS)

To quantify which feature groups contribute most to real-world performance, we systematically remove each category and measure accuracy degradation:

| Configuration | Features | Accuracy | Δ Accuracy |
|--------------|----------|----------|------------|
| Full model | 72 | 45.99% | — |
| − Vitals | 65 | 43.63% | **−2.36%** |
| − Demographics | 61 | 44.29% | −1.69% |
| − Comorbidities | 49 | 44.50% | −1.49% |
| − Missingness | 62 | 45.01% | −0.98% |
| − Interactions | 68 | 45.24% | −0.75% |
| − Clinical Flags | 57 | 45.71% | −0.27% |

**Vitals are the most impactful feature group** (Δ = −2.36%), followed by demographics and comorbidities. Notably, missingness features contribute nearly 1 percentage point — validating our hypothesis that documentation patterns carry genuine clinical signal even on real data. Clinical flags (threshold-based binary indicators) have minimal independent contribution, suggesting their information overlaps with raw vital values.

### Failure Mode Analysis

High-confidence errors (model wrong with >80% confidence) are enumerated and profiled. Undertriage risk factor analysis identifies which patient features correlate with being under-triaged. The fairness audit across demographic dimensions confirms no subgroup accuracy disparity exceeding 3 percentage points from the population mean.

---

## Limitations and Critical Reflection

1. **Synthetic over-specification confirmed**: Our NHAMCS validation demonstrates a 45.1 percentage point accuracy gap between synthetic (99.97%) and real (54.85%) data. The competition dataset contains feature combinations that deterministically encode ESI — a well-documented property of synthetic data generators. Real-world performance is the honest benchmark.

2. **NLP dependency**: The 82.8% misclassification rate for ESI 2→3 on NHAMCS (which lacks free-text) strongly suggests that chief complaint semantics are essential for distinguishing emergent from urgent patients. Structured vitals alone are insufficient for clinically meaningful triage.

3. **No outcome validation**: Without 30-day mortality, ICU admission, or readmission data, the clinical impact of predictions remains theoretical. Undertriage is defined relative to assigned ESI, not patient outcomes.

4. **Temporal dynamics absent**: We model arrival-time features only. Real ED patients deteriorate over time, and a triage tool that does not incorporate serial reassessment data misses this critical dimension.

5. **Class imbalance on real data**: NHAMCS ESI distribution is heavily skewed (52% ESI 3), causing the model to bias toward the majority class. Techniques like class-weighted loss or oversampling could improve minority-class detection.

6. **Language and cultural bias**: Sentence-BERT embeddings are trained primarily on English text. Performance on non-English chief complaints or culturally specific symptom descriptions may differ.

---

## Reproducibility

- Full code in a single Kaggle Notebook that runs end-to-end without errors (27.7 minutes runtime)
- GitHub repository: [github.com/ashokpukkalla/triagegeist](https://github.com/ashokpukkalla/triagegeist) — complete source code, documentation, and requirements
- Interactive clinical decision support demo embedded in the notebook (ipywidgets-based)
- Random seed 42 set for all stochastic operations (numpy, sklearn, LightGBM, XGBoost, MLP)
- Competition data: train.csv, chief_complaints.csv, patient_history.csv, test.csv
- External validation data: CDC NHAMCS 2022 (publicly available via Kaggle dataset radhikaaaaaaa/nhamcs2022)
- Dependencies: lightgbm, xgboost, scikit-learn, sentence-transformers, shap, scipy, matplotlib, seaborn, pandas, numpy

---

## Clinical Implications

This work demonstrates two key findings. First, a multi-modal calibrated ensemble achieves near-perfect synthetic performance, confirming the methodology is technically sound. Second, and more importantly, real-world validation on NHAMCS 2022 reveals the honest limitations: structured features alone achieve 54.85% accuracy (QWK 0.239), far below human inter-rater reliability — proving that NLP-encoded chief complaints and direct clinical observation are indispensable for accurate triage.

The ablation study quantifies this: vitals contribute most (−2.36% when removed), but even all structured features combined cannot replicate what a triage nurse learns from hearing a patient describe their symptoms. This finding has direct implications for clinical AI deployment:

1. **Multi-modal data is non-negotiable**: Systems relying solely on structured EHR fields will underperform. Chief complaint NLP and documentation patterns must be integrated.

2. **Decision support, not automation**: At 54.85% real-world accuracy, this system cannot replace triage nurses. It can flag potential undertriage for reassessment — a "second pair of eyes" that catches the 20.27% of patients receiving lower-than-warranted acuity.

3. **Conservative thresholds**: Accept higher false-positive alerting rates to minimize missed critical patients. The 82.8% ESI 2→3 misclassification rate on real data underscores the stakes of undertriage.

4. **Continuous validation**: The 45-point synthetic-to-real accuracy gap is a cautionary result for all competition-based clinical ML. Real-world validation on external datasets must become standard practice.

---

## References

1. Gilboy, N., et al. (2012). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4. AHRQ.
2. Farrohknia, N., et al. (2011). Emergency department triage scales and their components: A systematic review. Scandinavian Journal of Trauma.
3. Levin, S., et al. (2018). Machine-learning-based electronic triage more accurately differentiates patients. Annals of Emergency Medicine.
4. Lundberg, S. & Lee, S.I. (2017). A unified approach to interpreting model predictions. NeurIPS.
5. Klug, M., et al. (2020). A gradient boosting ML model for predicting early mortality in the ED. Academic Emergency Medicine.
6. Obermeyer, Z., et al. (2019). Dissecting racial bias in an algorithm used to manage population health. Science.
7. FDA (2021). AI/ML-Based Software as a Medical Device Action Plan.
8. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
9. Christ, M., et al. (2010). Modern triage in the emergency department. Deutsches Ärzteblatt International.
10. Dugas, A.F., et al. (2016). An electronic emergency triage system to improve patient distribution. J. Emergency Medicine.
