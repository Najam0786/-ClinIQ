# 🏥 ClinIQ — Clinical Intelligence Platform

> **Plug-in your medical data. Get clinical risk predictions instantly.**

ClinIQ is a production-ready, plug-and-play clinical ML platform built for 5 disease domains + a universal auto-mode. Drop in any structured medical dataset — clean or messy — and the system automatically profiles the data, fixes quality issues, engineers 20+ clinical features, selects the best model via cross-validation, and serves a full interactive dashboard with SHAP explainability and PDF export.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-cliniq1.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cliniq1.streamlit.app/)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What Is ClinIQ?

Like plugging in a washing machine — you put in dirty data, you get clean predictions.

**No manual preprocessing. No feature engineering scripts. No model selection loops.**
Just upload your CSV and click **Run**.

```
Upload CSV → Auto-Profile → Auto-Clean → Feature Engineering
                                       → 5-Fold CV Model Selection
                                       → Interactive Dashboard (6 tabs)
                                       → SHAP Explainability + PDF Export
```

---

## 5 Disease Modules + Universal Auto-Mode

| Module | Prediction Task | Demo Data |
|---|---|---|
| 🎗️ **Breast Cancer** | Malignant vs Benign tumour | ✅ Wisconsin dataset (569 patients) |
| 🫀 **Heart Disease** | Cardiac event risk | Upload your CSV |
| 🩸 **Diabetes** | Diabetic vs Non-diabetic | Upload your CSV |
| 🧠 **Stroke** | Stroke risk prediction | Upload your CSV |
| 🫁 **Respiratory** | Respiratory disease risk | Upload your CSV |
| 🌐 **Universal Auto** | Any binary clinical target | Any CSV — zero config |

> **Try it instantly:** Select **Breast Cancer** and click **⚡ Load & Analyse Demo** — no upload needed.

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ClinIQ.git
cd ClinIQ
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use

1. **Select a disease module** in the sidebar
2. **Upload your CSV or Excel file** — or click **⚡ Load & Analyse Demo** (Breast Cancer)
3. ClinIQ **auto-detects your target column** (override if needed)
4. Click **▶ Run ClinIQ Analysis** — everything else is automatic

### What ClinIQ Does Automatically

| Step | What Happens |
|---|---|
| **Profile** | Detects column types, missing values, duplicates, invalid values; computes a 0–100 quality score |
| **Clean** | Removes duplicates, replaces impossible negatives with NaN, imputes with median/mode |
| **Engineer** | Builds 20+ clinical features: age groups, BMI categories, BP flags, polypharmacy, lab abnormality flags, comorbidity counts |
| **Train** | Runs 5-fold CV on Random Forest, Logistic Regression, Gradient Boosting, and HistGradientBoosting; selects best by AUC |
| **Evaluate** | ROC curve, confusion matrix, per-class classification report on held-out test set |
| **Explain** | SHAP waterfall plots + feature importance ranked bar chart |
| **Export** | One-click PDF report with all results |

---

## Machine Learning Models

### Candidate Models — Auto-Selected by 5-Fold Cross-Validation

ClinIQ trains **4 models in parallel** on every run. The one with the highest mean CV AUC is automatically chosen as the winner.

| Model | Type | Strengths in Clinical Data |
|---|---|---|
| 🌲 **Random Forest** | Ensemble (bagging) | Handles mixed feature types, robust to outliers, built-in feature importance |
| 📈 **Logistic Regression** | Linear | Fast, interpretable coefficients, strong baseline for well-engineered features |
| 🚀 **Gradient Boosting** | Ensemble (boosting) | High accuracy on structured tabular data, captures non-linear interactions |
| ⚡ **HistGradientBoosting** | Ensemble (boosting) | Native missing-value handling, fast on large datasets (>10k rows) |

---

### Model Selection Pipeline

```
Raw Features
     │
     ▼
┌──────────────────────────────────────────┐
│  Preprocessing (ColumnTransformer)        │
│  • Numeric  → Median imputation + StandardScaler  │
│  • Categorical → Most-frequent imputation + OneHotEncoder │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  5-Fold Stratified Cross-Validation       │
│  All 4 models evaluated in parallel       │
│  Metric: ROC-AUC (handles class imbalance)│
└──────────────┬───────────────────────────┘
               │
               ▼
      Best model by mean CV AUC
               │
               ▼
┌──────────────────────────────────────────┐
│  Final Training on full train set         │
│  Evaluation on held-out test set (20%)    │
└──────────────────────────────────────────┘
```

---

### Evaluation Metrics

| Metric | Description | Why It Matters Clinically |
|---|---|---|
| **ROC-AUC** | Area under the ROC curve | Primary selection metric — works well with imbalanced classes |
| **Precision** | TP / (TP + FP) | How many flagged high-risk patients are truly high risk |
| **Recall (Sensitivity)** | TP / (TP + FN) | How many true high-risk patients are correctly caught |
| **F1-Score** | Harmonic mean of precision & recall | Balance between missing cases and false alarms |
| **Confusion Matrix** | TN / FP / FN / TP breakdown | Direct view of misclassification types |
| **Classification Report** | Per-class precision, recall, F1 | Full breakdown for both risk classes |

---

### Adaptive Complexity for Large Datasets

To prevent excessive training times on large datasets, ClinIQ **automatically reduces model complexity**:

| Dataset Size | Trees (RF/GBM) | CV Folds | Behaviour |
|---|---|---|---|
| < 5,000 rows | 300 | 5 | Full complexity |
| 5,000 – 20,000 rows | 150 | 3 | Reduced trees |
| > 20,000 rows | 100 | 3 | Minimal trees, fastest training |

---

### Feature Engineering (20+ Auto-Built Features)

Before model training, ClinIQ auto-engineers clinical features from raw columns:

| Feature | Source Columns | Clinical Meaning |
|---|---|---|
| `age_group` | `age` | Paediatric / Adult / Elderly risk bands |
| `bmi_category` | `bmi` | Underweight / Normal / Overweight / Obese |
| `hypertension_stage` | `systolic_bp`, `diastolic_bp` | BP severity staging |
| `hyperglycaemia` | `avg_glucose` | Flag: glucose > 126 mg/dL |
| `poor_glycaemic_control` | `avg_hba1c` | Flag: HbA1c > 7.0% |
| `anaemia` | `avg_hemoglobin` | Flag: haemoglobin below normal |
| `high_cholesterol` | `avg_cholesterol` | Flag: total cholesterol > 200 mg/dL |
| `polypharmacy` | `n_meds` | Flag: taking ≥ 5 medications |
| `high_utilisation` | `n_visits` | Flag: ≥ 5 clinical visits |
| `comorbidity_count` | disease flags | Number of co-existing chronic conditions |
| `low_adherence` | `med_adherence_score` | Flag: adherence score < 0.70 |

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| 📋 **Overview** | KPI cards, risk distribution, model comparison, ROC curve — all at a glance |
| 📊 **Data Profile** | Column summary table, missing value chart, quality score gauge |
| 🧹 **Auto-Clean Report** | Cleaning action log, cells-fixed-per-column chart |
| 🤖 **Model Results** | CV AUC comparison, confusion matrix, ROC curve, classification report |
| 🎯 **Predictions** | Per-patient risk scores with colour-coding, downloadable CSV |
| 🔍 **Feature Insights** | Top-5 driver cards, top-20 importance chart, SHAP explainability, clinical interpretation guide |
| 🩺 **Patient Predictor** | Real-time single-patient risk form → probability gauge + clinical recommendation |

---

## Screenshots

### 🏥 Main Dashboard — Landing Page
![Main Dashboard](screenshots/01_Dashboard.png)
*ClinIQ home screen — select a disease module, upload your data, and launch the analysis.*

### 📋 Overview Dashboard
![Overview Dashboard](screenshots/02_overview.png)
*KPI cards, risk distribution donut, model comparison leaderboard, and ROC curve — all in one view.*

### 📊 Data Profile
![Data Profile](screenshots/03_data_profile.png)
*Automated data quality scoring, missing value visualisation, and column-level statistics.*

### 🤖 Model Results
![Model Results](screenshots/04_model_results.png)
*5-fold CV AUC leaderboard, confusion matrix heatmap, and full classification report.*

### 🎯 Predictions
![Predictions Table](screenshots/05_predictions.png)
*Per-patient risk scores with colour-coded High / Low risk labels and downloadable CSV.*

### 🔮 SHAP Explainability
![SHAP Waterfall](screenshots/06_shap.png)
*Global mean |SHAP| summary + individual patient waterfall — exactly why the model made each decision.*

### 🩺 Patient Predictor
![Patient Predictor](screenshots/07_patient_predictor.png)
*Dynamic clinical form + real-time risk gauge + colour-coded clinical recommendation.*

---

## Project Structure

```
ClinIQ/
├── app.py                          ← Streamlit dashboard (main entry point)
│
├── core/
│   ├── profiler.py                 ← Auto-detect schema, types, quality issues
│   ├── cleaner.py                  ← Auto-impute, dedup, fix invalid values
│   ├── feature_engine.py           ← 20+ clinical feature engineering functions
│   └── model_selector.py           ← 4-model CV selection + SHAP support
│
├── modules/
│   ├── breast_cancer.py            ← Wisconsin diagnostic pipeline
│   ├── heart_disease.py            ← Cardiac risk pipeline
│   ├── diabetes.py                 ← Diabetes prediction pipeline
│   ├── stroke.py                   ← Stroke risk pipeline
│   ├── respiratory.py              ← Respiratory disease pipeline
│   └── universal.py                ← Zero-config fallback for any CSV
│
├── sample_data/
│   ├── breast_cancer/
│   │   └── breast_cancer_wisconsin.csv   ← 569 patients, 32 features
│   ├── heart_disease/
│   ├── diabetes/
│   ├── stroke/
│   └── respiratory/
│
├── screenshots/                    ← Dashboard screenshots for README
├── .github/workflows/
│   ├── ci.yml                      ← Quality gate (syntax + pytest) on every push
│   ├── cd.yml                      ← Auto-tag versioned release on main push
│   └── rollback.yml                ← One-click rollback to any previous release
├── requirements.txt
├── .gitignore
├── ROLLBACK.md                     ← Rollback & recovery guide
└── README.md
```

---

## Bring Your Own Data

ClinIQ works with any structured medical CSV or Excel file.

| Requirement | Detail |
|---|---|
| **Format** | CSV or Excel (`.csv`, `.xls`, `.xlsx`) |
| **Rows** | Minimum ~200 rows for reliable model training |
| **Target column** | Must be binary (0/1 or two distinct class labels) |
| **Missing values** | Handled automatically — no pre-cleaning needed |
| **Column names** | Any names — ClinIQ auto-maps known clinical variables |

**Clinical variables auto-recognised and enriched:**
`age`, `bmi`, `systolic_bp`, `diastolic_bp`, `avg_glucose`, `avg_hba1c`,
`avg_cholesterol`, `avg_hemoglobin`, `avg_creatinine`, `n_meds`, `n_visits`,
`med_adherence_score`, `hospitalizations_6m`, `prev_admissions_12m`, and more.

---

## Tech Stack

| Library | Role |
|---|---|
| `streamlit` | Interactive web dashboard |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, 4 models, metrics |
| `shap` | SHAP explainability for individual predictions |
| `plotly` | Interactive charts (ROC, confusion matrix, SHAP) |
| `fpdf2` | PDF report generation |
| `openpyxl` / `xlrd` | Excel file support |
| `scipy` | Statistical utilities |

---

## Roadmap

- [x] 5-disease specialist modules
- [x] Universal auto-mode for any CSV
- [x] Auto target detection with confidence scoring
- [x] Adaptive model complexity for large datasets
- [x] SHAP explainability plots (global summary + per-patient waterfall)
- [x] PDF report export (one-click downloadable clinical report)
- [x] Single patient real-time predictor (live risk form + gauge)
- [x] CI/CD pipeline with GitHub Actions + auto-rollback
- [ ] Multi-file upload (auto-join relational tables)
- [ ] REST API wrapper (FastAPI) for EMR integration
- [ ] Docker containerisation

---

## Author

**Nazmul Farooquee**
*Data Scientist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nazmul%20Farooquee-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nazmul-farooquee-mba-0b433b1b/)

Built as a showcase of production-grade clinical ML engineering.
Designed for hospitals, clinics, and health-tech teams who need fast,
interpretable risk predictions without months of data science setup.
