# Respiratory / Pneumonia Data

Place your respiratory disease CSV files here.

## Supported Dataset Types
| Type | Target Column | Example Values |
|---|---|---|
| Pneumonia | `pneumonia` / `diagnosis` | 0 / 1 |
| COVID-19 | `covid_positive` / `result` | positive / negative |
| ICU admission | `icu_admitted` | 0 / 1 |
| Mortality | `died` / `death` | 0 / 1 |

## Known Compatible Datasets (Kaggle)
- **COVID-19 Clinical Dataset** — symptoms + labs, target: `covid_positive`
- **Pneumonia Patient Records** — target: `pneumonia` (0/1)
- **ICU Respiratory Patients** — target: `icu_admitted`

## Key Features ClinIQ Recognises
`age`, `oxygen_saturation`, `spo2`, `respiratory_rate`,
`fever`, `cough`, `shortness_of_breath`, `chest_pain`,
`pcr_result`, `ct_score`, `wbc`, `crp`, `d_dimer`,
`ventilator_required`, `comorbidities`
