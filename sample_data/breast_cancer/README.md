# Breast Cancer Data

Place your breast cancer CSV files here.

## Supported Dataset Types
| Type | Target Column | Example Values |
|---|---|---|
| Diagnosis | `diagnosis` | M / B (Malignant / Benign) |
| Staging | `stage` | Early / Late, 0 / 1 |
| Survival | `survived` | 0 / 1 |
| Recurrence | `recurrence` | yes / no |

## Known Compatible Datasets (Kaggle)
- **Breast Cancer Wisconsin (Diagnostic)** — 30 morphology features, target: `diagnosis` (M/B)
- **Breast Cancer Staging** — clinical stage features, target: `stage`
- **METABRIC** — gene expression + clinical, target: `vital_status`

## Key Features ClinIQ Recognises
`radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`,
`smoothness_mean`, `compactness_mean`, `concavity_mean`,
`concave_points_mean`, `symmetry_mean`, `fractal_dimension_mean`,
`radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`,
`tumor_size`, `lymph_nodes`, `hormone_receptor`, `her2_status`, `grade`
