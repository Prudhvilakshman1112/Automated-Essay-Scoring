# Stage 2 — Biased Fusion: Nie (2025) + Doewes (2026) Metrics

## What This Stage Does
This stage **does not retrain the model**. It loads the weights saved in Stage 1 and applies the evaluation framework from **Doewes (2026)** to expose a critical flaw: the model may be biased toward essay length.

## Key Additions Over Stage 1

| Addition                          | File        | Purpose                                                   |
|-----------------------------------|-------------|-----------------------------------------------------------|
| `pearson_length_correlation()`    | `metric.py` | Measures Pearson r between predicted scores and word counts |
| Length Bias Report                | `main.py`   | Prints r value and flags if bias is high                  |
| Fluff Robustness Test             | `main.py`   | Appends ~200 filler words to best essay, checks if score rises |

## How to Run
> ⚠️ **Run Stage 1 first** to generate `saved_model.pth`

```sh
cd Stage_2
python main.py
```

## ✅ Last Run Results — 2026-02-18 23:41

### Dataset Split

| Split      |  Count |
|------------|-------:|
| Total      | 12,976 |
| Train      |  9,342 |
| Validation |  1,038 |
| Test       |  2,596 |

### Accuracy Metrics (Test Set) — Same Model as Stage 1

| Metric                             |      Value |
|------------------------------------|------------|
| **Quadratic Weighted Kappa (QWK)** | **0.9486** |
| **Mean Squared Error (MSE)**       | **3.8802** |
| **R² Score**                       | **0.9502** |

> These are identical to Stage 1 — Stage 2 does not retrain, it only measures bias.

### Length Bias Analysis (Doewes 2026)

| Metric                            | Value                                             |
|-----------------------------------|---------------------------------------------------|
| **Pearson r (score vs word count)** | **0.5876**                                      |
| **p-value**                       | **7.2020 × 10⁻²⁴¹**                              |
| **Verdict**                       | ⚠️ HIGH BIAS DETECTED — model is rewarding essay length! |

### Robustness Test (Fluff Attack)

| Item                       | Value                                       |
|----------------------------|---------------------------------------------|
| Original essay word count  | 832                                         |
| Fluffed essay word count   | 1,312                                       |
| Original predicted score   | 8.0629                                      |
| Score after fluff added    | 7.9472                                      |
| Score change (Δ)           | −0.1157                                     |
| **Result**                 | **✅ PASSED — Δ = −0.1157 (within tolerance)** |

## The Research Finding
- **High R²** proves the model is *accurate*
- **Pearson r = 0.5876** proves the model is *biased* — it rewards length, not quality
- → Run **Stage 3** to fix this with the DOL filter
