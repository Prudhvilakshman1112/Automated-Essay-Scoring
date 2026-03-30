# Stage 1 — Baseline: SBERT + LSTM Attention

## What This Stage Does
This is the **baseline model** — a direct replication of the SBERT + LSTM Attention architecture from **Nie (2025)**. It trains from scratch and saves the model weights for use in Stage 2.

## Architecture
```
Essay → Sentence Split → SBERT (384-dim) → LSTM (140-dim) → Attention → FC → Score
```

## Hyperparameters

| Parameter          | Value                                |
|--------------------|--------------------------------------|
| SBERT Model        | `paraphrase-MiniLM-L6-v2`            |
| Embedding Dim      | 384                                  |
| LSTM Hidden Dim    | 140                                  |
| Optimizer          | Adam                                 |
| Learning Rate      | 0.01                                 |
| Epochs             | 100                                  |
| Loss               | 0.8 × MSE − 0.2 × CosineSimilarity  |
| Train / Val / Test | 72% / 8% / 20%                       |

## How to Run
```sh
cd Stage_1
python main.py
```

## ✅ Last Run Results — 2026-02-18 23:28

### Dataset Split

| Split      |  Count |
|------------|-------:|
| Total      | 12,976 |
| Train      |  9,342 |
| Validation |  1,038 |
| Test       |  2,596 |

### Accuracy Metrics (Test Set)

| Metric                             |      Value |
|------------------------------------|------------|
| **Quadratic Weighted Kappa (QWK)** | **0.9486** |
| **Mean Squared Error (MSE)**       | **3.8802** |
| **R² Score**                       | **0.9502** |

### Training Loss Curve

| Epoch | Train Loss | Val Loss |
|------:|-----------:|---------:|
|     1 |   101.3974 |  99.7377 |
|    10 |    59.5631 |  60.7952 |
|    20 |    32.2868 |  32.5897 |
|    30 |    14.7476 |  14.7717 |
|    40 |     6.1934 |   7.1356 |
|    50 |     3.0764 |   3.9875 |
|    60 |     2.1618 |   3.9086 |
|    70 |     1.7881 |   3.5932 |
|    80 |     1.6015 |   3.5954 |
|    90 |     1.3251 |   3.6328 |
|   100 |     1.2186 |   3.7635 |

## What Is NOT Measured Here
- Length bias is **not evaluated** in Stage 1
- The model may be rewarding essay length without us knowing
- → Run **Stage 2** to discover the bias
