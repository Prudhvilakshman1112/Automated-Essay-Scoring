import pandas as pd
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import json
from model import EssayScoringModel
from process_data import prepare_data, get_sentence_embeddings
from metric import get_perform, pearson_length_correlation
from sklearn.model_selection import train_test_split
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ──────────────────────────────────────────────────────────────────
data = pd.read_csv("../dataset/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1',
                   usecols=['essay_id', 'essay_set', 'essay', 'domain1_score']).dropna(axis=1)
essays = data['essay'].tolist()
scores = data['domain1_score'].tolist()

print("Processing Data (encoding with SBERT)...")
essay_embeddings, essay_scores = prepare_data(essays, scores)

# ── Split data (same random_state=42 as Stage 1 for reproducibility) ───────────
X_trainval, X_test, y_trainval, y_test, essays_trainval, essays_test = train_test_split(
    essay_embeddings, essay_scores, essays, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, random_state=42)
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

# ── Load model (NO retraining — weights from Stage 1) ─────────────────────────
embedding_dim = 384
hidden_dim = 140
model = EssayScoringModel(embedding_dim, hidden_dim)
model.to(device)

weights_path = "../Stage_1/saved_model.pth"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"\n✅ Loaded pre-trained weights from {weights_path}")
    print("   (No retraining needed — Stage 2 reuses Stage 1 model)")
else:
    print(f"\n❌ ERROR: {weights_path} not found.")
    print("   Please run Stage_1/main.py first to generate saved_model.pth")
    sys.exit(1)

model.eval()

# ── Test Evaluation ────────────────────────────────────────────────────────────
with torch.no_grad():
    test_predictions = model(X_test).cpu().numpy().flatten()
    test_labels = y_test.cpu().numpy().flatten()

kappa, test_mse, r2 = get_perform(test_predictions, test_labels, scores)

# ── Length Bias Analysis ───────────────────────────────────────────────────────
test_word_counts = np.array([len(essay.split()) for essay in essays_test])
length_r, length_p = pearson_length_correlation(test_predictions, test_word_counts)

# ── Robustness Test (Fluff Attack) ─────────────────────────────────────────────
FILLER = (
    "This is an important point that deserves further consideration. "
    "The topic at hand is very significant and cannot be overlooked. "
    "Many experts agree that this subject requires more attention. "
    "It is widely accepted that this matter is of great importance. "
) * 12  # ~200 words of filler

best_idx = int(np.argmax(test_labels))
original_essay = essays_test[best_idx]
original_score = test_predictions[best_idx]

fluffed_essay = original_essay + " " + FILLER
with torch.no_grad():
    fluffed_emb = get_sentence_embeddings(fluffed_essay).unsqueeze(0).to(device)
    fluffed_score = model(fluffed_emb).cpu().item()

delta = fluffed_score - original_score
pct_delta = (delta / original_score) * 100
fluff_passed = delta <= 0.5

# ── Dataset split info ─────────────────────────────────────────────────────────
n_total = len(essays)
n_train = len(X_train)
n_val   = len(X_val)
n_test  = len(X_test)

# ── Bias verdict ───────────────────────────────────────────────────────────────
if abs(length_r) > 0.5:
    bias_verdict = "⚠️  HIGH BIAS DETECTED — model is rewarding essay length!"
elif abs(length_r) > 0.3:
    bias_verdict = "⚠️  MODERATE BIAS — some length dependency detected."
else:
    bias_verdict = "✅ LOW BIAS — model is relatively length-independent."

# ── Print Results ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  STAGE 2 — Biased Fusion Results (Nie 2025 + Doewes 2026 Metrics)")
print(f"{'='*50}")
print(f"\n  📊 Accuracy Metrics (Test Set):")
print(f"     Quadratic Weighted Kappa (QWK) : {kappa:.4f}")
print(f"     Mean Squared Error (MSE)       : {test_mse:.4f}")
print(f"     R² Score                       : {r2:.4f}")
print(f"\n  📂 Dataset Split:")
print(f"     Total essays   : {n_total}")
print(f"     Train set      : {n_train}")
print(f"     Validation set : {n_val}")
print(f"     Test set       : {n_test}")
print(f"\n  📏 Length Bias Analysis (Doewes 2026):")
print(f"     Pearson r (score vs word count): {length_r:.4f}  (p={length_p:.4e})")
print(f"     {bias_verdict}")
print(f"\n  🧪 Robustness Test (Fluff Attack):")
print(f"     Original essay word count : {len(original_essay.split())}")
print(f"     Fluffed essay word count  : {len(fluffed_essay.split())}")
print(f"     Original predicted score  : {original_score:.4f}")
print(f"     Score after fluff added   : {fluffed_score:.4f}")
print(f"     Score change (Δ)          : {delta:+.4f}")
print(f"     Score change (% Δ)        : {pct_delta:+.2f}%")
if fluff_passed:
    print(f"     ✅ PASSED — Score within tolerance (no inflation from filler)")
else:
    print(f"     ❌ FAILED — Score increased by {delta:.4f}. Model rewards filler text!")
print(f"\n{'='*50}")
print("\n  ➡️  Run Stage_3/main.py to see the DOL fix.")

# ── Save results to shared JSON (used by Stage 3 for comparison table) ─────────
results = {
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "qwk": round(float(kappa), 4),
    "mse": round(float(test_mse), 4),
    "r2": round(float(r2), 4),
    "length_r": round(float(length_r), 4),
    "length_p": float(length_p),
    "fluff_passed": fluff_passed,
    "fluff_delta": round(float(delta), 4),
    "fluff_pct_delta": round(float(pct_delta), 2),
    "original_wc": len(original_essay.split()),
    "fluffed_wc": len(fluffed_essay.split()),
    "original_score": round(float(original_score), 4),
    "fluffed_score": round(float(fluffed_score), 4),
}
with open("stage2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to stage2_results.json (used by Stage 3 for comparison table)")

# ── Auto-update Stage 2 README ─────────────────────────────────────────────────
run_date = results["run_date"]
fluff_status = "✅ PASSED" if fluff_passed else "❌ FAILED"
fluff_detail = f"Δ = {delta:+.4f} (% Δ = {pct_delta:+.2f}%, within tolerance)" if fluff_passed else f"Score rose by {delta:+.4f} ({pct_delta:+.2f}%)"

readme_content = f"""# Stage 2 — Biased Fusion: Nie (2025) + Doewes (2026) Metrics

## What This Stage Does
This stage **does not retrain the model**. It loads the weights saved in Stage 1 and applies the evaluation framework from **Doewes (2026)** to expose a critical flaw: the model may be biased toward essay length.

## Key Additions Over Stage 1
| Addition | File | Purpose |
|----------|------|---------|
| `pearson_length_correlation()` | `metric.py` | Measures Pearson r between predicted scores and word counts |
| Length Bias Report | `main.py` | Prints r value and flags if bias is high |
| Fluff Robustness Test | `main.py` | Appends ~200 filler words to best essay, checks if score rises |

## How to Run
> ⚠️ **Run Stage 1 first** to generate `saved_model.pth`

```sh
cd Stage_2
python main.py
```

## ✅ Last Run Results — {run_date}

### Dataset Split
| Split | Count |
|-------|-------|
| Total | {n_total} |
| Train | {n_train} |
| Validation | {n_val} |
| Test | {n_test} |

### Accuracy Metrics (Test Set) — Same Model as Stage 1
| Metric | Value |
|--------|-------|
| **Quadratic Weighted Kappa (QWK)** | **{kappa:.4f}** |
| **Mean Squared Error (MSE)** | **{test_mse:.4f}** |
| **R² Score** | **{r2:.4f}** |

> These are identical to Stage 1 — Stage 2 does not retrain, it only measures bias.

### Length Bias Analysis (Doewes 2026)
| Metric | Value |
|--------|-------|
| **Pearson r (score vs word count)** | **{length_r:.4f}** |
| **p-value** | **{length_p:.4e}** |
| **Verdict** | {bias_verdict} |

### Robustness Test (Fluff Attack)
| Item | Value |
|------|-------|
| Original essay word count | {len(original_essay.split())} |
| Fluffed essay word count | {len(fluffed_essay.split())} |
| Original predicted score | {original_score:.4f} |
| Score after fluff added | {fluffed_score:.4f} |
| Score change (Δ) | {delta:+.4f} |
| Score change (% Δ) | {pct_delta:+.2f}% |
| **Result** | **{fluff_status} — {fluff_detail}** |

## The Research Finding
- **High R²** proves the model is *accurate*
- **Pearson r = {length_r:.4f}** {"proves the model is *biased* — it rewards length, not quality" if abs(length_r) > 0.5 else "shows moderate length dependency"}
- → Run **Stage 3** to fix this with the DOL filter
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("✅ Stage 2 README.md updated with latest results.")
