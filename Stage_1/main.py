import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import EssayScoringModel
from process_data import prepare_data
from metric import get_perform
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ──────────────────────────────────────────────────────────────────
data = pd.read_csv("../dataset/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1',
                   usecols=['essay_id', 'essay_set', 'essay', 'domain1_score']).dropna(axis=1)
essays = data['essay'].tolist()
scores = data['domain1_score'].tolist()

print("Processing Data (encoding with SBERT)...")
essay_embeddings, essay_scores = prepare_data(essays, scores)

# ── Split data ─────────────────────────────────────────────────────────────────
X_trainval, X_test, y_trainval, y_test = train_test_split(
    essay_embeddings, essay_scores, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, random_state=42)
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

# ── Model ──────────────────────────────────────────────────────────────────────
embedding_dim = 384
hidden_dim = 140
model = EssayScoringModel(embedding_dim, hidden_dim)
model.to(device)

# LR=0.01 — same as Stage 3 for consistent convergence speed
optimizer = optim.Adam(model.parameters(), lr=0.01)
mse = nn.MSELoss()

# ── Training ───────────────────────────────────────────────────────────────────
num_epochs = 100
print(f"\nTraining for {num_epochs} epochs (LR=0.01)...")

epoch_log = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = 0.8 * mse(predictions, y_train) - 0.2 * F.cosine_similarity(predictions, y_train, dim=0).mean()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = 0.8 * mse(val_predictions, y_val) - 0.2 * F.cosine_similarity(val_predictions, y_val, dim=0).mean()

    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}')
    if (epoch + 1) in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        epoch_log.append((epoch + 1, loss.item(), val_loss.item()))

# ── Save model weights ─────────────────────────────────────────────────────────
torch.save(model.state_dict(), "saved_model.pth")
print("\n✅ Model weights saved to saved_model.pth")

# ── Test Evaluation ────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).cpu().numpy().flatten()
    test_labels = y_test.cpu().numpy().flatten()

kappa, test_mse, r2 = get_perform(test_predictions, test_labels, scores)

# ── Dataset split info ─────────────────────────────────────────────────────────
n_total = len(essays)
n_train = len(X_train)
n_val   = len(X_val)
n_test  = len(X_test)

# ── Print Results ──────────────────────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"  STAGE 1 — Baseline Results")
print(f"{'='*45}")
print(f"\n  📊 Accuracy Metrics (Test Set):")
print(f"     Quadratic Weighted Kappa (QWK) : {kappa:.4f}")
print(f"     Mean Squared Error (MSE)       : {test_mse:.4f}")
print(f"     R² Score                       : {r2:.4f}")
print(f"\n  📂 Dataset Split:")
print(f"     Total essays   : {n_total}")
print(f"     Train set      : {n_train}")
print(f"     Validation set : {n_val}")
print(f"     Test set       : {n_test}")
print(f"\n  ⚙️  Hyperparameters:")
print(f"     Epochs         : {num_epochs}")
print(f"     Learning Rate  : 0.01")
print(f"     LSTM Hidden    : {hidden_dim}")
print(f"     Embedding Dim  : {embedding_dim}")
print(f"     Loss           : 0.8×MSE − 0.2×CosineSim")
print(f"\n  ℹ️  Length bias is NOT measured in Stage 1.")
print(f"  Run Stage_2/main.py to see the bias analysis.")
print(f"{'='*45}")

# ── Save results to shared JSON (used by Stage 2 and Stage 3 for comparison table) ──
results = {
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "qwk": round(float(kappa), 4),
    "mse": round(float(test_mse), 4),
    "r2": round(float(r2), 4),
    "n_total": n_total,
    "n_train": n_train,
    "n_val": n_val,
    "n_test": n_test,
    "epochs": num_epochs,
    "lr": 0.01,
    "epoch_log": epoch_log
}
with open("stage1_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to stage1_results.json (used by Stage 2 & 3 for comparison table)")

# ── Auto-update Stage 1 README ─────────────────────────────────────────────────
run_date = results["run_date"]
readme_content = f"""# Stage 1 — Baseline: SBERT + LSTM Attention

## What This Stage Does
This is the **baseline model** — a direct replication of the SBERT + LSTM Attention architecture from **Nie (2025)**. It trains from scratch and saves the model weights for use in Stage 2.

## Architecture
```
Essay → Sentence Split → SBERT (384-dim) → LSTM (140-dim) → Attention → FC → Score
```

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| SBERT Model | `paraphrase-MiniLM-L6-v2` |
| Embedding Dim | 384 |
| LSTM Hidden Dim | 140 |
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Epochs | {num_epochs} |
| Loss | 0.8 × MSE − 0.2 × CosineSimilarity |
| Train / Val / Test | 72% / 8% / 20% |

## How to Run
```sh
cd Stage_1
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

### Accuracy Metrics (Test Set)
| Metric | Value |
|--------|-------|
| **Quadratic Weighted Kappa (QWK)** | **{kappa:.4f}** |
| **Mean Squared Error (MSE)** | **{test_mse:.4f}** |
| **R² Score** | **{r2:.4f}** |

### Training Loss Curve
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
"""
for ep, tl, vl in epoch_log:
    readme_content += f"| {ep} | {tl:.4f} | {vl:.4f} |\n"

readme_content += """
## What Is NOT Measured Here
- Length bias is **not evaluated** in Stage 1
- The model may be rewarding essay length without us knowing
- → Run **Stage 2** to discover the bias
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("✅ Stage 1 README.md updated with latest results.")
