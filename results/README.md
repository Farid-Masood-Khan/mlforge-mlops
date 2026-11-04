# MLForge — Measured Holdout Metrics

All numbers below were measured directly on the held-out test split (15% stratified, seed=42) from dataset hash `5de3ddc9f589b369e71eae2a4dbcb5a96615910b1093c326f0ea57e3f9870571`.

| Model / Run | Run ID | Val PR-AUC | Holdout PR-AUC | Holdout ROC-AUC | Holdout F1 | Decision Threshold |
|---|---|---|---|---|---|---|
| `baseline-logistic-regression` | `4bc57285` | 0.4973 | 0.4945 | 0.6705 | 0.5547 | 0.28 |
| `hgb-lr0.05-leaf15-min20` | `0d7a5d71` | 0.4920 | 0.4835 | 0.6570 | 0.5668 | 0.28 |
| `hgb-lr0.05-leaf15-min40` | `cc4bd279` | 0.5002 | 0.4931 | 0.6603 | 0.5654 | 0.31 |
| `hgb-lr0.05-leaf31-min20` | `bdca87c6` | 0.4839 | 0.4782 | 0.6528 | 0.5618 | 0.29 |
| `hgb-lr0.05-leaf31-min40` | `5ec7b95c` | 0.5025 | 0.4757 | 0.6555 | 0.5671 | 0.29 |
| `hgb-lr0.1-leaf15-min20` | `74516052` | 0.4785 | 0.4752 | 0.6503 | 0.5643 | 0.30 |
| `hgb-lr0.1-leaf15-min40` | `0a7b18a8` | 0.4871 | 0.4664 | 0.6499 | 0.5544 | 0.31 |
| `hgb-lr0.1-leaf31-min20` | `34fc1a16` | 0.4844 | 0.4676 | 0.6425 | 0.5415 | 0.33 |
| `hgb-lr0.1-leaf31-min40` | `313f3246` | 0.4931 | 0.4772 | 0.6518 | 0.5410 | 0.32 |

## Registry Status
- **Model Name:** `mlforge-churn`
- **Active Champion Run:** `5ec7b95c4482450bb750bbf065513c44`
- **Assigned Alias:** `challenger`
- **Promotion Audit:** Version 2 assigned to 'challenger'. PR-AUC 0.4757 did not beat champion v1 (0.4757 + 0.002).
