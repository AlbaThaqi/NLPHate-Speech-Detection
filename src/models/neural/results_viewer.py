import os
import io
import base64
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_confusion(cm, labels=("normal", "hatespeech")) -> bytes:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def save_results_html(metrics: Dict, out_dir: str = "analysis/neural", filename: str = "neural_results.html") -> str:
    os.makedirs(out_dir, exist_ok=True)
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    img_bytes = _plot_confusion(cm)
    b64 = base64.b64encode(img_bytes).decode("ascii")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Neural Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .card {{ border: 1px solid #eee; padding: 18px; border-radius: 8px; max-width: 900px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }}
    h1 {{ margin-top: 0; }}
    .metrics {{ display:flex; gap:24px; }}
    .metric {{ padding:8px; background:#fafafa; border-radius:6px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Neural Models Results</h1>
    <div class="metrics">
      <div class="metric"><strong>Accuracy:</strong> {metrics.get('accuracy', metrics.get('accuracy', 'N/A'))}</div>
      <div class="metric"><strong>Macro F1:</strong> {metrics.get('f1_macro', 'N/A')}</div>
    </div>
    <h2>Confusion Matrix</h2>
    <img src="data:image/png;base64,{b64}" alt="confusion matrix" />
  </div>
</body>
</html>"""
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path
