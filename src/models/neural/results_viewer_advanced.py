"""Advanced HTML results viewer: per-epoch curves, confusion matrices, and model comparison."""
import os
import io
import json
import base64
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plot_confusion(cm, labels=("normal", "hatespeech"), title="Confusion Matrix") -> bytes:
    cm = np.array(cm)  
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(cm[i][j])), ha="center", va="center", fontsize=12, color="black" if cm[i][j] < cm.max()/2 else "white")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _plot_epoch_curves(model_results: Dict[str, Any]) -> bytes:
    """Plot per-epoch accuracy and F1 curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for model_name, data in model_results.items():
        if isinstance(data, dict) and "history" in data:
            history = data["history"]
            epochs = [m["epoch"] for m in history]
            accs = [m["accuracy"] for m in history]
            f1s = [m["f1_macro"] for m in history]
            ax1.plot(epochs, accs, marker="o", label=model_name, linewidth=2)
            ax2.plot(epochs, f1s, marker="s", label=model_name, linewidth=2)
    
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Accuracy vs Epoch", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, max([m.get("epoch", 1) for data in model_results.values() if isinstance(data, dict) and "history" in data for m in data["history"]], default=1) + 1))
    
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Macro F1", fontsize=11)
    ax2.set_title("Macro F1 vs Epoch", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, max([m.get("epoch", 1) for data in model_results.values() if isinstance(data, dict) and "history" in data for m in data["history"]], default=1) + 1))
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _plot_final_comparison(model_results: Dict[str, Any]) -> bytes:
    """Plot final accuracy and F1 comparison across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    models = []
    accs = []
    f1s = []
    
    for model_name, data in model_results.items():
        if isinstance(data, dict) and "final" in data:
            final = data["final"]
            if "accuracy" in final and "f1_macro" in final:
                models.append(model_name.upper())
                accs.append(final["accuracy"])
                f1s.append(final["f1_macro"])
    
    if models:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][:len(models)]
        ax1.bar(models, accs, color=colors, alpha=0.7)
        ax1.set_ylabel("Accuracy", fontsize=11)
        ax1.set_title("Final Accuracy by Model", fontsize=12)
        ax1.set_ylim([0, 1.0])
        for i, v in enumerate(accs):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
        
        ax2.bar(models, f1s, color=colors, alpha=0.7)
        ax2.set_ylabel("Macro F1", fontsize=11)
        ax2.set_title("Final Macro F1 by Model", fontsize=12)
        ax2.set_ylim([0, 1.0])
        for i, v in enumerate(f1s):
            ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def save_results_html_advanced(
    model_results: Dict[str, Any],
    out_dir: str = "analysis/neural",
    filename: str = "neural_results_detailed.html"
) -> str:
    """Save advanced HTML report with epoch curves, confusion matrices, and comparison."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate plots
    epoch_curves_img = _plot_epoch_curves(model_results)
    final_comp_img = _plot_final_comparison(model_results)
    epoch_curves_b64 = base64.b64encode(epoch_curves_img).decode("ascii")
    final_comp_b64 = base64.b64encode(final_comp_img).decode("ascii")
    
    # Build confusion matrix images for each model's final epoch
    confusion_imgs = {}
    for model_name, data in model_results.items():
        if isinstance(data, dict) and "final" in data:
            final = data["final"]
            if "confusion_matrix" in final:
                cm = final["confusion_matrix"]
                img = _plot_confusion(cm, title=f"{model_name.upper()} - Final Confusion Matrix")
                confusion_imgs[model_name] = base64.b64encode(img).decode("ascii")
    
    # Build epoch details HTML
    epoch_details_html = ""
    for model_name, data in model_results.items():
        if isinstance(data, dict) and "history" in data:
            history = data["history"]
            epoch_details_html += f"<h3>{model_name.upper()}</h3>\n"
            epoch_details_html += "<table style='border-collapse:collapse; margin:16px 0;'>\n"
            epoch_details_html += "<tr style='background:#f0f0f0;'><th style='border:1px solid #ccc; padding:8px;'>Epoch</th><th style='border:1px solid #ccc; padding:8px;'>Accuracy</th><th style='border:1px solid #ccc; padding:8px;'>Macro F1</th></tr>\n"
            for m in history:
                epoch = m.get("epoch", "?")
                acc = m.get("accuracy", 0)
                f1 = m.get("f1_macro", 0)
                epoch_details_html += f"<tr><td style='border:1px solid #ccc; padding:8px;'>{epoch}</td><td style='border:1px solid #ccc; padding:8px;'>{acc:.4f}</td><td style='border:1px solid #ccc; padding:8px;'>{f1:.4f}</td></tr>\n"
            epoch_details_html += "</table>\n"
    
    # Build confusion matrix section
    confusion_html = ""
    for model_name, b64_img in confusion_imgs.items():
        confusion_html += f"<div style='margin:24px 0;'><h3>{model_name.upper()} - Final Confusion Matrix</h3><img src='data:image/png;base64,{b64_img}' style='max-width:500px;' /></div>\n"
    
    # Summary stats
    summary_html = ""
    for model_name, data in model_results.items():
        if isinstance(data, dict) and "final" in data:
            final = data["final"]
            if "accuracy" in final:
                acc = final["accuracy"]
                f1 = final["f1_macro"]
                summary_html += f"<div style='background:#f9f9f9; padding:12px; margin:8px 0; border-radius:6px;'><strong>{model_name.upper()}</strong>: Accuracy={acc:.4f}, Macro F1={f1:.4f}</div>\n"
    
    # Full HTML document
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Neural Models - Detailed Results</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: white;
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      overflow: hidden;
    }}
    .header {{
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 40px 30px;
      text-align: center;
    }}
    .header h1 {{
      font-size: 2.5em;
      margin-bottom: 10px;
    }}
    .header p {{
      font-size: 1.1em;
      opacity: 0.95;
    }}
    .content {{
      padding: 40px;
    }}
    .section {{
      margin: 32px 0;
    }}
    .section h2 {{
      font-size: 1.8em;
      color: #333;
      margin-bottom: 20px;
      border-bottom: 3px solid #667eea;
      padding-bottom: 10px;
    }}
    .section h3 {{
      font-size: 1.3em;
      color: #555;
      margin-top: 20px;
      margin-bottom: 15px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 20px;
      margin: 20px 0;
    }}
    .metric-card {{
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 20px;
      border-radius: 8px;
      text-align: center;
    }}
    .metric-card .label {{
      font-size: 0.9em;
      opacity: 0.9;
      margin-bottom: 8px;
    }}
    .metric-card .value {{
      font-size: 2em;
      font-weight: bold;
    }}
    .chart-container {{
      text-align: center;
      margin: 30px 0;
    }}
    .chart-container img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border-radius: 8px;
      overflow: hidden;
    }}
    th {{
      background: #667eea;
      color: white;
      padding: 12px;
      text-align: left;
      font-weight: 600;
    }}
    td {{
      padding: 12px;
      border-bottom: 1px solid #e0e0e0;
    }}
    tr:hover {{
      background: #f5f5f5;
    }}
    .footer {{
      background: #f9f9f9;
      padding: 20px 40px;
      text-align: center;
      font-size: 0.9em;
      color: #666;
      border-top: 1px solid #e0e0e0;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1> Neural Models - Detailed Results</h1>
      <p>Hate Speech Detection: LSTM, CNN & Transformer Comparison</p>
    </div>
    <div class="content">
      <!-- Summary Section -->
      <div class="section">
        <h2> Summary</h2>
        {summary_html}
      </div>

      <!-- Epoch Curves Section -->
      <div class="section">
        <h2> Per-Epoch Performance</h2>
        <p>Model performance across training epochs:</p>
        <div class="chart-container">
          <img src="data:image/png;base64,{epoch_curves_b64}" alt="Epoch Curves"/>
        </div>
      </div>

      <!-- Final Comparison Section -->
      <div class="section">
        <h2> Final Model Comparison</h2>
        <p>Accuracy and F1 score at final epoch:</p>
        <div class="chart-container">
          <img src="data:image/png;base64,{final_comp_b64}" alt="Final Comparison"/>
        </div>
      </div>

      <!-- Detailed Epoch Metrics -->
      <div class="section">
        <h2>Detailed Epoch Metrics</h2>
        {epoch_details_html}
      </div>

      <!-- Confusion Matrices -->
      <div class="section">
        <h2> Confusion Matrices (Final Epoch)</h2>
        {confusion_html}
      </div>

      <!-- Interpretation Section -->
      <div class="section">
        <h2> Model Interpretation & Insights</h2>
        <div style="background:#f9f9f9; padding:20px; border-left:4px solid #667eea; border-radius:4px;">
          <h3>Hate Speech Detection Insights</h3>
          <ul style="margin-left:20px; line-height:1.8;">
            <li><strong>LSTM:</strong> Bidirectional architecture captures sequential context and dependencies in hate speech patterns. Strong for understanding token-level relationships.</li>
            <li><strong>CNN:</strong> Convolutional filters extract local n-gram patterns and hate speech keywords effectively. Faster training, excellent for short texts.</li>
            <li><strong>Transformer (DistilBERT):</strong> Pre-trained contextual embeddings and attention mechanisms provide robust understanding of implicit bias and domain-specific hate cues. Best overall performance expected.</li>
          </ul>
          <h3 style="margin-top:20px;">Domain-Specific Enhancements Applied</h3>
          <ul style="margin-left:20px; line-height:1.8;">
            <li><strong>Class Weighting:</strong> Mitigates label imbalance (more normal posts than hate speech)</li>
            <li><strong>Encoder Freezing (CPU):</strong> Speeds up transformer training without sacrificing accuracy on small data</li>
            <li><strong>Full Fine-tuning (GPU):</strong> Leverages GPU to fully adapt DistilBERT to hate speech detection task</li>
          </ul>
        </div>
      </div>

      <!-- Integration Note -->
      <div class="section" style="background:#f0f7ff; padding:20px; border-radius:8px; border-left:4px solid #667eea;">
        <h2> Integration with Classical Methods</h2>
        <p>These neural models complement traditional NLP methods (TF-IDF + Logistic Regression/SVM) implemented in <code>src/hatexplain/train_classical.py</code>.</p>
        <p><strong>Classical Strengths:</strong> Interpretability (LIME), feature importance, fast training.</p>
        <p><strong>Neural Strengths:</strong> Context modeling, attention-based explanation, end-to-end learning.</p>
        <p><strong>Combined Approach:</strong> Use classical models for quick baseline and explainability; neural models for state-of-the-art performance and deep semantic understanding of hate speech cues.</p>
      </div>
    </div>
    <div class="footer">
      <p>Generated by NLP Hate Speech Detection Project | Combining Traditional & Neural Techniques</p>
      <p>Description: Create a model to detect and interpret Hate Speech Detection-related cues using both traditional and neural NLP techniques.</p>
    </div>
  </div>
</body>
</html>"""

    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path
