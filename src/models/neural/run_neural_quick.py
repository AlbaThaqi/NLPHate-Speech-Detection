"""Quick runner (package-local) to train neural baselines and save results.

Placed inside `src.models.neural` so it can use package-relative imports.
Trains LSTM, CNN, and Transformer (with GPU/CPU auto-detection) for 3 epochs each.
Saves per-epoch history and generates detailed HTML visualization.
"""
import json
import os
from .train_neural import train_lstm, train_cnn, train_transformer
from .results_viewer import save_results_html
from .results_viewer_advanced import save_results_html_advanced
from .data_loader import load_hatexplain_dataset


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "analysis", "neural")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

results = {}

try:
    train, dev = load_hatexplain_dataset()
    print("Dataset loaded:", len(train), "train", len(dev), "dev")
except Exception as e:
    print("Failed to load dataset:", e)
    train = dev = None

try:
    print("Training LSTM (3 epochs)...")
    metrics_lstm = train_lstm(train_examples=train, dev_examples=dev, epochs=3, batch_size=128, save_path=os.path.join("models", "neural", "lstm.pt"))
    results['lstm'] = metrics_lstm
    print("LSTM done")
except Exception as e:
    print("LSTM failed:", e)
    results['lstm'] = {"error": str(e)}

try:
    print("Training CNN (3 epochs)...")
    metrics_cnn = train_cnn(train_examples=train, dev_examples=dev, epochs=3, batch_size=128, save_path=os.path.join("models", "neural", "cnn.pt"))
    results['cnn'] = metrics_cnn
    print("CNN done")
except Exception as e:
    print("CNN failed:", e)
    results['cnn'] = {"error": str(e)}

try:
    import torch
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA detected: running transformer full fine-tune on GPU (3 epochs).")
        # on GPU, prefer fine-tuning full model with larger seq length and no subsample
        metrics_tr = train_transformer(
            train_examples=train,
            dev_examples=dev,
            epochs=3,
            batch_size=16,
            save_path=os.path.join("models", "neural", "transformer"),
            freeze_encoder=False,
            max_length=128,
            subset_size=None,
        )
    else:
        print("No CUDA: running transformer with frozen encoder + small subset (3 epochs, fast on CPU).")
        metrics_tr = train_transformer(
            train_examples=train,
            dev_examples=dev,
            epochs=3,
            batch_size=32,
            save_path=os.path.join("models", "neural", "transformer"),
            freeze_encoder=True,
            max_length=64,
            subset_size=3000,
        )
    results['transformer'] = metrics_tr
    print("Transformer done")
except Exception as e:
    print("Transformer failed or not available:", e)
    results['transformer'] = {"error": str(e)}

# Save results JSON
with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

# Write a simple summary txt
summary_lines = ["Neural NLP Training Results", "========================================"]
for k, v in results.items():
    if isinstance(v, dict) and "final" in v and "f1_macro" in v.get("final", {}):
        final = v["final"]
        summary_lines.append(f"{k.upper()}: Accuracy={final.get('accuracy')}, F1={final.get('f1_macro')}")
    else:
        summary_lines.append(f"{k.upper()}: FAILED or not run -- {v}")

summary_lines.append("\nModels saved to: models\\neural")

with open(os.path.join(OUT_DIR, "results_summary.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(summary_lines))

# Create ADVANCED HTML with epoch curves, confusion matrices, and comparison
out_html_adv = save_results_html_advanced(results, out_dir=OUT_DIR, filename="neural_results_detailed.html")
print("Saved detailed HTML to", out_html_adv)

# Also create simple HTML for backward compatibility
best = None
for name in ("transformer", "lstm", "cnn"):
    val = results.get(name, {})
    if isinstance(val, dict) and "final" in val and "f1_macro" in val.get("final", {}):
        best = val.get("final")
        break

if best:
    out_html = save_results_html(best, out_dir=OUT_DIR, filename="neural_results.html")
    print("Saved simple HTML to", out_html)
else:
    print("No successful model metrics to render HTML.")

print("Done. Results in", OUT_DIR)
