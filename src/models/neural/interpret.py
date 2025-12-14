from typing import List, Dict, Any
import torch


def explain_text(model: Any, tokenizer: Any, text: str, mode: str = "transformer") -> Dict:
    """Return a simple interpretability output for a single text.

    - For transformer: return attention maps (if available) and top tokens by attention.
    - For LSTM/CNN: return token saliency via gradients on embeddings.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "transformer":
        enc = tokenizer(text, return_tensors="pt", truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
        # pick last layer attentions mean across heads
        attentions = out.attentions  # tuple (layers, batch, heads, seq, seq)
        if attentions:
            att = attentions[-1].mean(dim=1).squeeze(0)  # seq x seq
            # token importance via attention to [CLS] (or first token)
            importances = att[0].cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
            ranked = sorted(zip(tokens, importances), key=lambda x: -x[1])[:10]
            return {"type": "transformer", "attention_top_tokens": ranked}
        return {"type": "transformer", "attention_top_tokens": []}
    else:
        # assume model has embedding layer named `emb`
        toks = text.split()
        model.to(device)
        model.eval()
        # build dummy ids: user must provide vocabulary mapping externally
        raise NotImplementedError("Saliency for LSTM/CNN requires vocabulary and embedding access")
