"""
03_evaluate.py — Evaluacion del modelo entrenado sobre el test set
Calcula BLEU-4, WER y ROUGE-L. Muestra ejemplos de predicciones.

Uso:
    python 03_evaluate.py                  # evalua phase_b_best.pt
    python 03_evaluate.py --ckpt phase_c_best.pt
    python 03_evaluate.py --examples 20   # mostrar 20 ejemplos
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, BOS_ID, EOS_ID, PAD_ID, BATCH_SIZE
from model import LSAModel, get_device
from dataset import Vocabulary, LSADataset, collate_fn


def evaluate(ckpt_name: str, n_examples: int = 10):
    device = get_device()
    vocab  = Vocabulary()

    # Cargar modelo
    ckpt_path = CHECKPOINTS_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint no encontrado: {ckpt_path}")
        print(f"Checkpoints disponibles:")
        for f in sorted(CHECKPOINTS_DIR.glob("*.pt")):
            print(f"  {f.name}")
        return

    model = LSAModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    print(f"[eval] Modelo cargado: {ckpt_path}")

    test_ds     = LSADataset("test", vocab, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    print(f"[eval] Test set: {len(test_ds)} muestras")

    hypotheses, references = [], []

    with torch.no_grad():
        for batch in test_loader:
            kps      = batch["keypoints"].to(device)
            src_pad  = batch["pad_mask"].to(device)

            embeddings, _ = model.encoder(kps, src_pad)
            pred_ids = model.decoder.greedy_decode(
                embeddings, BOS_ID, EOS_ID, max_len=64
            )

            for ids, ref in zip(pred_ids, batch["labels"]):
                hypotheses.append(vocab.decode(ids))
                references.append(ref)

    print(f"\n[eval] {len(hypotheses)} predicciones generadas")

    # BLEU-4
    try:
        from sacrebleu.metrics import BLEU
        bleu_scorer = BLEU(effective_order=True)
        bleu_score  = bleu_scorer.corpus_score(hypotheses, [references])
        print(f"BLEU-4:  {bleu_score.score:.2f}")
    except ImportError:
        print("BLEU-4:  (instalar sacrebleu)")

    # WER
    try:
        from jiwer import wer
        wer_score = wer(references, hypotheses) * 100
        print(f"WER:     {wer_score:.1f}%")
    except ImportError:
        print("WER:     (instalar jiwer)")

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
        scores = [scorer.score(r, h)["rougeL"].fmeasure
                  for r, h in zip(references, hypotheses)]
        rouge_l = sum(scores) / len(scores)
        print(f"ROUGE-L: {rouge_l:.3f}")
    except ImportError:
        print("ROUGE-L: (instalar rouge-score)")

    # Ejemplos
    if n_examples > 0:
        print(f"\n{'='*60}")
        print(f"Ejemplos (primeros {n_examples}):")
        print(f"{'='*60}")
        for i, (hyp, ref) in enumerate(zip(hypotheses[:n_examples], references[:n_examples])):
            print(f"\n[{i+1}] REF: {ref}")
            print(f"     HYP: {hyp}")

    # Guardar resultados
    results = {
        "checkpoint": ckpt_name,
        "n_samples": len(hypotheses),
        "predictions": [
            {"reference": r, "hypothesis": h}
            for r, h in zip(references, hypotheses)
        ],
    }
    out_path = CHECKPOINTS_DIR / f"{Path(ckpt_name).stem}_eval.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluacion del modelo LSA")
    parser.add_argument("--ckpt",     default="phase_b_best.pt",
                        help="Nombre del checkpoint (default: phase_b_best.pt)")
    parser.add_argument("--examples", type=int, default=10,
                        help="Cantidad de ejemplos a mostrar (default: 10)")
    args = parser.parse_args()
    evaluate(args.ckpt, args.examples)
