from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path} line {line_no}: JSON decode error: {e}") from e
    return rows


def predict_suspect(sample: dict, threshold: float) -> bool:
    """
    Sample-level JEV rule:
    - any sentence whose JEV label is not ENT -> SUSPECT
    - any ENT sentence whose p_entailment < threshold -> SUSPECT
    """
    for js in sample.get("jev_sentences", []):
        label = js.get("jev_label")
        p = js.get("p_entailment")

        if label != "ENT":
            return True
        if p is not None and p < threshold:
            return True

    return False


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return 0.0 if denom == 0.0 else (1.0 + b2) * precision * recall / denom


def evaluate(jev_rows: list[dict], gold_map: dict[str, dict],
             threshold: float, beta: float) -> dict:
    tp = fp = tn = fn = 0

    for row in jev_rows:
        sid = row["sample_id"]
        gold_unsupported = (
            gold_map[sid]["sample_reference_label"] == "Unsupported"
        )
        pred_suspect = predict_suspect(row, threshold)

        if pred_suspect and gold_unsupported:
            tp += 1
        elif pred_suspect and not gold_unsupported:
            fp += 1
        elif not pred_suspect and gold_unsupported:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1 = fbeta(precision, recall, 1.0)
    fb = fbeta(precision, recall, beta)

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "suspect_count": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        f"f{beta:g}": fb,
        "specificity": specificity,
        "accuracy": accuracy,
        "fnr": 1.0 - recall,
        "fpr": 1.0 - specificity,
    }


def make_thresholds(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--step must be > 0")
    vals = []
    x = start
    # avoid floating-point accumulation issues in CSV output
    while x <= end + step / 1000:
        vals.append(round(x, 10))
        x += step
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jev",
        type=Path,
        default=Path("data/jev_results.jsonl"),
        help="run_jev.py output",
    )
    ap.add_argument(
        "--gold",
        type=Path,
        default=Path("data/reference_all_236_labeled.jsonl"),
        help="blind reference labels",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/jev_threshold_sweep.csv"),
    )
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="F-beta score. beta=2 weights recall more than precision.",
    )
    args = ap.parse_args()

    jev_rows = load_jsonl(args.jev)
    gold_rows = load_jsonl(args.gold)

    gold_map = {r["sample_id"]: r for r in gold_rows}
    jev_ids = {r["sample_id"] for r in jev_rows}
    gold_ids = set(gold_map)

    missing_gold = jev_ids - gold_ids
    missing_jev = gold_ids - jev_ids
    if missing_gold or missing_jev:
        raise RuntimeError(
            f"sample_id mismatch: missing_gold={len(missing_gold)}, "
            f"missing_jev={len(missing_jev)}"
        )

    thresholds = make_thresholds(args.start, args.end, args.step)
    results = [
        evaluate(jev_rows, gold_map, t, args.beta)
        for t in thresholds
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nThreshold sweep")
    print(
        f"{'Thr':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} "
        f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'F2':>7} {'Spec':>7}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['threshold']:>5.2f} "
            f"{r['tp']:>3} "
            f"{r['fp']:>3} "
            f"{r['fn']:>3} "
            f"{r['tn']:>3} "
            f"{r['precision']:>7.3f} "
            f"{r['recall']:>7.3f} "
            f"{r['f1']:>7.3f} "
            f"{r['f2']:>7.3f} "
            f"{r['specificity']:>7.3f}"
        )

    best_f1 = max(results, key=lambda r: (r["f1"], r["recall"]))
    fbeta_key = f"f{args.beta:g}"
    best_fb = max(results, key=lambda r: (r[fbeta_key], r["recall"]))

    gold_unsupported = sum(
        r["sample_reference_label"] == "Unsupported" for r in gold_rows
    )

    print(f"JEV samples: {len(jev_rows)}")
    print(
        f"Gold: Supported={len(gold_rows) - gold_unsupported}, "
        f"Unsupported={gold_unsupported}"
    )
    print(f"Saved: {args.out.resolve()}")

    print("\nBest F1")
    print(
        f"  threshold={best_f1['threshold']:.2f} "
        f"P={best_f1['precision']:.3f} "
        f"R={best_f1['recall']:.3f} "
        f"F1={best_f1['f1']:.3f} "
        f"Spec={best_f1['specificity']:.3f} "
        f"TP/FP/FN/TN="
        f"{best_f1['tp']}/{best_f1['fp']}/{best_f1['fn']}/{best_f1['tn']}"
    )

    print(f"\nBest F{args.beta:g} (recall-weighted)")
    print(
        f"  threshold={best_fb['threshold']:.2f} "
        f"P={best_fb['precision']:.3f} "
        f"R={best_fb['recall']:.3f} "
        f"F{args.beta:g}={best_fb[fbeta_key]:.3f} "
        f"Spec={best_fb['specificity']:.3f} "
        f"TP/FP/FN/TN="
        f"{best_fb['tp']}/{best_fb['fp']}/{best_fb['fn']}/{best_fb['tn']}"
    )


if __name__ == "__main__":
    main()
