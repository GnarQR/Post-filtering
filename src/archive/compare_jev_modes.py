"""
compare_jev_modes.py

Sentence-only vs Full-answer vs Hybrid(OR) JEV를 비교한다.

Sentence-only:
  문장 하나라도 label != ENT 또는 p_entailment < sentence_threshold 이면 SUSPECT

Full-answer:
  전체 answer의 label != ENT 또는 p_entailment < full_threshold 이면 SUSPECT

Hybrid:
  Sentence-only SUSPECT OR Full-answer SUSPECT

권장:
- sentence threshold는 기존 calibration 결과인 0.90으로 고정
- full-answer threshold만 sweep
- 최종 rule은 calibration 후 freeze하고 별도 test set에서 평가

사용:
  python .\src\compare_jev_modes.py

fine sweep 예:
  python .\src\compare_jev_modes.py --start 0.70 --end 0.95 --step 0.01
"""

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{path} line {line_no}: JSON decode error: {e}"
                ) from e
    return rows


def sentence_suspect(sample, threshold):
    for js in sample.get("jev_sentences", []):
        label = js.get("jev_label")
        p = js.get("p_entailment")
        if label != "ENT":
            return True
        if p is not None and p < threshold:
            return True
    return False


def full_suspect(sample, threshold):
    label = sample.get("jev_full_label")
    p = sample.get("p_entailment")
    if label != "ENT":
        return True
    if p is not None and p < threshold:
        return True
    return False


def fbeta(precision, recall, beta):
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def metrics_from_predictions(ids, gold_map, pred_map, beta):
    tp = fp = tn = fn = 0

    for sid in ids:
        gold_unsupported = (
            gold_map[sid]["sample_reference_label"] == "Unsupported"
        )
        pred_suspect = pred_map[sid]

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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
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


def make_thresholds(start, end, step):
    if step <= 0:
        raise ValueError("--step은 0보다 커야 함")

    vals = []
    x = start
    while x <= end + step / 1000:
        vals.append(round(x, 10))
        x += step
    return vals


def validate_ids(sentence_rows, full_rows, gold_rows):
    s_ids = {r["sample_id"] for r in sentence_rows}
    f_ids = {r["sample_id"] for r in full_rows}
    g_ids = {r["sample_id"] for r in gold_rows}

    if s_ids != f_ids or s_ids != g_ids:
        print("\n[ID mismatch]")
        print(f"sentence only: {len(s_ids)}")
        print(f"full answer:   {len(f_ids)}")
        print(f"gold:          {len(g_ids)}")
        print(f"sentence - full: {len(s_ids - f_ids)}")
        print(f"full - sentence: {len(f_ids - s_ids)}")
        print(f"sentence - gold: {len(s_ids - g_ids)}")
        print(f"gold - sentence: {len(g_ids - s_ids)}")
        raise RuntimeError("세 파일의 sample_id 집합이 일치하지 않음")


def pretty_print(title, row, beta):
    fb_key = f"f{beta:g}"
    print(f"\n{title}")
    print(
        f"  P={row['precision']:.3f} "
        f"R={row['recall']:.3f} "
        f"F1={row['f1']:.3f} "
        f"F{beta:g}={row[fb_key]:.3f} "
        f"Spec={row['specificity']:.3f} "
        f"TP/FP/FN/TN="
        f"{row['tp']}/{row['fp']}/{row['fn']}/{row['tn']}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sentence",
        type=Path,
        default=Path("data/jev_results.jsonl"),
        help="문장별 JEV raw 결과",
    )
    ap.add_argument(
        "--full",
        type=Path,
        default=Path("data/jev_full_results.jsonl"),
        help="전체 answer JEV raw 결과",
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
        default=Path("results/jev_mode_comparison.csv"),
    )

    ap.add_argument(
        "--sentence-threshold",
        type=float,
        default=0.90,
        help="기존 sentence-only calibration threshold",
    )
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="F-beta. beta=2는 recall을 더 중요하게 봄.",
    )

    args = ap.parse_args()

    sentence_rows = load_jsonl(args.sentence)
    full_rows = load_jsonl(args.full)
    gold_rows = load_jsonl(args.gold)

    validate_ids(sentence_rows, full_rows, gold_rows)

    sentence_map = {r["sample_id"]: r for r in sentence_rows}
    full_map = {r["sample_id"]: r for r in full_rows}
    gold_map = {r["sample_id"]: r for r in gold_rows}
    ids = [r["sample_id"] for r in gold_rows]

    gold_unsupported = sum(
        r["sample_reference_label"] == "Unsupported"
        for r in gold_rows
    )
    gold_supported = len(gold_rows) - gold_unsupported

    print(f"Samples: {len(ids)}")
    print(
        f"Gold: Supported={gold_supported}, "
        f"Unsupported={gold_unsupported}"
    )

    if gold_unsupported == 23:
        print(
            "⚠ Gold Unsupported가 23개임. "
            "p18 gold 수정(Supported)이 아직 반영되지 않았는지 확인 권장."
        )

    sentence_pred = {
        sid: sentence_suspect(
            sentence_map[sid],
            args.sentence_threshold,
        )
        for sid in ids
    }

    sentence_metrics = metrics_from_predictions(
        ids, gold_map, sentence_pred, args.beta
    )

    sentence_row = {
        "mode": "sentence_only",
        "sentence_threshold": args.sentence_threshold,
        "full_threshold": "",
        **sentence_metrics,
    }

    thresholds = make_thresholds(
        args.start, args.end, args.step
    )

    rows = [sentence_row]

    for t in thresholds:
        full_pred = {
            sid: full_suspect(full_map[sid], t)
            for sid in ids
        }

        full_metrics = metrics_from_predictions(
            ids, gold_map, full_pred, args.beta
        )
        rows.append({
            "mode": "full_only",
            "sentence_threshold": "",
            "full_threshold": t,
            **full_metrics,
        })

        hybrid_pred = {
            sid: sentence_pred[sid] or full_pred[sid]
            for sid in ids
        }

        hybrid_metrics = metrics_from_predictions(
            ids, gold_map, hybrid_pred, args.beta
        )
        rows.append({
            "mode": "hybrid_or",
            "sentence_threshold": args.sentence_threshold,
            "full_threshold": t,
            **hybrid_metrics,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fb_key = f"f{args.beta:g}"

    full_eval = [r for r in rows if r["mode"] == "full_only"]
    hybrid_eval = [r for r in rows if r["mode"] == "hybrid_or"]

    best_full_f1 = max(
        full_eval,
        key=lambda r: (r["f1"], r["recall"]),
    )
    best_full_fb = max(
        full_eval,
        key=lambda r: (r[fb_key], r["recall"]),
    )
    best_hybrid_f1 = max(
        hybrid_eval,
        key=lambda r: (r["f1"], r["recall"]),
    )
    best_hybrid_fb = max(
        hybrid_eval,
        key=lambda r: (r[fb_key], r["recall"]),
    )

    pretty_print(
        f"Sentence-only @ {args.sentence_threshold:.2f}",
        sentence_row,
        args.beta,
    )
    pretty_print(
        f"Best Full-only F1 @ {best_full_f1['full_threshold']:.2f}",
        best_full_f1,
        args.beta,
    )
    pretty_print(
        f"Best Full-only F{args.beta:g} @ {best_full_fb['full_threshold']:.2f}",
        best_full_fb,
        args.beta,
    )
    pretty_print(
        f"Best Hybrid F1 "
        f"(sentence={args.sentence_threshold:.2f}, "
        f"full={best_hybrid_f1['full_threshold']:.2f})",
        best_hybrid_f1,
        args.beta,
    )
    pretty_print(
        f"Best Hybrid F{args.beta:g} "
        f"(sentence={args.sentence_threshold:.2f}, "
        f"full={best_hybrid_fb['full_threshold']:.2f})",
        best_hybrid_fb,
        args.beta,
    )

    print(f"\nSaved: {args.out.resolve()}")
    print(
        "\n먼저 coarse sweep으로 full threshold의 좋은 구간을 찾고, "
        "그 구간만 --step 0.01로 다시 돌리면 됨."
    )

    print("\n[Sentence-only에서는 CLEAN인데 Full-answer에서는 SUSPECT인 샘플]")

    for sid in ids:
        s_pred = sentence_pred[sid]
        f_pred = full_suspect(full_map[sid], 0.0)

        if (not s_pred) and f_pred:
            print(
                f"- {sid}"
                f" | gold={gold_map[sid]['sample_reference_label']}"
                f" | full={full_map[sid]['jev_full_label']}"
                f" | p={full_map[sid]['p_entailment']}"
            )

    print("\n[Full-answer CON 전체]")

    for sid in ids:
        full_row = full_map[sid]

        if full_row["jev_full_label"] == "CON":
            s_pred = sentence_pred[sid]

            print(
                f"- {sid}"
                f" | gold={gold_map[sid]['sample_reference_label']}"
                f" | sentence={'SUSPECT' if s_pred else 'CLEAN'}"
                f" | full_p={full_row['p_entailment']}"
            )

if __name__ == "__main__":
    main()