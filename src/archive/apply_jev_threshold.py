import argparse
import json
from pathlib import Path


def main(input_path: Path, out_path: Path, threshold: float):
    rows = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            row["jev_suspect"] = any(
                js["jev_label"] != "ENT"
                or (
                    js["p_entailment"] is not None
                    and js["p_entailment"] < threshold
                )
                for js in row["jev_sentences"]
            )

            row["jev_threshold"] = threshold
            rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    suspects = sum(r["jev_suspect"] for r in rows)

    print(f"threshold = {threshold:.2f}")
    print(f"전체 = {len(rows)}")
    print(f"SUSPECT = {suspects}")
    print(f"CLEAN = {len(rows) - suspects}")
    print(f"저장: {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/jev_results.jsonl")
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/jev_results_t090.jsonl")
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.90
    )

    args = ap.parse_args()
    main(args.input, args.out, args.threshold)