"""Score a results file against a reference, using the grader's three checks.

    python evaluate.py --pred out/results_fast.jsonl --ref out/results_allframes.jsonl

Both files are JSON Lines in the run.py output format (plain or .gz).
The reference is normally an every-frame run of run.py; the prediction is a faster variant.

Metrics (all over the seconds present in the reference):
  * objects F1  - micro F1 over (video, second, object) triples. A second missing from the
                  prediction counts its reference objects as false negatives; a second only in the
                  prediction counts its objects as false positives.          pass if >= 0.95
  * people exact - share of reference seconds where people counts are equal.   pass if >= 0.85
  * people +-1   - share of reference seconds where |pred - ref| <= 1.          pass if >= 0.95
Missing seconds count as wrong for both people metrics. Error lines are reported but not scored.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]


def load(path: str):
    """Return ({(video, second): (frozenset(objects), people)}, {video: error})."""
    opener = gzip.open if path.endswith(".gz") else open
    seconds, errors = {}, {}
    with opener(path, "rt", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: invalid JSON ({e})")
            if "error" in r:
                errors[r["video"]] = r["error"]
                continue
            key = (r["video"], int(r["second"]))
            if key in seconds:
                print(f"warning: duplicate line for {key} in {path}", file=sys.stderr)
            seconds[key] = (frozenset(r.get("objects", [])), int(r.get("people", 0)))
    return seconds, errors


def evaluate(pred: dict, ref: dict) -> dict:
    tp = fp = fn = 0
    per_obj = {o: Counter() for o in TARGETS}
    exact = within1 = 0
    per_video = defaultdict(lambda: Counter())

    for key, (r_obj, r_ppl) in ref.items():
        p_obj, p_ppl = pred.get(key, (frozenset(), None))
        v = per_video[key[0]]
        v["seconds"] += 1
        for o in TARGETS:
            in_r, in_p = o in r_obj, o in p_obj
            if in_r and in_p:
                per_obj[o]["tp"] += 1
            elif in_p:
                per_obj[o]["fp"] += 1
            elif in_r:
                per_obj[o]["fn"] += 1
        t, f_p, f_n = len(r_obj & p_obj), len(p_obj - r_obj), len(r_obj - p_obj)
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
        v["obj_errors"] += f_p + f_n
        if p_ppl is None:
            v["missing"] += 1
            continue
        if p_ppl == r_ppl:
            exact += 1
            v["exact"] += 1
        if abs(p_ppl - r_ppl) <= 1:
            within1 += 1

    # seconds the prediction has but the reference does not
    extra = [k for k in pred if k not in ref]
    for k in extra:
        objs = pred[k][0]
        fp += len(objs)
        for o in objs:
            if o in per_obj:
                per_obj[o]["fp"] += 1

    def f1(t, a, b):
        return 1.0 if t + a + b == 0 else 2 * t / (2 * t + a + b)

    n = len(ref)
    return {
        "seconds_ref": n,
        "seconds_pred": len(pred),
        "seconds_missing": sum(1 for k in ref if k not in pred),
        "seconds_extra": len(extra),
        "tp": tp, "fp": fp, "fn": fn,
        "objects_f1": f1(tp, fp, fn),
        "objects_precision": tp / (tp + fp) if tp + fp else 1.0,
        "objects_recall": tp / (tp + fn) if tp + fn else 1.0,
        "people_exact": exact / n if n else 1.0,
        "people_within1": within1 / n if n else 1.0,
        "per_object": {o: {**c, "f1": f1(c["tp"], c["fp"], c["fn"])} for o, c in per_obj.items()},
        "per_video": per_video,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="results to score")
    ap.add_argument("--ref", required=True, help="reference results (every-frame run)")
    ap.add_argument("--min-f1", type=float, default=0.95)
    ap.add_argument("--min-exact", type=float, default=0.85)
    ap.add_argument("--min-within1", type=float, default=0.95)
    ap.add_argument("--worst", type=int, default=10, help="show the N worst videos")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args()

    pred, pred_err = load(args.pred)
    ref, ref_err = load(args.ref)
    m = evaluate(pred, ref)

    checks = [
        ("objects F1", m["objects_f1"], args.min_f1),
        ("people exact", m["people_exact"], args.min_exact),
        ("people within 1", m["people_within1"], args.min_within1),
    ]
    passed = all(v >= t for _, v, t in checks)

    if args.json:
        out = {k: v for k, v in m.items() if k != "per_video"}
        out.update(passed=passed, pred_errors=len(pred_err), ref_errors=len(ref_err))
        print(json.dumps(out, indent=2))
        sys.exit(0 if passed else 1)

    print(f"reference seconds: {m['seconds_ref']}  predicted: {m['seconds_pred']}  "
          f"missing: {m['seconds_missing']}  extra: {m['seconds_extra']}")
    print(f"error lines - pred: {len(pred_err)}  ref: {len(ref_err)}")
    only_pred = sorted(set(pred_err) - set(ref_err))
    if only_pred:
        print(f"  videos that error only in pred: {only_pred[:10]}{' ...' if len(only_pred) > 10 else ''}")
    print()
    for name, v, t in checks:
        print(f"  {name:<16} {v:.4f}   (need >= {t:.2f})  {'PASS' if v >= t else 'FAIL'}")
    print(f"  {'precision':<16} {m['objects_precision']:.4f}")
    print(f"  {'recall':<16} {m['objects_recall']:.4f}")
    print("\nper object:")
    for o, c in m["per_object"].items():
        print(f"  {o:<11} tp={c.get('tp', 0):>7} fp={c.get('fp', 0):>6} fn={c.get('fn', 0):>6}  f1={c['f1']:.4f}")
    if args.worst:
        worst = sorted(m["per_video"].items(),
                       key=lambda kv: kv[1]["exact"] / max(kv[1]["seconds"], 1))[: args.worst]
        print(f"\n{args.worst} videos with the lowest people-exact rate:")
        for vid, c in worst:
            print(f"  {vid:<40} seconds={c['seconds']:>5} exact={c['exact'] / max(c['seconds'], 1):.3f} "
                  f"obj_errors={c['obj_errors']} missing={c['missing']}")
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
