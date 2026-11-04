"""중첩 교차검증(nested CV) 집계 — 선택 편향 없는 정직한 추정.

문제: stage2 의 'fixed' 는 conf 를 fold0 에서 골라 전 fold 에 적용했다.
fold0 에 운좋게 맞는 conf 가 다른 fold 에서는 나쁠 수 있고(v8m conf=0.25 과대계수),
'oracle' 은 각 fold 자기 자신에서 골랐으니 낙관 편향이다.

해법: fold k 를 평가할 때 conf 는 **k 를 제외한 fold 들**의 성능으로만 고른다.
fold k 의 테스트 데이터는 선택에 전혀 쓰이지 않으므로 완전 홀드아웃이다.
(각 fold 의 val 은 다른 fold 의 train 에 포함되지만, conf 는 학습된 가중치를
 바꾸지 않는 후처리 파라미터이고 k 의 데이터를 보지 않으므로 유효하다.
 이 점은 보고서에 그대로 명시한다.)
"""
import sys, pathlib, json, collections
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, NFOLD

S2 = f"{ROOT}/work/stage2_cv.json"


def key_of(s):
    return (s["conf"], s["tw"])


def nested(res, key, metric="count_acc"):
    fl = res[key]["folds"]
    rows = []
    for k in range(NFOLD):
        # k 제외 fold 들의 평균으로 최적 (conf,tw) 선택
        agg = collections.defaultdict(list)
        for j in range(NFOLD):
            if j == k:
                continue
            for s in fl[str(j)]["sweeps"]:
                agg[key_of(s)].append(s[metric])
        best = max(agg.items(), key=lambda kv: np.mean(kv[1]))[0]
        sel = next(s for s in fl[str(k)]["sweeps"] if key_of(s) == best)
        rows.append(dict(fold=k, sel_conf=best[0], sel_tw=best[1], **{
            m: sel[m] for m in ("count_acc", "count_mae", "count_r2", "count_exact",
                                "dice", "miou", "iou", "pix_acc", "inst_f1",
                                "inst_prec", "inst_rec", "inst_tp", "inst_fp",
                                "inst_fn", "gt_total", "pred_total")}))
    return rows


def boot_ci(vals, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    bs = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def report(res, keys=None):
    keys = keys or [k for k, v in res.items() if len(v["folds"]) == NFOLD]
    out = {}
    print(f"{'model':16s} {'mode':8s} {'cnt_acc':>16s} {'dice':>8s} {'mIoU':>8s} "
          f"{'instF1':>8s} {'MAE':>6s}")
    for key in keys:
        fl = res[key]["folds"]
        modes = {
            "fixed": [fl[str(k)]["fixed"] for k in range(NFOLD)],
            "nested": nested(res, key),
            "oracle": [fl[str(k)]["oracle"] for k in range(NFOLD)],
        }
        out[key] = {}
        for mode, lst in modes.items():
            a = [x["count_acc"] for x in lst]
            lo, hi = boot_ci(a)
            rec = dict(
                count_acc=float(np.mean(a)), count_acc_sd=float(np.std(a)),
                count_acc_ci=[lo, hi],
                count_mae=float(np.mean([x["count_mae"] for x in lst])),
                dice=float(np.mean([x["dice"] for x in lst])),
                dice_sd=float(np.std([x["dice"] for x in lst])),
                miou=float(np.mean([x["miou"] for x in lst])),
                inst_f1=float(np.mean([x["inst_f1"] for x in lst])),
                inst_f1_sd=float(np.std([x["inst_f1"] for x in lst])),
                pix_acc=float(np.mean([x["pix_acc"] for x in lst])),
                gt_total=int(sum(x["gt_total"] for x in lst)),
                pred_total=int(sum(x["pred_total"] for x in lst)),
                inst_tp=int(sum(x["inst_tp"] for x in lst)),
                inst_fp=int(sum(x["inst_fp"] for x in lst)),
                inst_fn=int(sum(x["inst_fn"] for x in lst)),
                per_fold=lst)
            out[key][mode] = rec
            print(f"{key:16s} {mode:8s} {rec['count_acc']:.4f}±{rec['count_acc_sd']:.4f} "
                  f"{rec['dice']:8.4f} {rec['miou']:8.4f} {rec['inst_f1']:8.4f} "
                  f"{rec['count_mae']:6.3f}   CI[{lo:.4f},{hi:.4f}]")
        if "nested" in out[key]:
            sc = [(r["fold"], r["sel_conf"], r["sel_tw"]) for r in modes["nested"]]
            print(f"{'':16s} nested 선택 conf/tw per fold: {sc}")
    json.dump(out, open(f"{ROOT}/work/cv_report.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    res = json.load(open(S2))
    report(res, sys.argv[1:] or None)
