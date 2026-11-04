"""단계 3b: 실제 D4 TTA + 모델 앙상블로 97% 돌파 시도.

모든 수치는 nested 집계 (fold k 의 하이퍼파라미터는 k 를 제외한 fold 에서만 선택).
"""
import os, sys, pathlib, json, itertools, collections
import pathlib
import numpy as np
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, NFOLD, IMD, INST
from evaluate import merge_instances, match_f1, summarize
from tta import predict_image_tta

OUT = f"{ROOT}/work/stage3b.json"


def val_files(fold):
    return json.load(open(f"{ROOT}/work/ds_fold{fold}/build_meta.json"))["val_files"]


def ckpt(key, fold):
    return f"{ROOT}/work/runs/{key}_f{fold}/weights/best.pt"


def eval_cfg(models, confs, files, tta, min_support, vote, dup_iou=0.3, min_area=150):
    rows = []
    for f in files:
        img = cv2.imread(os.path.join(IMD, f))
        gi = np.load(os.path.join(INST, f.replace(".png", ".npy"))).astype(np.int32)
        gt = gi > 0
        allc, shape = [], None
        for mi, (m, cf) in enumerate(zip(models, confs)):
            cands, shape = predict_image_tta(m, img, conf=cf, tta=tta,
                                             min_support=min_support, min_area=min_area)
            for k in merge_instances(cands, shape, dup_iou):
                k = dict(k); k["mi"] = mi
                allc.append(k)
        # 모델 간 클러스터링 + 투표
        allc.sort(key=lambda c: -c["score"])
        clusters = []
        for c in allc:
            placed = False
            for cl in clusters:
                kb, cb = cl["bbox"], c["bbox"]
                if cb[0] > kb[2] or cb[2] < kb[0] or cb[1] > kb[3] or cb[3] < kb[1]:
                    continue
                inter = int(np.logical_and(c["full"], cl["full"]).sum())
                if inter and (inter / int(np.logical_or(c["full"], cl["full"]).sum()) >= dup_iou
                              or inter / min(c["full"].sum(), cl["full"].sum()) >= 0.6):
                    cl["members"].append(c); cl["mis"].add(c["mi"]); placed = True
                    break
            if not placed:
                clusters.append(dict(full=c["full"], bbox=c["bbox"], score=c["score"],
                                     members=[c], mis={c["mi"]}))
        keep = []
        for cl in clusters:
            if len(cl["mis"]) < vote:
                continue
            acc = np.zeros(cl["full"].shape, np.int16)
            for mm in cl["members"]:
                acc += mm["full"]
            fm = acc >= max(1, (len(cl["members"]) + 1) // 2)
            if fm.sum() < min_area:
                continue
            ys, xs = np.nonzero(fm)
            keep.append(dict(full=fm, score=cl["score"], area=int(fm.sum()),
                             bbox=(xs.min(), ys.min(), xs.max(), ys.max())))
        pm = np.zeros(shape, bool)
        for k in keep:
            pm |= k["full"]
        inter = int(np.logical_and(pm, gt).sum())
        iou_fg = inter / (int(np.logical_or(pm, gt).sum()) + 1e-9)
        bg = int(np.logical_and(~pm, ~gt).sum()) / (int(np.logical_or(~pm, ~gt).sum()) + 1e-9)
        tp, fp, fn, ngt = match_f1(keep, gi, 0.5)
        rows.append(dict(file=f, n_pred=len(keep), n_gt=ngt,
                         dice=2 * inter / (pm.sum() + gt.sum() + 1e-9), iou=iou_fg,
                         miou=(iou_fg + bg) / 2, pix_acc=float((pm == gt).mean()),
                         tp=tp, fp=fp, fn=fn))
    return summarize(rows)


def nested_agg(sw, metric="count_acc"):
    kf = lambda s: (s["conf"], s["min_support"], s["vote"])
    rows = []
    for k in range(NFOLD):
        agg = collections.defaultdict(list)
        for j in range(NFOLD):
            if j == k:
                continue
            for s in sw[j]:
                agg[kf(s)].append(s[metric])
        best = max(agg.items(), key=lambda kv: np.mean(kv[1]))[0]
        rows.append(next(s for s in sw[k] if kf(s) == best))
    return rows


def summary(rows):
    a = [r["count_acc"] for r in rows]
    rng = np.random.default_rng(0)
    bs = rng.choice(np.array(a), size=(20000, len(a)), replace=True).mean(axis=1)
    return dict(count_acc=float(np.mean(a)), sd=float(np.std(a)),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                count_mae=float(np.mean([r["count_mae"] for r in rows])),
                count_r2=float(np.mean([r["count_r2"] for r in rows])),
                count_exact=float(np.mean([r["count_exact"] for r in rows])),
                dice=float(np.mean([r["dice"] for r in rows])),
                miou=float(np.mean([r["miou"] for r in rows])),
                inst_f1=float(np.mean([r["inst_f1"] for r in rows])),
                pix_acc=float(np.mean([r["pix_acc"] for r in rows])),
                gt_total=int(sum(r["gt_total"] for r in rows)),
                pred_total=int(sum(r["pred_total"] for r in rows)),
                sel=[(r["conf"], r["min_support"], r["vote"]) for r in rows],
                per_fold=rows)


def run(tag, keys, tta, confs, supports, votes):
    from ultralytics import YOLO
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if tag in res:
        print(f"[skip] {tag}")
        return res
    sw = {}
    for fold in range(NFOLD):
        models = [YOLO(ckpt(k, fold)) for k in keys]
        vf = val_files(fold)
        lst = []
        for conf, sup, vote in itertools.product(confs, supports, votes):
            if vote > len(keys):
                continue
            s = eval_cfg(models, [conf] * len(keys), vf, tta, sup, vote)
            s.update(conf=conf, min_support=sup, vote=vote, tta=tta, fold=fold,
                     keys=keys)
            lst.append(s)
            print(f"  {tag} f{fold} conf={conf} sup={sup} vote={vote} -> "
                  f"cnt_acc={s['count_acc']:.4f} mae={s['count_mae']:.3f} "
                  f"dice={s['dice']:.4f} f1={s['inst_f1']:.4f} "
                  f"pred={s['pred_total']}/{s['gt_total']}", flush=True)
        sw[fold] = lst
    n = summary(nested_agg(sw))
    res[tag] = dict(nested=n, sweeps=sw, keys=keys, tta=tta)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f">> {tag} NESTED cnt_acc={n['count_acc']:.4f}±{n['sd']:.4f} "
          f"CI{[round(x,4) for x in n['ci']]} dice={n['dice']:.4f} "
          f"miou={n['miou']:.4f} f1={n['inst_f1']:.4f} mae={n['count_mae']:.3f} "
          f"sel={n['sel']}", flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "tta"):
        run("T1_d4tta_11s", ["yolo11s-seg"], True,
            confs=(0.45, 0.55, 0.65), supports=(2, 4, 5), votes=(1,))
    if which in ("all", "ens"):
        run("T2_ens_notta", ["yolo11s-seg", "yolov8m-seg"], False,
            confs=(0.45, 0.55, 0.65), supports=(1,), votes=(1, 2))
    if which in ("all", "enstta"):
        run("T3_ens_d4tta", ["yolo11s-seg", "yolov8m-seg"], True,
            confs=(0.45, 0.55, 0.65), supports=(4,), votes=(1, 2))
    if which in ("all", "ens2"):
        # v8n 은 단일 최고 + 구조 다양성(최소 모델). 2-모델 조합들
        run("T4_ens_11s_v8n", ["yolo11s-seg", "yolov8n-seg"], False,
            confs=(0.45, 0.55, 0.65), supports=(1,), votes=(1, 2))
        run("T5_ens_v8m_v8n", ["yolov8m-seg", "yolov8n-seg"], False,
            confs=(0.45, 0.55, 0.65), supports=(1,), votes=(1, 2))
    if which in ("all", "ens3"):
        run("T6_ens3", ["yolo11s-seg", "yolov8m-seg", "yolov8n-seg"], False,
            confs=(0.45, 0.55, 0.65), supports=(1,), votes=(2, 3))
    if which == "ens3x":
        run("T7_ens3x", ["yolo11s-seg", "yolov8n-seg", "yolov8x-seg"], False,
            confs=(0.45, 0.55, 0.65), supports=(1,), votes=(2, 3))
        run("T8_ens4", ["yolo11s-seg", "yolov8m-seg", "yolov8n-seg", "yolov8x-seg"],
            False, confs=(0.55, 0.65), supports=(1,), votes=(2, 3))
    print("\n===== STAGE3B SUMMARY =====")
    res = json.load(open(OUT))
    for tag, v in res.items():
        n = v["nested"]
        print(f"{tag:20s} cnt_acc={n['count_acc']:.4f}±{n['sd']:.4f} "
              f"CI{[round(x,4) for x in n['ci']]} dice={n['dice']:.4f} "
              f"f1={n['inst_f1']:.4f} mae={n['count_mae']:.3f}")
