"""주요 모델별 cell segmentation / counting 적용 예시 이미지 생성.

생성물 (figures/examples/)
 1. per-model/<model>_<file>.png  — 모델별 개별 결과 (GT 녹색 / 예측 빨강)
 2. grid_<file>.png               — 한 영상을 여러 모델로 비교한 격자
 3. pipeline_<file>.png           — 원본 → 타일 → 마스크 → 계수 단계별 시각화
 4. errors_<model>.png            — FP/FN 사례 확대 (오차가 어떻게 생기는지)

중요: 각 영상은 **그 영상을 학습에 쓰지 않은 fold 모델**로만 추론한다.
      학습에 쓴 모델로 그린 그림은 성능을 과대 표현하므로 금지.
"""
import os
import sys
import json
import pathlib
import numpy as np
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, IMD, INST, TILE, tile_origins, folds
from evaluate import predict_image, merge_instances

OUT = f"{ROOT}/work/figures/examples"
GT_COLOR = (0, 255, 0)      # 녹색 = ground truth
PR_COLOR = (0, 0, 255)      # 빨강 = prediction
TP_COLOR = (0, 200, 255)    # 주황 = 매칭된 예측
FN_COLOR = (255, 0, 255)    # 자홍 = 미검출
FP_COLOR = (0, 0, 255)      # 빨강 = 오검출

# 성능표 기준 주요 모델 + 권장 conf (README 와 동일)
MODELS = [
    ("yolov8n-seg", 0.55, "3.3M · 엣지 권장"),
    ("yolov8m-seg", 0.65, "27.2M"),
    ("yolov8x-seg", 0.65, "71.8M · 단독 최고"),
    ("yolo11n-seg", 0.65, "2.8M · 최소"),
    ("yolo11s-seg", 0.65, "10.1M"),
    ("yolo11m-seg", 0.55, "22.4M"),
]


def fold_of(fname):
    """해당 영상이 val 로 들어간 fold 번호 (= 학습에 쓰이지 않은 모델)."""
    files = sorted(os.listdir(IMD))
    for k, fold in enumerate(folds(files)):
        if fname in fold:
            return k
    raise ValueError(fname)


def ckpt(model, fold):
    p = f"{ROOT}/work/runs/{model}_f{fold}/weights/best.pt"
    return p if os.path.exists(p) else None


def label(img, text, org=(12, 34), scale=0.8, color=(255, 255, 255), bg=True):
    if bg:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(img, (org[0] - 6, org[1] - h - 8),
                      (org[0] + w + 6, org[1] + 8), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2,
                cv2.LINE_AA)
    return img


def match(preds, gi, thr=0.5):
    """예측-GT 1:1 매칭 → (매칭된 예측 idx, 미검출 GT id)."""
    gids = [int(v) for v in np.unique(gi) if v]
    pairs = []
    for i, p in enumerate(preds):
        pv = gi[p["full"]]
        if pv.size == 0:
            continue
        vals, cnts = np.unique(pv[pv > 0], return_counts=True)
        for g, inter in zip(vals, cnts):
            union = p["full"].sum() + int((gi == g).sum()) - int(inter)
            if int(inter) / union >= thr:
                pairs.append((int(inter) / union, i, int(g)))
    pairs.sort(reverse=True)
    mp, mg = set(), set()
    for _, i, g in pairs:
        if i in mp or g in mg:
            continue
        mp.add(i)
        mg.add(g)
    return mp, [g for g in gids if g not in mg]


def draw(img, preds, gi, matched=None, missed=None, color_code=False, lw=3):
    """GT/예측 윤곽을 겹쳐 그린다. color_code=True 면 TP/FP/FN 색 구분.

    lw: 선 두께. 격자용으로 축소되면 2px 선은 거의 사라지므로 넉넉히 둔다.
    """
    vis = img.copy()
    cs, _ = cv2.findContours((gi > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs, -1, GT_COLOR, lw)
    for i, p in enumerate(preds):
        c = PR_COLOR
        if color_code:
            c = TP_COLOR if (matched and i in matched) else FP_COLOR
        cs, _ = cv2.findContours(p["full"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, c, lw)
    if color_code and missed:
        for g in missed:
            cs, _ = cv2.findContours((gi == g).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, FN_COLOR, lw + 2)
    return vis


def legend(img, items):
    """우상단 범례."""
    x = img.shape[1] - 310
    y = 30
    cv2.rectangle(img, (x - 12, 10), (img.shape[1] - 10, 10 + 32 * len(items)),
                  (0, 0, 0), -1)
    for txt, col in items:
        cv2.line(img, (x, y - 6), (x + 34, y - 6), col, 4)
        cv2.putText(img, txt, (x + 44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 32
    return img


def run_model(model, conf, fname):
    """해당 영상을 학습에 쓰지 않은 fold 모델로 추론."""
    from ultralytics import YOLO
    k = fold_of(fname)
    cp = ckpt(model, k)
    if cp is None:
        return None
    img = cv2.imread(os.path.join(IMD, fname))
    gi = np.load(os.path.join(INST, fname.replace(".png", ".npy"))).astype(np.int32)
    cands, shape = predict_image(YOLO(cp), img, conf=conf, iou_nms=0.5)
    keep = merge_instances(cands, shape, 0.3)
    return dict(img=img, gi=gi, preds=keep, fold=k, n_gt=int(gi.max()),
                n_pred=len(keep), conf=conf)


def fig_per_model(files):
    d = f"{OUT}/per-model"
    os.makedirs(d, exist_ok=True)
    rows = []
    for model, conf, note in MODELS:
        for fname in files:
            r = run_model(model, conf, fname)
            if r is None:
                print(f"  [skip] {model} (fold{fold_of(fname)} 체크포인트 없음)")
                continue
            mp, missed = match(r["preds"], r["gi"])
            vis = draw(r["img"], r["preds"], r["gi"], mp, missed, color_code=True)
            label(vis, f"{model}  ({note})")
            label(vis, f"GT {r['n_gt']}  /  pred {r['n_pred']}"
                       f"   TP {len(mp)}  FP {r['n_pred']-len(mp)}  FN {len(missed)}",
                  org=(12, 72), scale=0.7)
            label(vis, f"{fname}   fold{r['fold']} holdout   conf {conf}",
                  org=(12, 108), scale=0.6)
            legend(vis, [("GT", GT_COLOR), ("TP", TP_COLOR),
                         ("FP", FP_COLOR), ("FN", FN_COLOR)])
            p = f"{d}/{model}_{fname}"
            cv2.imwrite(p, vis)
            rows.append(dict(model=model, file=fname, fold=r["fold"],
                             n_gt=r["n_gt"], n_pred=r["n_pred"], tp=len(mp),
                             fp=r["n_pred"] - len(mp), fn=len(missed), conf=conf))
            print(f"  {model:14s} {fname}  GT {r['n_gt']:2d} pred {r['n_pred']:2d}"
                  f"  TP {len(mp):2d} FP {r['n_pred']-len(mp)} FN {len(missed)}")
    return rows


def fig_grid(fname, cols=3):
    """한 영상을 여러 모델로 비교한 격자."""
    tiles = []
    for model, conf, note in MODELS:
        r = run_model(model, conf, fname)
        if r is None:
            continue
        mp, missed = match(r["preds"], r["gi"])
        vis = draw(r["img"], r["preds"], r["gi"], mp, missed,
                   color_code=True, lw=5)
        err = r["n_pred"] - r["n_gt"]
        label(vis, f"{model}", scale=1.1)
        label(vis, f"pred {r['n_pred']} / GT {r['n_gt']}  ({err:+d})",
              org=(12, 76), scale=0.9,
              color=(120, 255, 120) if err == 0 else (120, 220, 255))
        tiles.append(cv2.resize(vis, (688, 516)))
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols])
                      for i in range(0, len(tiles), cols)])
    head = np.zeros((70, grid.shape[1], 3), np.uint8)
    label(head, f"Model comparison on {fname}  "
                f"(each model evaluated on its own hold-out fold)",
          org=(16, 46), scale=1.0, bg=False)
    out = np.vstack([head, grid])
    p = f"{OUT}/grid_{fname}"
    cv2.imwrite(p, out)
    print(f"  격자: {p}  ({len(MODELS)}종)")
    return p


def fig_pipeline(fname, model="yolov8x-seg", conf=0.65):
    """원본 → 타일 분할 → 예측 마스크 → 계수 결과 단계별 시각화."""
    r = run_model(model, conf, fname)
    if r is None:
        return None
    img, gi, preds = r["img"], r["gi"], r["preds"]
    h, w = img.shape[:2]

    # (1) 원본
    s1 = img.copy()
    label(s1, "1. Input  1376x1032  (label-free brightfield)")

    # (2) 타일 격자
    s2 = img.copy()
    xs, ys = tile_origins(w, h)
    for ox in xs:
        for oy in ys:
            cv2.rectangle(s2, (ox, oy), (ox + TILE, oy + TILE), (255, 200, 0), 2)
    label(s2, f"2. Tiling  {TILE}x{TILE} x {len(xs)*len(ys)} tiles  (stride 432)")

    # (3) 예측 마스크 (인스턴스별 색상)
    s3 = img.copy()
    rng = np.random.default_rng(0)
    overlay = np.zeros_like(img)
    for p in preds:
        overlay[p["full"]] = rng.integers(70, 255, 3)
    s3 = cv2.addWeighted(s3, 0.55, overlay, 0.45, 0)
    label(s3, f"3. Instance masks  {len(preds)} instances  (tile-merged)")

    # (4) 계수 결과: 중심점 번호
    s4 = img.copy()
    cs, _ = cv2.findContours((gi > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(s4, cs, -1, GT_COLOR, 2)
    for i, p in enumerate(sorted(preds, key=lambda q: q["bbox"][1]), 1):
        ys_, xs_ = np.nonzero(p["full"])
        cx, cy = int(xs_.mean()), int(ys_.mean())
        cs, _ = cv2.findContours(p["full"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(s4, cs, -1, PR_COLOR, 2)
        cv2.circle(s4, (cx, cy), 4, (0, 255, 255), -1)
        cv2.putText(s4, str(i), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
    label(s4, f"4. Counting  predicted {len(preds)}  /  GT {r['n_gt']}")

    g = np.vstack([np.hstack([cv2.resize(s1, (688, 516)),
                              cv2.resize(s2, (688, 516))]),
                   np.hstack([cv2.resize(s3, (688, 516)),
                              cv2.resize(s4, (688, 516))])])
    head = np.zeros((70, g.shape[1], 3), np.uint8)
    label(head, f"Pipeline on {fname}  ({model}, fold{r['fold']} hold-out)",
          org=(16, 46), scale=1.0, bg=False)
    p = f"{OUT}/pipeline_{fname}"
    cv2.imwrite(p, np.vstack([head, g]))
    print(f"  파이프라인: {p}")
    return p


def fig_errors(model="yolov8x-seg", conf=0.65, pad=80, cols=5, maxn=10):
    """FP/FN 사례 확대 — 오차가 어떤 모양인지 보여준다."""
    files = sorted(os.listdir(IMD))
    crops = {"FP": [], "FN": []}
    for fname in files:
        if len(crops["FP"]) >= maxn and len(crops["FN"]) >= maxn:
            break
        r = run_model(model, conf, fname)
        if r is None:
            continue
        mp, missed = match(r["preds"], r["gi"])
        img, gi = r["img"], r["gi"]
        h, w = gi.shape

        def crop(cx, cy, kind, mask):
            y0, y1 = max(0, cy - pad), min(h, cy + pad)
            x0, x1 = max(0, cx - pad), min(w, cx + pad)
            sub = img[y0:y1, x0:x1].copy()
            cs, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub, cs, -1,
                             FP_COLOR if kind == "FP" else FN_COLOR, 2)
            sub = cv2.resize(sub, (260, 260), interpolation=cv2.INTER_CUBIC)
            label(sub, f"{kind} {fname[:-4]}", org=(8, 24), scale=0.5)
            return sub

        for i, p in enumerate(r["preds"]):
            if i in mp or len(crops["FP"]) >= maxn:
                continue
            ys_, xs_ = np.nonzero(p["full"])
            crops["FP"].append(crop(int(xs_.mean()), int(ys_.mean()), "FP",
                                    p["full"]))
        for g in missed:
            if len(crops["FN"]) >= maxn:
                break
            ys_, xs_ = np.nonzero(gi == g)
            crops["FN"].append(crop(int(xs_.mean()), int(ys_.mean()), "FN",
                                    gi == g))

    out = []
    for kind in ("FP", "FN"):
        cl = crops[kind]
        if not cl:
            continue
        while len(cl) % cols:
            cl.append(np.zeros((260, 260, 3), np.uint8))
        band = np.vstack([np.hstack(cl[i:i + cols])
                          for i in range(0, len(cl), cols)])
        head = np.zeros((54, band.shape[1], 3), np.uint8)
        txt = ("FP (predicted, not in GT) — texture matches real cells, "
               "likely GT annotation gaps" if kind == "FP"
               else "FN (in GT, not predicted)")
        label(head, txt, org=(12, 36), scale=0.62, bg=False)
        out += [head, band]
    if not out:
        return None
    p = f"{OUT}/errors_{model}.png"
    cv2.imwrite(p, np.vstack(out))
    print(f"  오차 사례: {p}  (FP {len(crops['FP'])}, FN {len(crops['FN'])})")
    return p


def pick_files():
    """예시 영상 선정 — 성공 사례만 고르면 성능을 과대 표현하므로,
    세포 밀도 스펙트럼 + **실제 오차가 발생한 영상**을 함께 넣는다."""
    counts = json.load(open(f"{ROOT}/work/instance_counts_filled.json"))
    srt = sorted(counts.items(), key=lambda kv: kv[1])
    picks = [srt[1][0], srt[len(srt) // 2][0], srt[-2][0]]   # 저/중/고 밀도
    # 오차가 가장 큰 영상 2장 추가 (error_analysis.json 의 파일별 순오차)
    ea = f"{ROOT}/work/error_analysis.json"
    if os.path.exists(ea):
        per = json.load(open(ea)).get("per_file", {})
        worst = sorted(per.items(), key=lambda kv: -abs(kv[1]))
        for f, _ in worst:
            if f not in picks and len(picks) < 5:
                picks.append(f)
    return picks, counts


def main():
    os.makedirs(OUT, exist_ok=True)
    picks, counts = pick_files()
    print(f"선택 영상 (저/중/고 밀도 + 오차 사례): "
          f"{[(f, counts[f]) for f in picks]}")

    print("\n[1] 모델별 개별 결과")
    rows = fig_per_model(picks)

    print("\n[2] 모델 비교 격자")
    for f in picks:
        fig_grid(f)

    print("\n[3] 파이프라인 단계별")
    fig_pipeline(picks[1])

    print("\n[4] 오차 사례")
    fig_errors()

    json.dump(rows, open(f"{OUT}/examples_meta.json", "w"), indent=1)
    print(f"\n저장 위치: {OUT}")
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
