"""전체 영상 타일 추론 -> 인스턴스 병합 -> 세그멘테이션/계수 평가.

계수 평가식은 참고 논문(Appl. Sci. 2026, 16, 357)과 동일:
    counting Acc = 1 - MAE / mean(N_GT)
세그멘테이션은 전경이 2.14% 뿐이라 픽셀 Accuracy 가 포화(99%+)되므로
Dice / mIoU / 인스턴스 F1@IoU0.5 를 주 지표로 쓴다.
"""
import os, sys, pathlib
import numpy as np
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import IMD, INST, TILE, UPSCALE, IMGSZ, tile_origins


def predict_image(model, img, conf, iou_nms, tta=False, device=0, min_area=150):
    h, w = img.shape[:2]
    xs, ys = tile_origins(w, h)
    tiles, offs = [], []
    for oy in ys:
        for ox in xs:
            t = img[oy:oy + TILE, ox:ox + TILE]
            if UPSCALE != 1:
                t = cv2.resize(t, (IMGSZ, IMGSZ), interpolation=cv2.INTER_CUBIC)
            tiles.append(t)
            offs.append((ox, oy))
    cands = []
    B = 9
    for i in range(0, len(tiles), B):
        res = model.predict(tiles[i:i + B], imgsz=IMGSZ, conf=conf, iou=iou_nms,
                            augment=tta, device=device, verbose=False,
                            retina_masks=True, max_det=300)
        for r, (ox, oy) in zip(res, offs[i:i + B]):
            if r.masks is None:
                continue
            md = r.masks.data.cpu().numpy().astype(np.uint8)
            sc = r.boxes.conf.cpu().numpy()
            for m, s in zip(md, sc):
                if m.shape != (IMGSZ, IMGSZ):
                    m = cv2.resize(m, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)
                if UPSCALE != 1:
                    m = cv2.resize(m, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
                if m.sum() < min_area:
                    continue
                ys_, xs_ = np.nonzero(m)
                cands.append(dict(
                    ox=ox, oy=oy, score=float(s), area=int(m.sum()), mask=m.astype(bool),
                    bbox=(ox + int(xs_.min()), oy + int(ys_.min()),
                          ox + int(xs_.max()), oy + int(ys_.max()))))
    return cands, (h, w)


def _full(c, shape):
    f = np.zeros(shape, bool)
    f[c["oy"]:c["oy"] + TILE, c["ox"]:c["ox"] + TILE] = c["mask"]
    return f


def merge_instances(cands, shape, dup_iou=0.3, contain=0.6):
    """오버랩 타일에서 중복 검출된 동일 세포를 마스크 IoU 로 제거(그리디)."""
    cands = sorted(cands, key=lambda c: -c["score"])
    keep = []
    for c in cands:
        fm = _full(c, shape)
        dup = False
        for k in keep:
            kb, cb = k["bbox"], c["bbox"]
            if cb[0] > kb[2] or cb[2] < kb[0] or cb[1] > kb[3] or cb[3] < kb[1]:
                continue
            inter = int(np.logical_and(fm, k["full"]).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(fm, k["full"]).sum())
            if inter / union >= dup_iou or inter / min(fm.sum(), k["full"].sum()) >= contain:
                dup = True
                break
        if not dup:
            c = dict(c)
            c["full"] = fm
            keep.append(c)
    return keep


def watershed_split(keep, tw, min_area=150):
    """논문과 동일한 distance-transform watershed 재분리."""
    out = []
    for k in keep:
        m = k["full"].astype(np.uint8)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = np.pad(m[y0:y1, x0:x1], 4)
        d = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        seeds = (d >= tw * d.max()).astype(np.uint8)
        ns, lab = cv2.connectedComponents(seeds, 8)
        good = [v for v in range(1, ns) if (lab == v).sum() >= 5]
        if len(good) <= 1:
            out.append(k)
            continue
        markers = np.zeros(sub.shape, np.int32)
        for j, v in enumerate(good, start=2):
            markers[lab == v] = j
        markers[sub == 0] = 1
        cv2.watershed(cv2.cvtColor(sub * 255, cv2.COLOR_GRAY2BGR), markers)
        for j in range(2, len(good) + 2):
            piece = (markers == j) & (sub > 0)
            if piece.sum() < min_area:
                continue
            full = np.zeros(k["full"].shape, bool)
            full[y0:y1, x0:x1] = piece[4:-4, 4:-4]
            ys_, xs_ = np.nonzero(full)
            out.append(dict(score=k["score"], full=full, area=int(full.sum()),
                            bbox=(xs_.min(), ys_.min(), xs_.max(), ys_.max())))
    return out


def match_f1(preds, gi, thr=0.5):
    gids = [int(v) for v in np.unique(gi) if v]
    pairs = []
    for i, p in enumerate(preds):
        pv = gi[p["full"]]
        if pv.size == 0:
            continue
        vals, cnts = np.unique(pv[pv > 0], return_counts=True)
        for g, inter in zip(vals, cnts):
            union = p["full"].sum() + int((gi == g).sum()) - int(inter)
            iou = int(inter) / union
            if iou >= thr:
                pairs.append((iou, i, int(g)))
    pairs.sort(reverse=True)
    up, ug = set(), set()
    for iou, i, g in pairs:
        if i in up or g in ug:
            continue
        up.add(i); ug.add(g)
    tp = len(up)
    return tp, len(preds) - tp, len(gids) - tp, len(gids)


def evaluate(model, files, conf, iou_nms=0.5, tw=None, tta=False, device=0,
             dup_iou=0.3, min_area=150, collect=None):
    rows = []
    for f in files:
        img = cv2.imread(os.path.join(IMD, f))
        gi = np.load(os.path.join(INST, f.replace(".png", ".npy"))).astype(np.int32)
        gt = gi > 0
        cands, shape = predict_image(model, img, conf, iou_nms, tta, device, min_area)
        keep = merge_instances(cands, shape, dup_iou)
        if tw is not None:
            keep = watershed_split(keep, tw, min_area)
        pm = np.zeros(shape, bool)
        for k in keep:
            pm |= k["full"]
        inter = int(np.logical_and(pm, gt).sum())
        dice = 2 * inter / (pm.sum() + gt.sum() + 1e-9)
        iou_fg = inter / (int(np.logical_or(pm, gt).sum()) + 1e-9)
        bg = int(np.logical_and(~pm, ~gt).sum()) / (int(np.logical_or(~pm, ~gt).sum()) + 1e-9)
        tp, fp, fn, ngt = match_f1(keep, gi, 0.5)
        rows.append(dict(file=f, n_pred=len(keep), n_gt=ngt, dice=float(dice),
                         iou=float(iou_fg), miou=float((iou_fg + bg) / 2),
                         pix_acc=float((pm == gt).mean()), tp=tp, fp=fp, fn=fn))
        if collect is not None:
            collect[f] = keep
    return rows


def summarize(rows):
    ng = np.array([r["n_gt"] for r in rows], float)
    npd = np.array([r["n_pred"] for r in rows], float)
    mae = float(np.abs(npd - ng).mean())
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    ss = float(np.sum((ng - ng.mean()) ** 2))
    return dict(
        n_files=len(rows), gt_total=int(ng.sum()), pred_total=int(npd.sum()),
        count_mae=mae, count_acc=float(1 - mae / ng.mean()),
        count_r2=float(1 - np.sum((npd - ng) ** 2) / (ss + 1e-9)),
        count_exact=float(np.mean(npd == ng)),
        dice=float(np.mean([r["dice"] for r in rows])),
        iou=float(np.mean([r["iou"] for r in rows])),
        miou=float(np.mean([r["miou"] for r in rows])),
        pix_acc=float(np.mean([r["pix_acc"] for r in rows])),
        inst_f1=float(2 * tp / (2 * tp + fp + fn + 1e-9)),
        inst_prec=float(tp / (tp + fp + 1e-9)), inst_rec=float(tp / (tp + fn + 1e-9)),
        inst_tp=tp, inst_fp=fp, inst_fn=fn)
