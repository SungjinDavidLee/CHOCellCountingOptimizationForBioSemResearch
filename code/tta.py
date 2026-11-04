"""D4 TTA 직접 구현 + 다중 모델 앙상블 추론.

배경: 이 ultralytics(8.4.150) 의 segment 태스크에서 predict(augment=True) 는
no-op 이었다 (검출 수와 confidence 가 augment=False 와 완전 동일함을 실측).
따라서 8방향(D4: 회전4 x 반전2) 변환을 직접 적용하고 마스크를 역변환해 투표한다.
"""
import sys, pathlib
import numpy as np
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import TILE, UPSCALE, IMGSZ, tile_origins

D4 = [(0, False), (1, False), (2, False), (3, False),
      (0, True), (1, True), (2, True), (3, True)]


def _fwd(t, k, fl):
    if fl:
        t = np.ascontiguousarray(t[:, ::-1])
    return np.ascontiguousarray(np.rot90(t, k)) if k else t


def _inv(m, k, fl):
    if k:
        m = np.rot90(m, -k)
    if fl:
        m = m[:, ::-1]
    return np.ascontiguousarray(m)


def predict_tiles_d4(model, tiles, conf, iou_nms, device=0, tta=True):
    """타일 리스트 -> 타일별 [(mask(TILE,TILE) bool, score)] 리스트."""
    variants = D4 if tta else [(0, False)]
    per_tile = [[] for _ in tiles]
    for k, fl in variants:
        batch = [_fwd(t, k, fl) for t in tiles]
        res = model.predict(batch, imgsz=IMGSZ, conf=conf, iou=iou_nms, device=device,
                            verbose=False, retina_masks=True, max_det=300)
        for ti, r in enumerate(res):
            if r.masks is None:
                continue
            md = r.masks.data.cpu().numpy().astype(np.uint8)
            sc = r.boxes.conf.cpu().numpy()
            for m, s in zip(md, sc):
                if m.shape != (IMGSZ, IMGSZ):
                    m = cv2.resize(m, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)
                m = _inv(m, k, fl)
                if UPSCALE != 1:
                    m = cv2.resize(m, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
                per_tile[ti].append((m.astype(bool), float(s)))
    return per_tile


def tta_fuse(dets, nvar, iou_thr=0.5, min_support=None, min_area=150):
    """같은 타일의 D4 검출들을 클러스터링해 마스크 과반 투표로 융합."""
    if min_support is None:
        min_support = max(1, (nvar + 1) // 2)
    dets = sorted(dets, key=lambda d: -d[1])
    clusters = []
    for m, s in dets:
        a = m.sum()
        if a < min_area // 2:
            continue
        placed = False
        for cl in clusters:
            inter = int(np.logical_and(m, cl["ref"]).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(m, cl["ref"]).sum())
            if inter / union >= iou_thr or inter / min(a, cl["ref"].sum()) >= 0.7:
                cl["acc"] += m.astype(np.int16)
                cl["n"] += 1
                cl["score"] = max(cl["score"], s)
                placed = True
                break
        if not placed:
            clusters.append(dict(ref=m, acc=m.astype(np.int16), n=1, score=s))
    out = []
    for cl in clusters:
        if cl["n"] < min_support:
            continue
        fm = cl["acc"] >= max(1, (cl["n"] + 1) // 2)
        if fm.sum() < min_area:
            continue
        out.append((fm, cl["score"], cl["n"] / nvar))
    return out


def predict_image_tta(model, img, conf, iou_nms=0.5, device=0, tta=True,
                      min_area=150, min_support=None):
    """전체 영상 -> 원본 좌표계 후보 인스턴스 목록."""
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
    nvar = len(D4) if tta else 1
    per_tile = predict_tiles_d4(model, tiles, conf, iou_nms, device, tta)
    cands = []
    for dets, (ox, oy) in zip(per_tile, offs):
        for fm, s, sup in tta_fuse(dets, nvar, min_support=min_support,
                                   min_area=min_area):
            ys_, xs_ = np.nonzero(fm)
            if xs_.size == 0:
                continue
            cands.append(dict(ox=ox, oy=oy, score=s, support=sup, mask=fm,
                              area=int(fm.sum()),
                              bbox=(ox + int(xs_.min()), oy + int(ys_.min()),
                                    ox + int(xs_.max()), oy + int(ys_.max()))))
    return cands, (h, w)
