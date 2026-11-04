"""윤곽선 GT 채우기: 끊긴 윤곽(2.7%)까지 보정하는 강건한 채움.

전략
 1) hole-fill: 윤곽 바깥에서 flood fill -> 반전 = 닫힌 윤곽 내부
 2) 닫힌 윤곽으로 채워지지 않은(= 링 두께만 남은) 객체는 윤곽이 끊긴 것:
    dilate(k) -> hole-fill -> erode(k) 로 k 를 키우며 재시도
 3) 영상 경계에 걸친 세포는 경계를 벽으로 삼아 닫는다(패딩 후 flood).
"""
import os
import numpy as np
import cv2

RING_RATIO_MAX = 0.35   # filled 대비 ring 면적비가 이보다 크면 '안 채워짐'으로 판정
MIN_AREA = 150


def _hole_fill(m):
    """m: uint8 0/1 윤곽. 반환: 닫힌 윤곽 내부를 포함해 채운 마스크."""
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    ff = pad.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    inside = (ff != 2)[1:-1, 1:-1]   # 바깥에서 도달 못한 영역 = 내부 + 윤곽
    return inside.astype(np.uint8)


def fill_outlines(m):
    """m: uint8 0/1 윤곽 마스크 전체 -> 채워진 마스크."""
    out = _hole_fill(m)
    # 객체별로 채워졌는지 검사
    n, lab, st, _ = cv2.connectedComponentsWithStats(out, 8)
    bad = []
    for i in range(1, n):
        a = st[i, 4]
        ring = int(m[lab == i].sum())
        if a > 0 and ring / a > RING_RATIO_MAX:
            bad.append(i)
    if not bad:
        return out
    # 끊긴 윤곽: 해당 영역만 떼어내 dilate-fill-erode
    for i in bad:
        x, y, w, h, _ = st[i]
        pad = 12
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(m.shape[1], x + w + pad), min(m.shape[0], y + h + pad)
        sub = ((lab[y0:y1, x0:x1] == i) & (m[y0:y1, x0:x1] > 0)).astype(np.uint8)
        fixed = None
        for k in (2, 3, 4, 6, 8, 11):
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            d = cv2.dilate(sub, ker)
            f = _hole_fill(d)
            e = cv2.erode(f, ker)
            e = e | sub
            ring = int(sub.sum())
            if e.sum() > 0 and ring / e.sum() <= RING_RATIO_MAX:
                fixed = e
                break
        if fixed is None:
            # 경계 절단 세포: 윤곽 볼록껍질로 근사
            cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                hull = cv2.convexHull(np.vstack(cs))
                fixed = np.zeros_like(sub)
                cv2.drawContours(fixed, [hull], -1, 1, cv2.FILLED)
        if fixed is not None:
            reg = out[y0:y1, x0:x1]
            reg[fixed > 0] = 1
    return out


def gt_instances(gt_path):
    g = cv2.imread(gt_path, 0)
    m = (g > 127).astype(np.uint8)
    filled = fill_outlines(m)
    n, lab, st, _ = cv2.connectedComponentsWithStats(filled, 8)
    inst = np.zeros(filled.shape, np.int32)
    nid = 0
    for i in range(1, n):
        if st[i, 4] < MIN_AREA:
            continue
        nid += 1
        inst[lab == i] = nid
    return inst, nid
