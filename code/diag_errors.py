"""계수 오차의 실제 원인 진단.

가설
 H1. TTA 가 no-op (augment 인자 무시) -> 직접 확인
 H2. 타일 경계에서 잘린 세포가 FP/FN 유발
 H3. 영상 경계에 걸친 세포(GT 에 포함됨) 검출 실패
 H4. 특정 파일에 오차가 집중 (디브리/초점)
"""
import os, sys, pathlib, json, collections
import pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, IMD, INST, TILE, tile_origins
from evaluate import predict_image, merge_instances, match_f1

KEY = "yolo11s-seg"


def ckpt(fold):
    return f"{ROOT}/work/runs/{KEY}_f{fold}/weights/best.pt"


def vf(fold):
    return json.load(open(f"{ROOT}/work/ds_fold{fold}/build_meta.json"))["val_files"]


def main():
    from ultralytics import YOLO
    # H1: TTA 실제 작동 확인
    m = YOLO(ckpt(0))
    img = cv2.imread(os.path.join(IMD, vf(0)[0]))
    t = img[0:TILE, 0:TILE]
    r0 = m.predict([t], imgsz=TILE, conf=0.3, augment=False, verbose=False,
                   retina_masks=True, device=0)[0]
    r1 = m.predict([t], imgsz=TILE, conf=0.3, augment=True, verbose=False,
                   retina_masks=True, device=0)[0]
    n0 = 0 if r0.masks is None else len(r0.masks.data)
    n1 = 0 if r1.masks is None else len(r1.masks.data)
    c0 = sorted(r0.boxes.conf.cpu().numpy().round(4).tolist())
    c1 = sorted(r1.boxes.conf.cpu().numpy().round(4).tolist())
    print(f"H1 TTA check: n_det augment=False {n0} / True {n1}")
    print(f"   confs equal? {c0 == c1}")
    print(f"   conf F: {c0[:6]}")
    print(f"   conf T: {c1[:6]}")

    # H2~H4: fold 전체 오차 위치 분석
    err_border, err_tileedge, err_inner = 0, 0, 0
    fp_rec, fn_rec = [], []
    per_file = collections.Counter()
    for fold in range(5):
        mm = YOLO(ckpt(fold))
        for f in vf(fold):
            img = cv2.imread(os.path.join(IMD, f))
            gi = np.load(os.path.join(INST, f.replace(".png", ".npy"))).astype(np.int32)
            cands, shape = predict_image(mm, img, conf=0.65, iou_nms=0.5)
            keep = merge_instances(cands, shape, 0.3)
            h, w = shape
            xs, ys = tile_origins(w, h)
            # 매칭
            gids = [int(v) for v in np.unique(gi) if v]
            pairs = []
            for i, p in enumerate(keep):
                pv = gi[p["full"]]
                vals, cnts = np.unique(pv[pv > 0], return_counts=True)
                for g, inter in zip(vals, cnts):
                    u = p["full"].sum() + int((gi == g).sum()) - int(inter)
                    if int(inter) / u >= 0.5:
                        pairs.append((int(inter) / u, i, int(g)))
            pairs.sort(reverse=True)
            up, ug = set(), set()
            for _, i, g in pairs:
                if i in up or g in ug:
                    continue
                up.add(i); ug.add(g)
            for g in gids:
                if g in ug:
                    continue
                ys_, xs_ = np.nonzero(gi == g)
                touch_img = (xs_.min() <= 2 or ys_.min() <= 2 or
                             xs_.max() >= w - 3 or ys_.max() >= h - 3)
                a = int((gi == g).sum())
                fn_rec.append(dict(file=f, area=a, border=bool(touch_img),
                                   cx=int(xs_.mean()), cy=int(ys_.mean())))
                per_file[f] -= 1
                if touch_img:
                    err_border += 1
                else:
                    err_inner += 1
            for i, p in enumerate(keep):
                if i in up:
                    continue
                ys_, xs_ = np.nonzero(p["full"])
                fp_rec.append(dict(file=f, area=int(p["full"].sum()),
                                   score=round(p["score"], 3),
                                   border=bool(xs_.min() <= 2 or ys_.min() <= 2 or
                                               xs_.max() >= w - 3 or ys_.max() >= h - 3),
                                   cx=int(xs_.mean()), cy=int(ys_.mean())))
                per_file[f] += 1
    print(f"\nH2/H3 FN total={len(fn_rec)} (영상경계 접촉 {err_border}, 내부 {err_inner})")
    print(f"      FP total={len(fp_rec)}")
    fa = np.array([r["area"] for r in fn_rec]) if fn_rec else np.array([0])
    pa = np.array([r["area"] for r in fp_rec]) if fp_rec else np.array([0])
    print(f"  FN area: med={np.median(fa):.0f} (GT med 2271)  "
          f"경계접촉비율={np.mean([r['border'] for r in fn_rec]):.2f}")
    print(f"  FP area: med={np.median(pa):.0f}  "
          f"경계접촉비율={np.mean([r['border'] for r in fp_rec]):.2f}  "
          f"score med={np.median([r['score'] for r in fp_rec]):.3f}")
    print("\nH4 파일별 순오차(상위):")
    for f, v in per_file.most_common(6):
        print(f"   {f} net={v:+d}")
    for f, v in per_file.most_common()[-6:]:
        print(f"   {f} net={v:+d}")
    json.dump(dict(fn=fn_rec, fp=fp_rec, per_file=dict(per_file)),
              open(f"{ROOT}/work/error_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
