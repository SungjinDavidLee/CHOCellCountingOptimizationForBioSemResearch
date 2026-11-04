"""FP/FN 사례를 크롭 이미지로 뽑아 GT 주석 품질을 검증한다.

FP confidence 중간값 0.805 (높음) -> 모델이 확신하는 검출이 GT 에 없다.
두 가지 가능성: (a) 모델 오검출, (b) GT 주석 누락.
크롭을 눈으로 확인하고, 추가로 정량 지표(국소 대비/텍스처)를 GT 세포와 비교한다.
"""
import os, sys, pathlib, json
import pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, IMD, GTD, INST

E = json.load(open(f"{ROOT}/work/error_analysis.json"))
OUT = f"{ROOT}/work/preview/errors"
os.makedirs(OUT, exist_ok=True)
R = 70


def crop(f, cx, cy):
    img = cv2.imread(os.path.join(IMD, f))
    gi = np.load(os.path.join(INST, f.replace(".png", ".npy"))).astype(np.int32)
    ring = cv2.imread(os.path.join(GTD, f), 0) > 127
    h, w = gi.shape
    y0, y1 = max(0, cy - R), min(h, cy + R)
    x0, x1 = max(0, cx - R), min(w, cx + R)
    sub = img[y0:y1, x0:x1].copy()
    sub[ring[y0:y1, x0:x1]] = (0, 255, 0)
    big = cv2.resize(sub, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    return big, img, gi, (x0, y0, x1, y1)


def stats(f, cx, cy, gi, img):
    """국소 대비/텍스처: 후보 영역 vs GT 세포 평균과 비교."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    rr = 27
    h, w = g.shape
    y0, y1 = max(0, cy - rr), min(h, cy + rr)
    x0, x1 = max(0, cx - rr), min(w, cx + rr)
    patch = g[y0:y1, x0:x1]
    ring = g[max(0, cy - 55):min(h, cy + 55), max(0, cx - 55):min(w, cx + 55)]
    return dict(std=float(patch.std()),
                contrast=float(patch.mean() - ring.mean()),
                lap=float(cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F).var()))


def main():
    rows = []
    for kind in ("fp", "fn"):
        for i, r in enumerate(sorted(E[kind], key=lambda x: -x.get("score", 0))):
            big, img, gi, _ = crop(r["file"], r["cx"], r["cy"])
            tag = f"{kind}{i:02d}_{r['file'][:-4]}_{r['cx']}_{r['cy']}"
            cv2.imwrite(f"{OUT}/{tag}.png", big)
            s = stats(r["file"], r["cx"], r["cy"], gi, img)
            rows.append(dict(kind=kind, tag=tag, area=r["area"],
                             score=r.get("score"), **s))
    # GT 세포들의 동일 지표 분포 (기준선)
    base = []
    for f in sorted(os.listdir(IMD))[:20]:
        img = cv2.imread(os.path.join(IMD, f))
        gi = np.load(os.path.join(INST, f.replace(".png", ".npy"))).astype(np.int32)
        for v in range(1, gi.max() + 1):
            ys, xs = np.nonzero(gi == v)
            if len(xs) == 0:
                continue
            base.append(stats(f, int(xs.mean()), int(ys.mean()), gi, img))
    for nm in ("std", "contrast", "lap"):
        b = np.array([x[nm] for x in base])
        fp = np.array([x[nm] for x in rows if x["kind"] == "fp"])
        fn = np.array([x[nm] for x in rows if x["kind"] == "fn"])
        print(f"{nm:9s} GT세포 med={np.median(b):8.2f}  "
              f"FP med={np.median(fp):8.2f}  FN med={np.median(fn):8.2f}")
    print(f"\nFP area med={np.median([r['area'] for r in rows if r['kind']=='fp']):.0f} "
          f"GT area med=2271")
    print(f"크롭 저장: {OUT} (FP {len(E['fp'])}개, FN {len(E['fn'])}개)")
    json.dump(rows, open(f"{ROOT}/work/error_stats.json", "w"), indent=1)
    # 몽타주
    for kind in ("fp", "fn"):
        imgs = [cv2.imread(f"{OUT}/{r['tag']}.png") for r in rows if r["kind"] == kind]
        imgs = [cv2.resize(i, (280, 280)) for i in imgs if i is not None]
        if not imgs:
            continue
        cols = 6
        while len(imgs) % cols:
            imgs.append(np.zeros((280, 280, 3), np.uint8))
        grid = np.vstack([np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)])
        cv2.imwrite(f"{ROOT}/work/preview/montage_{kind}.png", grid)
        print(f"몽타주: {ROOT}/work/preview/montage_{kind}.png")


if __name__ == "__main__":
    main()
