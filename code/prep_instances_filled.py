"""CHO 세포 GT(1px 윤곽선) -> 채워진 인스턴스 라벨맵.

[중요] 이 데이터셋의 ground_truth PNG 는 채워진 영역 마스크가 아니라
세포 외곽선(두께 ~1.19 px)이다. 실측 근거:
  - ring/filled 면적비 중간값 0.089
  - 채운 뒤 등가직경 중간값 53.8 px (윤곽선만 쓰면 16.0 px 로 오산)
  - 채운 뒤 전경 비율 2.14% (윤곽선 기준 0.2%)
따라서 반드시 윤곽을 채운 뒤 사용한다. 윤곽이 끊긴 세포(실측 20개, 2.7%)는
gt_fill.fill_outlines() 가 dilate-fill-erode / convexHull 로 단계 보정한다.

접촉 세포 분리: Tw 0.4~0.9 전 구간에서 추가 분리가 0건임을 실측 확인했으므로
연결요소 = 인스턴스. (논문과 동일한 distance watershed 를 적용해도 결과 동일)

출력: $CHO_ROOT/work/instances_filled/*.npy (uint16 인스턴스 id 맵)
"""
import os, sys, json, pathlib
import numpy as np
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, IMD, GTD, INST
from gt_fill import gt_instances


def main():
    os.makedirs(INST, exist_ok=True)
    files = sorted(os.listdir(IMD))
    meta, areas, unfilled = {}, [], []
    for f in files:
        inst, n = gt_instances(os.path.join(GTD, f))
        np.save(os.path.join(INST, f.replace(".png", ".npy")),
                inst.astype(np.uint16))
        meta[f] = n
        ring = (cv2.imread(os.path.join(GTD, f), 0) > 127).astype(np.uint8)
        for v in range(1, n + 1):
            m = inst == v
            a = int(m.sum())
            areas.append(a)
            if int(ring[m].sum()) / a > 0.35:
                unfilled.append((f, v, a))
    json.dump(meta, open(f"{ROOT}/work/instance_counts_filled.json", "w"), indent=1)

    c = np.array(list(meta.values()))
    a = np.array(areas)
    d = 2 * (a / np.pi) ** 0.5
    cov = []
    for f in files:
        inst = np.load(os.path.join(INST, f.replace(".png", ".npy")))
        cov.append((inst > 0).mean())
    print(f"files={len(files)} total_instances={a.size}")
    print(f"per-img: min={c.min()} med={np.median(c):.0f} max={c.max()} "
          f"mean={c.mean():.2f}")
    print(f"area: med={np.median(a):.0f} p5={np.percentile(a,5):.0f} "
          f"p95={np.percentile(a,95):.0f}")
    print(f"equiv diam px: med={np.median(d):.1f} p5={np.percentile(d,5):.1f} "
          f"p95={np.percentile(d,95):.1f} max={d.max():.1f}")
    print(f"foreground coverage: mean={np.mean(cov)*100:.2f}% "
          f"({min(cov)*100:.2f}-{max(cov)*100:.2f}%)")
    print(f"still-unfilled objects: {len(unfilled)}   <- 0 이어야 정상")
    for r in unfilled[:10]:
        print("   ", r)


if __name__ == "__main__":
    main()
