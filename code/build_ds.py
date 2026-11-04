"""채운 인스턴스 라벨맵 -> YOLO-seg 타일 데이터셋 (5-fold)."""
import os, sys, pathlib, json, shutil, random
import pathlib
import numpy as np
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import (ROOT, IMD, INST, TILE, UPSCALE, IMGSZ, SEED,
                    MIN_VISIBLE_FRAC, MIN_INST_AREA, EMPTY_KEEP,
                    tile_origins, folds)


def inst_to_polys(inst_tile, full_areas):
    polys = []
    for v in np.unique(inst_tile):
        if v == 0:
            continue
        m = (inst_tile == v).astype(np.uint8)
        a = int(m.sum())
        if a < MIN_INST_AREA:
            continue
        if a / full_areas[v] < MIN_VISIBLE_FRAC:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        if len(c) < 3:
            continue
        c = c.astype(np.float32) * UPSCALE
        c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)
        if len(c) < 3:
            continue
        polys.append(np.clip(c.reshape(-1, 2) / IMGSZ, 0.0, 1.0))
    return polys


def build(fold):
    outdir = f"{ROOT}/work/ds_fold{fold}"
    rng = random.Random(SEED + fold)
    files = sorted(os.listdir(IMD))
    val_files = set(folds(files)[fold])
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            d = f"{outdir}/{sp}/{sub}"
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}
    for f in files:
        sp = "val" if f in val_files else "train"
        img = cv2.imread(os.path.join(IMD, f))
        inst = np.load(os.path.join(INST, f.replace(".png", ".npy"))).astype(np.int32)
        full_areas = {int(v): int((inst == v).sum()) for v in np.unique(inst) if v}
        h, w = inst.shape
        xs, ys = tile_origins(w, h)
        for oy in ys:
            for ox in xs:
                it = inst[oy:oy + TILE, ox:ox + TILE]
                polys = inst_to_polys(it, full_areas)
                if not polys and rng.random() > EMPTY_KEEP:
                    continue
                tl = img[oy:oy + TILE, ox:ox + TILE]
                if UPSCALE != 1:
                    tl = cv2.resize(tl, (IMGSZ, IMGSZ), interpolation=cv2.INTER_CUBIC)
                name = f"{f[:-4]}_{ox}_{oy}"
                cv2.imwrite(f"{outdir}/{sp}/images/{name}.png", tl)
                with open(f"{outdir}/{sp}/labels/{name}.txt", "w") as fh:
                    for p in polys:
                        fh.write("0 " + " ".join(f"{x:.6f}" for x in p.flatten()) + "\n")
                stats[sp][0] += 1
                stats[sp][1] += 0 if polys else 1
                stats[sp][2] += len(polys)
    with open(f"{outdir}/data.yaml", "w") as fh:
        fh.write(f"path: {outdir}\ntrain: train/images\nval: val/images\nnames:\n  0: cell\n")
    json.dump({"fold": fold, "val_files": sorted(val_files), "stats": stats,
               "tile": TILE, "imgsz": IMGSZ, "min_visible_frac": MIN_VISIBLE_FRAC},
              open(f"{outdir}/build_meta.json", "w"), indent=1)
    print(f"fold{fold}: " + "  ".join(
        f"{sp}=tiles {stats[sp][0]}(empty {stats[sp][1]}) inst {stats[sp][2]}"
        for sp in ("train", "val")))


if __name__ == "__main__":
    for k in (range(5) if len(sys.argv) < 2 else [int(sys.argv[1])]):
        build(k)
