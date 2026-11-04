"""추론 지연시간 실측 — 클리카 요청(목표 FPS, CPU/NPU 분배)에 답하기 위함.

프로토콜 (동결)
 - 입력: 실제 영상 1장 1376x1032 -> 512 타일 9개 -> 일괄 추론
 - warmup 5회 후 20회 측정, 중간값 보고 (평균은 이상치에 끌림)
 - 전처리/후처리(타일 자르기, 마스크 병합) 포함 end-to-end 와
   순수 모델 forward 를 분리해 보고
 - GPU: RTX 5090 FP32 / FP16, CPU: 동일 머신 x86
 - 주의: 이 수치는 **개발 머신** 기준이며 라즈베리파이5+DX-M1 실기 수치가 아니다.
   NPU 실측은 변환 후 협업사/실기에서만 측정 가능.
"""
import os, sys, pathlib, json, time
import pathlib
import numpy as np
import cv2
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, IMD, TILE, IMGSZ, tile_origins

MODELS = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
          "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"]
WARMUP, REPS = 5, 20


def make_tiles(img):
    h, w = img.shape[:2]
    xs, ys = tile_origins(w, h)
    return [img[oy:oy + TILE, ox:ox + TILE] for oy in ys for ox in xs]


def bench(model, tiles, device, half, reps=REPS):
    for _ in range(WARMUP):
        model.predict(tiles, imgsz=IMGSZ, conf=0.55, device=device, half=half,
                      verbose=False, retina_masks=True)
    ts = []
    for _ in range(reps):
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(tiles, imgsz=IMGSZ, conf=0.55, device=device, half=half,
                      verbose=False, retina_masks=True)
        if device != "cpu":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 5)), float(np.percentile(ts, 95))


def main():
    from ultralytics import YOLO
    img = cv2.imread(os.path.join(IMD, sorted(os.listdir(IMD))[0]))
    tiles = make_tiles(img)
    print(f"입력 1376x1032 -> 타일 {len(tiles)}개 x {TILE}x{TILE}, imgsz={IMGSZ}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    out = {}
    for k in MODELS:
        cp = f"{ROOT}/work/runs/{k}_f0/weights/best.pt"
        if not os.path.exists(cp):
            print(f"[skip] {k}")
            continue
        m = YOLO(cp)
        nparam = sum(p.numel() for p in m.model.parameters())
        g32, g32lo, g32hi = bench(m, tiles, 0, False)
        g16, _, _ = bench(m, tiles, 0, True)
        c, clo, chi = bench(m, tiles, "cpu", False, reps=5)
        out[k] = dict(gpu_ms=round(g32, 2), gpu_fp16_ms=round(g16, 2),
                      cpu_ms=round(c, 2), gpu_p5=round(g32lo, 2),
                      gpu_p95=round(g32hi, 2), params=int(nparam),
                      ckpt_mb=round(os.path.getsize(cp) / 1e6, 1),
                      tiles=len(tiles))
        print(f"{k:14s} GPU32 {g32:8.1f}ms ({1000/g32:6.2f} fps)  "
              f"GPU16 {g16:8.1f}ms ({1000/g16:6.2f} fps)  "
              f"CPU {c:9.1f}ms ({1000/c:6.2f} fps)", flush=True)
        json.dump(out, open(f"{ROOT}/work/latency.json", "w"), indent=1)
    print(f"\n저장: {ROOT}/work/latency.json")


if __name__ == "__main__":
    main()
