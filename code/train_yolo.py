"""YOLOv8/v11-seg 학습 드라이버.

공정성: 모든 모델에 동일 분할(fold), 동일 epochs/patience, 동일 증강, 동일 imgsz.
batch 는 모델 크기에 따라 VRAM 한계로 조정되나, 유효 배치는 동일하게 유지한다.
"""
import os, sys, pathlib, json, time
from ultralytics import YOLO
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, IMGSZ

EPOCHS = 300
PATIENCE = 80
EFF_BATCH = 16   # 모든 모델 공통 유효 배치

# 모델별 물리 배치 (나머지는 grad accumulation 으로 유효배치 16 맞춤)
PHYS_BATCH = {"n": 16, "s": 16, "m": 16, "l": 8, "x": 8}

COMMON = dict(
    imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, device=0, workers=8,
    optimizer="AdamW", lr0=1e-3, lrf=0.01, warmup_epochs=5, cos_lr=True,
    weight_decay=5e-4,
    # 증강: 세포는 회전 불변 -> 회전/반전 적극. 그레이스케일이라 색상 증강 최소.
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=180.0, translate=0.1, scale=0.3,
    shear=0.0, perspective=0.0, flipud=0.5, fliplr=0.5,
    mosaic=0.5, mixup=0.0, copy_paste=0.3, close_mosaic=25,
    overlap_mask=False, mask_ratio=1,
    val=True, plots=False, save=True, exist_ok=True, deterministic=False, seed=0,
    amp=True, cache="disk",
)


def size_of(weights):
    base = os.path.basename(weights)
    for s in ("n", "s", "m", "l", "x"):
        if base.startswith(f"yolov8{s}-seg") or base.startswith(f"yolo11{s}-seg"):
            return s
    return "m"


def train(weights, fold, epochs=None, name=None):
    data = f"{ROOT}/work/ds_fold{fold}/data.yaml"
    sz = size_of(weights)
    pb = PHYS_BATCH[sz]
    name = name or f"{os.path.splitext(os.path.basename(weights))[0]}_f{fold}"
    kw = dict(COMMON)
    kw["batch"] = pb
    kw["nbs"] = EFF_BATCH
    if epochs:
        kw["epochs"] = epochs
    t0 = time.time()
    m = YOLO(weights)
    m.train(data=data, project=f"{ROOT}/work/runs", name=name, **kw)
    out = dict(model=os.path.basename(weights), fold=fold, name=name,
               best=f"{ROOT}/work/runs/{name}/weights/best.pt",
               train_sec=round(time.time() - t0, 1), phys_batch=pb)
    print("TRAIN_DONE " + json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    w = sys.argv[1]
    fold = int(sys.argv[2])
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else None
    train(w, fold, epochs=ep)
