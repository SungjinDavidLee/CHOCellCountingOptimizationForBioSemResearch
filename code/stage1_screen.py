"""YOLOv8 / YOLOv11 seg 10종 전수 학습 + 후처리 스윕 (단계 1: fold0 스크리닝).

공정성 규칙
 - 동일 fold(0), 동일 epochs/patience, 동일 증강, 동일 imgsz, 동일 유효배치
 - 스윕 조합 전부 기록, 최선값만 보고하지 않음
 - fold0 val 에서 고른 후처리값은 '홀드아웃에서 선택된' 값 -> 선택 편향 명시
"""
import os, sys, pathlib, json, time, itertools, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT
from train_yolo import train

MODELS = [f"yolov8{s}-seg.pt" for s in "nsmlx"] + [f"yolo11{s}-seg.pt" for s in "nsmlx"]
FOLD = 0
EPOCHS = 300
CONFS = (0.25, 0.35, 0.45, 0.55, 0.65)
TWS = (None, 0.5, 0.65, 0.8)
DUPS = (0.3,)
OUT = f"{ROOT}/work/stage1_screen.json"


def main():
    from ultralytics import YOLO
    from evaluate import evaluate, summarize
    vf = json.load(open(f"{ROOT}/work/ds_fold{FOLD}/build_meta.json"))["val_files"]
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for w in MODELS:
        key = os.path.splitext(w)[0]
        if key in allres and allres[key].get("done"):
            print(f"[skip] {key}", flush=True)
            continue
        print(f"\n===== TRAIN {w} fold{FOLD} =====", flush=True)
        try:
            info = train(w, FOLD, epochs=EPOCHS)
        except Exception:
            traceback.print_exc()
            allres[key] = {"done": False, "error": "train_failed"}
            json.dump(allres, open(OUT, "w"), indent=1)
            continue
        m = YOLO(info["best"])
        nparam = sum(p.numel() for p in m.model.parameters())
        sweeps = []
        t0 = time.time()
        for conf, tw, du in itertools.product(CONFS, TWS, DUPS):
            rows = evaluate(m, vf, conf=conf, tw=tw, dup_iou=du)
            s = summarize(rows)
            s.update(conf=conf, tw=tw, dup_iou=du, tta=False)
            sweeps.append(s)
            print(f"  conf={conf} tw={tw} -> cnt_acc={s['count_acc']:.4f} "
                  f"mae={s['count_mae']:.3f} dice={s['dice']:.4f} "
                  f"miou={s['miou']:.4f} f1={s['inst_f1']:.4f} "
                  f"pred={s['pred_total']}/{s['gt_total']}", flush=True)
        best = max(sweeps, key=lambda r: (r["count_acc"], r["dice"]))
        allres[key] = dict(done=True, model=w, fold=FOLD, best_ckpt=info["best"],
                           train_sec=info["train_sec"], params=int(nparam),
                           sweep_sec=round(time.time() - t0, 1),
                           sweeps=sweeps, best=best)
        json.dump(allres, open(OUT, "w"), indent=1)
        print(f"[{key}] BEST cnt_acc={best['count_acc']:.4f} dice={best['dice']:.4f} "
              f"miou={best['miou']:.4f} f1={best['inst_f1']:.4f} "
              f"@conf={best['conf']} tw={best['tw']}", flush=True)
    print("\n===== STAGE1 SUMMARY =====", flush=True)
    rank = sorted((v for v in allres.values() if v.get("done")),
                  key=lambda v: -v["best"]["count_acc"])
    for v in rank:
        b = v["best"]
        print(f"{v['model']:18s} params={v['params']/1e6:5.1f}M "
              f"cnt_acc={b['count_acc']:.4f} dice={b['dice']:.4f} "
              f"miou={b['miou']:.4f} f1={b['inst_f1']:.4f} "
              f"conf={b['conf']} tw={b['tw']} train={v['train_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
