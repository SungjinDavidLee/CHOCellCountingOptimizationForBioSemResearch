"""단계 2: 상위 모델 5-fold 교차검증.

핵심: fold0 스크리닝에서 고른 후처리값(conf/tw)은 '그 fold의 홀드아웃에서
선택된' 값이므로 선택 편향이 있다. 여기서는 두 가지를 모두 산출한다.

 A) fixed  : conf/tw 를 fold0 에서 고른 값으로 고정하고 fold1~4 에서 평가
             -> 완전 홀드아웃. 이것이 정직한 일반화 수치.
 B) percv  : 각 fold 에서 다시 스윕한 값 (상한, 선택 편향 포함)

보고서에는 A 를 헤드라인으로, B 를 상한으로 명시한다.
"""
import os, sys, pathlib, json, itertools
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, NFOLD
from train_yolo import train

CONFS = (0.25, 0.35, 0.45, 0.55, 0.65)
TWS = (None, 0.5)
OUT = f"{ROOT}/work/stage2_cv.json"


def val_files(fold):
    return json.load(open(f"{ROOT}/work/ds_fold{fold}/build_meta.json"))["val_files"]


def run(models, epochs=300):
    from ultralytics import YOLO
    from evaluate import evaluate, summarize
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    s1 = json.load(open(f"{ROOT}/work/stage1_screen.json"))
    for w in models:
        key = os.path.splitext(w)[0]
        fixed_conf = s1[key]["best"]["conf"]
        fixed_tw = s1[key]["best"]["tw"]
        res.setdefault(key, {"model": w, "fixed_conf": fixed_conf,
                             "fixed_tw": fixed_tw, "folds": {}})
        for fold in range(NFOLD):
            fk = str(fold)
            if fk in res[key]["folds"]:
                print(f"[skip] {key} fold{fold}", flush=True)
                continue
            name = f"{key}_f{fold}"
            ckpt = f"{ROOT}/work/runs/{name}/weights/best.pt"
            if not os.path.exists(ckpt):
                print(f"\n===== TRAIN {w} fold{fold} =====", flush=True)
                train(w, fold, epochs=epochs)
            m = YOLO(ckpt)
            vf = val_files(fold)
            # A) 고정 후처리
            rowsA = evaluate(m, vf, conf=fixed_conf, tw=fixed_tw)
            A = summarize(rowsA)
            A.update(conf=fixed_conf, tw=fixed_tw)
            # B) fold 내 재스윕 (상한)
            sw = []
            for conf, tw in itertools.product(CONFS, TWS):
                if (conf, tw) == (fixed_conf, fixed_tw):
                    s = dict(A)
                else:
                    s = summarize(evaluate(m, vf, conf=conf, tw=tw))
                    s.update(conf=conf, tw=tw)
                sw.append(s)
            B = max(sw, key=lambda r: (r["count_acc"], r["dice"]))
            res[key]["folds"][fk] = dict(fixed=A, oracle=B, sweeps=sw,
                                         per_file=rowsA, ckpt=ckpt)
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"[{key} fold{fold}] fixed: cnt_acc={A['count_acc']:.4f} "
                  f"dice={A['dice']:.4f} miou={A['miou']:.4f} f1={A['inst_f1']:.4f} "
                  f"| oracle: cnt_acc={B['count_acc']:.4f} @conf={B['conf']} tw={B['tw']}",
                  flush=True)
        # fold 집계
        fl = res[key]["folds"]
        if len(fl) == NFOLD:
            for mode in ("fixed", "oracle"):
                acc = [fl[str(i)][mode]["count_acc"] for i in range(NFOLD)]
                dic = [fl[str(i)][mode]["dice"] for i in range(NFOLD)]
                f1 = [fl[str(i)][mode]["inst_f1"] for i in range(NFOLD)]
                print(f"  == {key} CV[{mode}] cnt_acc={np.mean(acc):.4f}±{np.std(acc):.4f} "
                      f"dice={np.mean(dic):.4f}±{np.std(dic):.4f} "
                      f"f1={np.mean(f1):.4f}±{np.std(f1):.4f}", flush=True)
    return res


if __name__ == "__main__":
    models = sys.argv[1:] or ["yolo11s-seg.pt"]
    run(models)
