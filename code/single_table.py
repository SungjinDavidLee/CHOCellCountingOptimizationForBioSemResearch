"""단독 모델 성능표 생성 (앙상블 제외).

증거 등급을 명확히 구분한다:
  A등급 = nested 5-fold CV (완전 홀드아웃, 743 세포 전부 평가)  -> 4종
  B등급 = fold0 단일 스크리닝 (선택 편향 포함 상한, 155 세포)   -> 6종
같은 표에 섞어 쓰되 등급 열을 반드시 둔다.
"""
import os, sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT

ORDER = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
         "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"]


def main():
    s1 = json.load(open(f"{ROOT}/work/stage1_screen.json"))
    cvr = json.load(open(f"{ROOT}/work/cv_report.json"))
    lat = {}
    p = f"{ROOT}/work/latency.json"
    if os.path.exists(p):
        lat = json.load(open(p))

    rows = []
    for k in ORDER:
        s = s1.get(k, {})
        if not s.get("done"):
            continue
        b = s["best"]
        r = dict(model=k, params=s["params"], train_sec=s["train_sec"],
                 ckpt_mb=None, grade="B",
                 acc=b["count_acc"], acc_sd=None, ci=None,
                 dice=b["dice"], miou=b["miou"], f1=b["inst_f1"],
                 mae=b["count_mae"], conf=b["conf"],
                 pix_acc=b.get("pix_acc"), r2=b.get("count_r2"),
                 nfolds=1, gt=b["gt_total"])
        if k in cvr:
            n = cvr[k]["nested"]
            r.update(grade="A", acc=n["count_acc"], acc_sd=n["count_acc_sd"],
                     ci=n["count_acc_ci"], dice=n["dice"], miou=n["miou"],
                     f1=n["inst_f1"], mae=n["count_mae"], pix_acc=n["pix_acc"],
                     nfolds=5, gt=n["gt_total"],
                     conf=cvr[k]["nested"]["per_fold"][0]["sel_conf"])
            r["acc_fixed"] = cvr[k]["fixed"]["count_acc"]
            r["acc_oracle"] = cvr[k]["oracle"]["count_acc"]
        cp = f"{ROOT}/work/runs/{k}_f0/weights/best.pt"
        if os.path.exists(cp):
            r["ckpt_mb"] = round(os.path.getsize(cp) / 1e6, 1)
        if k in lat:
            r.update(lat[k])
        rows.append(r)

    json.dump(rows, open(f"{ROOT}/work/single_model_table.json", "w"), indent=1)

    out = []
    A = out.append
    A("=" * 100)
    A("단독 모델 성능표 — YOLOv8 / YOLOv11 seg 10종 (앙상블 제외)")
    A("=" * 100)
    A("")
    nA = sum(1 for r in rows if r["grade"] == "A")
    nB = len(rows) - nA
    A("증거 등급  A = 중첩 5-fold 교차검증 (완전 홀드아웃, 세포 743개 전부 평가)")
    if nB:
        A("           B = fold0 단일 스크리닝 (후처리 conf 를 같은 fold 에서 선택 = 상한, 세포 155개)")
        A("           등급이 다른 행을 직접 비교하면 안 된다. B 는 체계적으로 낙관적이다.")
        A("           (실측 격차: yolo11s B 0.9806 vs A 0.9690, 1.2 %p)")
    else:
        A(f"           전 {nA}종 모두 A등급 — 동일 조건에서 직접 비교 가능.")
        A("           (참고: fold0 단일 스크리닝 값은 체계적으로 낙관적이었음.")
        A("            실측 격차 yolo11s 스크리닝 0.9806 vs 교차검증 0.9690, 1.2 %p)")
    A("")
    hdr = (f"{'모델':14s} {'등급':4s} {'파라미터':>9s} {'ckpt':>7s} "
           f"{'계수정확도':>16s} {'Dice':>7s} {'mIoU':>7s} {'instF1':>7s} {'MAE':>6s} {'conf':>5s}")
    A(hdr)
    A("-" * 100)
    for r in rows:
        acc = (f"{r['acc']:.4f}+-{r['acc_sd']:.4f}" if r["acc_sd"] is not None
               else f"{r['acc']:.4f}          ")
        A(f"{r['model']:14s} {r['grade']:^4s} {r['params']/1e6:8.1f}M "
          f"{r['ckpt_mb']:6.1f}M {acc:>16s} {r['dice']:7.4f} {r['miou']:7.4f} "
          f"{r['f1']:7.4f} {r['mae']:6.3f} {r['conf']:5.2f}")
    A("")
    nA = sum(1 for r in rows if r["grade"] == "A")
    A(f"A등급 {nA}종 상세 (3가지 선택 방식 비교)")
    A("-" * 100)
    A(f"{'모델':14s} {'fixed':>8s} {'nested':>18s} {'oracle':>8s}  {'95% CI (nested)':>20s}")
    for r in rows:
        if r["grade"] != "A":
            continue
        A(f"{r['model']:14s} {r['acc_fixed']:8.4f} "
          f"{r['acc']:.4f}+-{r['acc_sd']:.4f}  {r['acc_oracle']:8.4f}  "
          f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")
    A("")
    A("  fixed  = conf 를 fold0 에서 골라 전 fold 적용")
    A("  nested = fold k 의 conf 를 k 제외 4개 fold 에서만 선택  <== 헤드라인 수치")
    A("  oracle = 각 fold 자기 자신에서 선택 (낙관 편향, 상한)")
    A("")
    if any("cpu_ms" in r for r in rows):
        A("추론 지연시간 (영상 1장 = 1376x1032, 타일 9개 일괄 처리)")
        A("-" * 100)
        A(f"{'모델':14s} {'ckpt':>7s} {'GPU ms':>9s} {'GPU FPS':>8s} "
          f"{'CPU ms':>9s} {'CPU FPS':>8s}")
        for r in rows:
            if "cpu_ms" not in r:
                continue
            A(f"{r['model']:14s} {r['ckpt_mb']:6.1f}M {r['gpu_ms']:9.1f} "
              f"{1000/r['gpu_ms']:8.2f} {r['cpu_ms']:9.1f} {1000/r['cpu_ms']:8.2f}")
        A("")
    txt = "\n".join(out)
    open(f"{ROOT}/work/SINGLE_MODEL_TABLE.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
