"""논문/보고서용 그림 생성. --lang ko|en 으로 한/영 모두 출력.

저널 투고용 그림은 영문이어야 하므로 문자열을 분리해 두었다.
영문에서는 %p 대신 pp 표기를 쓴다.
"""
import os, sys, pathlib, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT, IMD, INST

L = {
    "ko": dict(
        f1_t="모델별 세포 계수 정확도 (fold0 스크리닝)",
        f1_x="모델 파라미터 수 (M)", f1_y="계수 정확도",
        f2_t="세포 수: 예측 vs 실측 (5-fold 교차검증)",
        f2_x="실측 세포 수", f2_y="예측 세포 수",
        f3_t="Bland-Altman 분석", f3_x="평균 세포 수 ((예측+실측)/2)",
        f3_y="차이 (예측 − 실측)",
        f4_t="정성 결과: 녹색=실측 윤곽, 빨강=예측 윤곽",
        f5_t="방법별 계수 정확도 (중첩 교차검증, 완전 홀드아웃)",
        f5_y="계수 정확도",
        f6_t="데이터셋 특성", target="목표 97%", mean="평균",
        loa="일치 한계 (±1.96σ)", ideal="이상선 (y=x)",
        f6a="세포 수 분포 (장당)", f6b="세포 등가직경 분포",
        f6c="영상 밝기 분포", f6d="전경 비율 분포",
        cells="세포 수", diam="등가직경 (px)", bright="평균 밝기 (계조)",
        fgpct="전경 비율 (%)", nimg="영상 수", ncell="세포 수",
    ),
    "en": dict(
        f1_t="Cell counting accuracy by model (fold-0 screening)",
        f1_x="Model parameters (M)", f1_y="Counting accuracy",
        f2_t="Predicted vs. ground-truth cell count (5-fold CV)",
        f2_x="Ground-truth cell count", f2_y="Predicted cell count",
        f3_t="Bland-Altman analysis", f3_x="Mean count ((pred+GT)/2)",
        f3_y="Difference (pred - GT)",
        f4_t="Qualitative results: green = GT contour, red = prediction",
        f5_t="Counting accuracy by method (nested CV, full hold-out)",
        f5_y="Counting accuracy",
        f6_t="Dataset characteristics", target="Target 97%", mean="Mean",
        loa="Limits of agreement (+/-1.96 SD)", ideal="Identity (y=x)",
        f6a="Cells per image", f6b="Cell equivalent diameter",
        f6c="Image mean intensity", f6d="Foreground fraction",
        cells="Cells per image", diam="Equivalent diameter (px)",
        bright="Mean intensity (grey level)", fgpct="Foreground fraction (%)",
        nimg="Number of images", ncell="Number of cells",
    ),
}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10})


def setup_font(lang):
    if lang == "ko":
        for f in ("NanumGothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"):
            try:
                matplotlib.font_manager.findfont(f, fallback_to_default=False)
                plt.rcParams["font.family"] = f
                break
            except Exception:
                continue
        plt.rcParams["axes.unicode_minus"] = False


def fig_models(t, out):
    s1 = json.load(open(f"{ROOT}/work/stage1_screen.json"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, fam in enumerate(("yolov8", "yolo11")):
        items = [(k, v) for k, v in s1.items() if k.startswith(fam) and v.get("done")]
        items.sort(key=lambda kv: kv[1]["params"])
        x = [v["params"] / 1e6 for _, v in items]
        acc = [v["best"]["count_acc"] for _, v in items]
        dice = [v["best"]["dice"] for _, v in items]
        lbl = [k.replace("-seg", "").replace(fam, "") for k, _ in items]
        ax[0].plot(x, acc, "o-", label=fam.upper())
        ax[1].plot(x, dice, "s-", label=fam.upper())
        for xi, yi, li in zip(x, acc, lbl):
            ax[0].annotate(li, (xi, yi), textcoords="offset points",
                           xytext=(0, 6), ha="center", fontsize=8)
    ax[0].axhline(0.97, ls="--", c="r", lw=1, label=t["target"])
    ax[0].set(xlabel=t["f1_x"], ylabel=t["f1_y"], title=t["f1_t"], xscale="log")
    ax[1].set(xlabel=t["f1_x"], ylabel="Dice", xscale="log")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_scatter_ba(t, out_s, out_b, per_fold):
    gtv, pdv = [], []
    for r in per_fold:
        for pf in r.get("per_file", []):
            gtv.append(pf["n_gt"]); pdv.append(pf["n_pred"])
    gtv, pdv = np.array(gtv), np.array(pdv)
    if gtv.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    jitter = (np.random.default_rng(0).random(gtv.size) - .5) * .25
    ax.scatter(gtv + jitter, pdv + jitter, s=26, alpha=.65, edgecolor="k", lw=.4)
    lim = [min(gtv.min(), pdv.min()) - 1, max(gtv.max(), pdv.max()) + 1]
    ax.plot(lim, lim, "k--", lw=1, label=t["ideal"])
    r2 = 1 - ((pdv - gtv) ** 2).sum() / ((gtv - gtv.mean()) ** 2).sum()
    ax.set(xlabel=t["f2_x"], ylabel=t["f2_y"], title=t["f2_t"], xlim=lim, ylim=lim)
    ax.legend(fontsize=8)
    ax.text(.05, .93, f"$R^2$ = {r2:.3f}\nn = {gtv.size}", transform=ax.transAxes,
            va="top", fontsize=9,
            bbox=dict(fc="w", alpha=.8, ec="gray", lw=.5))
    fig.tight_layout(); fig.savefig(out_s); plt.close(fig)

    d = pdv - gtv
    m, sd = d.mean(), d.std(ddof=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.scatter((pdv + gtv) / 2 + jitter, d + jitter, s=26, alpha=.65,
               edgecolor="k", lw=.4)
    ax.axhline(m, c="b", lw=1.2, label=f"{t['mean']} = {m:+.3f}")
    ax.axhline(m + 1.96 * sd, c="r", ls="--", lw=1, label=t["loa"])
    ax.axhline(m - 1.96 * sd, c="r", ls="--", lw=1)
    ax.axhline(0, c="k", lw=.6, alpha=.5)
    ax.set(xlabel=t["f3_x"], ylabel=t["f3_y"], title=t["f3_t"])
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_b); plt.close(fig)
    return dict(r2=float(r2), bias=float(m), sd=float(sd),
                loa=[float(m - 1.96 * sd), float(m + 1.96 * sd)], n=int(gtv.size))


def fig_dataset(t, out):
    fs = sorted(os.listdir(IMD))
    nc, dia, bri, fg = [], [], [], []
    for f in fs:
        gi = np.load(os.path.join(INST, f.replace(".png", ".npy"))).astype(np.int32)
        img = cv2.imread(os.path.join(IMD, f), 0)
        n = int(gi.max()); nc.append(n)
        bri.append(float(img.mean())); fg.append(float((gi > 0).mean() * 100))
        for v in range(1, n + 1):
            a = int((gi == v).sum())
            if a:
                dia.append(2 * (a / np.pi) ** .5)
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
    ax[0, 0].hist(nc, bins=range(min(nc), max(nc) + 2), color="steelblue",
                  edgecolor="k", lw=.5)
    ax[0, 0].set(title=t["f6a"], xlabel=t["cells"], ylabel=t["nimg"])
    ax[0, 1].hist(dia, bins=30, color="seagreen", edgecolor="k", lw=.5)
    ax[0, 1].set(title=t["f6b"], xlabel=t["diam"], ylabel=t["ncell"])
    ax[0, 1].axvline(np.median(dia), c="r", ls="--", lw=1,
                     label=f"median {np.median(dia):.1f}")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].hist(bri, bins=20, color="darkorange", edgecolor="k", lw=.5)
    ax[1, 0].set(title=t["f6c"], xlabel=t["bright"], ylabel=t["nimg"])
    ax[1, 1].hist(fg, bins=20, color="purple", edgecolor="k", lw=.5)
    ax[1, 1].set(title=t["f6d"], xlabel=t["fgpct"], ylabel=t["nimg"])
    fig.suptitle(t["f6_t"])
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return dict(cells_per_img=[int(min(nc)), float(np.median(nc)), int(max(nc))],
                diam=[float(np.percentile(dia, 5)), float(np.median(dia)),
                      float(np.percentile(dia, 95))],
                brightness=[float(min(bri)), float(np.mean(bri)), float(max(bri))],
                fg_pct=[float(min(fg)), float(np.mean(fg)), float(max(fg))])


def fig_methods(t, out, rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    cols = ["#4C72B0" if v < .97 else "#55A868" for v in vals]
    ax.bar(range(len(rows)), vals, yerr=errs, capsize=4, color=cols,
           edgecolor="k", lw=.6)
    ax.axhline(0.97, ls="--", c="r", lw=1.2, label=t["target"])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel=t["f5_y"], title=t["f5_t"], ylim=(0.93, 1.0))
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + .002, f"{v:.4f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_qual(t, out, key="yolo11s-seg", conf=0.55):
    from ultralytics import YOLO
    from evaluate import predict_image, merge_instances
    picks = [("cho01.png", 0), ("cho14.png", 3), ("cho40.png", 4), ("cho46.png", 0)]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4.3))
    for ax, (f, fold) in zip(np.atleast_1d(axes), picks):
        ck = f"{ROOT}/work/runs/{key}_f{fold}/weights/best.pt"
        m = YOLO(ck)
        img = cv2.imread(os.path.join(IMD, f))
        cands, shape = predict_image(m, img, conf=conf, iou_nms=0.5)
        keep = merge_instances(cands, shape, 0.3)
        gi = np.load(os.path.join(INST, f.replace(".png", ".npy"))).astype(np.int32)
        vis = img.copy()
        cs, _ = cv2.findContours((gi > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0), 2)
        for k in keep:
            cs, _ = cv2.findContours(k["full"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 0, 255), 2)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{f}  GT={int(gi.max())}  pred={len(keep)}", fontsize=9)
        ax.axis("off")
    fig.suptitle(t["f4_t"])
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["ko", "en"])
    a = ap.parse_args()
    t = L[a.lang]
    setup_font(a.lang)
    od = f"{ROOT}/work/figures/{a.lang}"
    os.makedirs(od, exist_ok=True)
    meta = {}
    fig_models(t, f"{od}/fig1_model_comparison.png")
    meta["dataset"] = fig_dataset(t, f"{od}/fig2_dataset.png")

    s3b = json.load(open(f"{ROOT}/work/stage3b.json"))
    cvr = json.load(open(f"{ROOT}/work/cv_report.json"))
    best_tag = max(s3b, key=lambda k: s3b[k]["nested"]["count_acc"])
    rows = []
    for k, v in cvr.items():
        rows.append((f"{k}\n(single, nested)", v["nested"]["count_acc"],
                     v["nested"]["count_acc_sd"]))
    for k, v in s3b.items():
        rows.append((f"{k}\n(nested)", v["nested"]["count_acc"], v["nested"]["sd"]))
    rows.sort(key=lambda r: r[1])
    fig_methods(t, f"{od}/fig3_methods.png", rows)

    # scatter/BA 는 per-file 기록이 있는 단일모델 결과로 생성
    s2 = json.load(open(f"{ROOT}/work/stage2_cv.json"))
    key = "yolo11s-seg"
    pf2 = [dict(per_file=s2[key]["folds"][str(i)]["per_file"]) for i in range(5)]
    meta["agreement"] = fig_scatter_ba(t, f"{od}/fig4_scatter.png",
                                       f"{od}/fig5_bland_altman.png", pf2)
    fig_qual(t, f"{od}/fig6_qualitative.png")
    meta["best_tag"] = best_tag
    json.dump(meta, open(f"{od}/figure_meta.json", "w"), indent=1)
    print(f"[{a.lang}] wrote figures to {od}")
    for f in sorted(os.listdir(od)):
        print("  ", f)
    print(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
