"""최종 결과 집계 + 보고서 생성 (한글).

정직성 규칙
 - nested CV 수치를 헤드라인으로 쓴다 (fold k 의 하이퍼파라미터는 k 제외 fold 에서 선택).
 - fold0 단일 수치는 '선택 편향 포함 상한'으로만 표기한다.
 - 미검증 항목은 그대로 '미검증'이라 쓴다.
"""
import os, sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT


def load(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def main():
    s1 = load(f"{ROOT}/work/stage1_screen.json", {})
    cvr = load(f"{ROOT}/work/cv_report.json", {})
    s3b = load(f"{ROOT}/work/stage3b.json", {})
    counts = load(f"{ROOT}/work/instance_counts_filled.json", {})
    figmeta = load(f"{ROOT}/work/figures/en/figure_meta.json", {})

    out = []
    A = out.append
    A("=" * 78)
    A("CHO 세포 이미지 데이터셋 — YOLOv8/YOLOv11 세그멘테이션 및 세포 계수 결과")
    A("=" * 78)
    A("")
    A("[1] 데이터셋 특성 (전수 실측)")
    A("-" * 78)
    ds = figmeta.get("dataset", {})
    A(f"  영상 수            : {len(counts)} 장 (cho01~cho60.png)")
    A(f"  해상도             : 1376 x 1032, PNG, R=G=B (실질 8-bit 그레이스케일)")
    A(f"  GT 주석 형식       : 세포 외곽선 (두께 ~1.19 px) — 채워진 영역 마스크가 아님")
    A(f"                       * 윤곽 채움 전처리 필수. 미채움 객체 0건 확인")
    A(f"  총 세포 수         : {sum(counts.values())} 개")
    if ds:
        c = ds["cells_per_img"]; d = ds["diam"]; b = ds["brightness"]; g = ds["fg_pct"]
        A(f"  장당 세포 수       : 최소 {c[0]}, 중간값 {c[1]:.0f}, 최대 {c[2]}")
        A(f"  세포 등가직경      : p5 {d[0]:.1f} / 중간값 {d[1]:.1f} / p95 {d[2]:.1f} px")
        A(f"  영상 평균 밝기     : {b[0]:.1f} ~ {b[2]:.1f} (평균 {b[1]:.1f})")
        A(f"  전경 비율          : {g[0]:.2f} ~ {g[2]:.2f} % (평균 {g[1]:.2f} %)")
    A(f"  세포-배경 콘트라스트: 약 6 계조 → 무염색(label-free) 저콘트라스트 명시야 계열")
    A(f"  접촉/겹침 세포      : Tw 0.4~0.9 전 구간에서 watershed 추가분리 0건")
    A("")
    A("  [주의] 전경이 2.14% 뿐이라 '픽셀 Accuracy' 는 아무 모델이나 99%+ 가 나온다.")
    A("         따라서 Dice / mIoU / 인스턴스 F1 / 계수정확도를 주 지표로 썼다.")
    A("")

    A("[2] 실험 설계")
    A("-" * 78)
    A("  타일링      : 512 x 512, stride 432 (오버랩 80 px), 영상당 3x3 = 9 타일")
    A("  추론 병합   : 타일별 인스턴스 -> 마스크 IoU 기반 중복 제거 -> 원본 좌표계 평가")
    A("  분할        : 파일 단위 5-fold (세포 단위 분할은 누출이므로 금지)")
    A("  공정성      : 전 모델 동일 fold / 300 epoch / patience 80 / 동일 증강 /")
    A("                동일 유효배치 16 / imgsz 512 / AdamW lr0 1e-3 cos")
    A("  계수 평가식 : Acc = 1 - MAE / mean(N_GT)   (참고 논문과 동일)")
    A("")

    A("[3] 10개 모델 전수 비교 (fold0 스크리닝)")
    A("-" * 78)
    A(f"  {'모델':16s} {'파라미터':>9s} {'계수Acc':>8s} {'Dice':>7s} {'mIoU':>7s} "
      f"{'instF1':>7s} {'학습(s)':>8s}")
    rank = sorted((v for v in s1.values() if v.get("done")),
                  key=lambda v: (-v["best"]["count_acc"], -v["best"]["dice"]))
    for v in rank:
        b = v["best"]
        A(f"  {v['model'].replace('.pt',''):16s} {v['params']/1e6:8.1f}M "
          f"{b['count_acc']:8.4f} {b['dice']:7.4f} {b['miou']:7.4f} "
          f"{b['inst_f1']:7.4f} {v['train_sec']:8.0f}")
    A("")
    A("  [단서] 위 수치는 fold0 val(12장)에서 후처리 conf 를 고른 뒤 같은 fold 에서")
    A("         측정한 값 = 선택 편향 포함 상한. 일반화 수치는 [4] 를 볼 것.")
    A("")

    A("[4] 5-fold 교차검증 (선택 편향 제거)")
    A("-" * 78)
    A("  nested = fold k 의 conf/Tw 를 k 를 제외한 4개 fold 에서만 선택 → fold k 완전 홀드아웃")
    A("  oracle = 각 fold 자기 자신에서 선택 (상한, 낙관 편향)")
    A("")
    A(f"  {'모델':16s} {'방식':8s} {'계수Acc':>16s} {'95% CI':>18s} {'Dice':>7s} {'instF1':>7s}")
    for k, v in cvr.items():
        for mode in ("fixed", "nested", "oracle"):
            r = v[mode]
            ci = f"[{r['count_acc_ci'][0]:.4f},{r['count_acc_ci'][1]:.4f}]"
            A(f"  {k:16s} {mode:8s} {r['count_acc']:.4f}+-{r['count_acc_sd']:.4f} "
              f"{ci:>18s} {r['dice']:7.4f} {r['inst_f1']:7.4f}")
    A("")

    A("[5] 97% 달성을 위한 에스케일레이션 (전부 nested = 완전 홀드아웃)")
    A("-" * 78)
    esc = [
        ("conf 격자 확장 (0.65->0.85)", "효과 없음 — 최적이 0.55~0.65 내부"),
        ("D4 TTA (직접 구현)", "역효과 — Dice 0.9487->0.9390. 모델이 이미 회전/반전 증강 학습"),
        ("면적 하한 필터", "역효과 — 진짜 세포를 제거"),
        ("watershed 재분리 (Tw)", "역효과 — 접촉 세포가 0건이라 과분할만 유발"),
        ("대형 단일 모델 (yolov8x)", "성공 — 단일로 0.9756 달성"),
        ("다중 모델 앙상블 합의 투표", "성공 — 최고 0.9768, 아래 표"),
    ]
    for a, b in esc:
        A(f"  - {a:28s} : {b}")
    A("")
    A("  [앙상블 방식] 각 모델이 독립적으로 타일 추론 -> 인스턴스 병합 -> 모델 간")
    A("  마스크 IoU 로 클러스터링 -> vote 개 이상 모델이 지지한 클러스터만 채택,")
    A("  마스크는 멤버 과반 픽셀 투표. vote 는 nested 로 선택(전 fold 일치).")
    A("")
    A(f"  {'방법':24s} {'계수Acc':>16s} {'95% CI':>18s} {'Dice':>7s} {'mIoU':>7s} {'instF1':>7s} {'MAE':>6s}")
    allrows = []
    for k, v in cvr.items():
        r = v["nested"]
        allrows.append((f"{k} (단일)", r["count_acc"], r["count_acc_sd"],
                        r["count_acc_ci"], r["dice"], r["miou"], r["inst_f1"],
                        r["count_mae"]))
    for k, v in s3b.items():
        r = v["nested"]
        allrows.append((k, r["count_acc"], r["sd"], r["ci"], r["dice"],
                        r["miou"], r["inst_f1"], r["count_mae"]))
    allrows.sort(key=lambda x: -x[1])
    for nm, a, sd, ci, dc, mi, f1, mae in allrows:
        mark = " <== 97% 달성" if a >= 0.97 else ""
        A(f"  {nm:24s} {a:.4f}+-{sd:.4f} "
          f"{'['+format(ci[0],'.4f')+','+format(ci[1],'.4f')+']':>18s} "
          f"{dc:7.4f} {mi:7.4f} {f1:7.4f} {mae:6.3f}{mark}")
    A("")

    A("[6] 참고 논문 대비 (Appl. Sci. 2026, 16, 357, CHO 데이터 Table 4/5)")
    A("-" * 78)
    A("  논문 CHO 결과 : Dice 최고 93.1 (UNet++), 계수정확도 0.91~0.96 (Tw 0.45~0.9)")
    A("                  최적 Tw = 0.65 (WBC 기준)")
    best = max(allrows, key=lambda x: x[1]) if allrows else None
    if best:
        A(f"  본 작업 최고  : 계수정확도 {best[1]:.4f}, Dice {best[4]:.4f}, "
          f"mIoU {best[5]:.4f}, 인스턴스 F1 {best[6]:.4f}")
        A(f"                  ({best[0]}, nested CV = 완전 홀드아웃)")
    A("  Dice 개선     : 93.1 -> 95.1 (+2.0 %p)")
    A("  계수 개선     : 0.96 -> 0.973 (+1.3 %p), 논문 미달 구간(0.91~0.96) 돌파")
    A("")
    A("  [논문 서술에 대한 함의]")
    A("  논문은 CHO 데이터에서 '계수 성능이 더 넓은 Tw 범위에서 안정적으로 유지된다'고")
    A("  적었는데, 실측 원인은 강건성이 아니라 **분리할 접촉 세포가 애초에 0건**이기")
    A("  때문이다. 본 작업에서 Tw 를 적용하면 전 모델/전 conf 에서 계수정확도가 떨어졌다.")
    A("  즉 이 데이터셋에서 watershed 단계는 불필요(또는 유해)하며, 논문의 Tw=0.65")
    A("  최적값은 WBC 데이터 고유 결과로 한정해 서술하는 것이 정확하다.")
    A("")

    A("[7] 검증 상태 — 미검증 항목 명시")
    A("-" * 78)
    A("  검증됨   : 60장 전수 5-fold 교차검증, 743개 세포 전부가 한 번씩 홀드아웃에 포함")
    A("             파일 단위 분할(세포 단위 누출 없음), 후처리 하이퍼파라미터 nested 선택")
    A("  미검증   : * 독립 배치/독립 장비로 촬영한 CHO 데이터에 대한 일반화")
    A("               (단일 배치 60장이므로 배치 간 전이는 측정 불가)")
    A("             * 엣지 가속기(DeepX M1 / Jetson / x86 CPU) 실기 지연시간·정확도")
    A("             * FP16/INT8 양자화 후 성능 (논문 대비 항목이나 본 작업 범위 외)")
    A("  주의     : fold 당 홀드아웃이 12장(약 150 세포)이라 세포 2~3개 오차가 1.5 %p 를")
    A("             움직인다. CI 는 fold 수준 부트스트랩(n=5)이므로 넓게 해석할 것.")
    A("")
    ag = figmeta.get("agreement")
    if ag:
        A("[8] 일치도 분석 (yolo11s-seg, 5-fold, 60장 전수)")
        A("-" * 78)
        A(f"  R^2            : {ag['r2']:.4f}")
        A(f"  편향(bias)     : {ag['bias']:+.3f} 세포/장 (예측이 약간 과다계수)")
        A(f"  일치 한계(LoA) : {ag['loa'][0]:+.3f} ~ {ag['loa'][1]:+.3f} 세포/장")
        A(f"  표본           : {ag['n']} 장")
        A("")

    A("[9] 오차 원인 분석")
    A("-" * 78)
    ea = load(f"{ROOT}/work/error_analysis.json", {})
    if ea:
        A(f"  FN(미검출) {len(ea['fn'])}건 / FP(오검출) {len(ea['fp'])}건 (총 743 세포 기준)")
        A(f"  FN 중 영상 경계 접촉 : {sum(1 for r in ea['fn'] if r['border'])}건")
        A("  FP 의 국소 텍스처 지표가 GT 세포와 사실상 동일(std 38.2 vs 38.6,")
        A("  Laplacian 398 vs 422) → FP 상당수가 **GT 주석 누락된 실제 세포**로 추정.")
        A("  즉 측정된 계수정확도는 모델 성능의 하한일 가능성이 있다.")
        A(f"  증거 크롭: {ROOT}/work/preview/montage_fp.png, montage_fn.png")
    A("")

    A("[10] 산출물 경로")
    A("-" * 78)
    A(f"  스크립트       : {ROOT}/work/*.py")
    A(f"  학습 가중치    : {ROOT}/work/runs/<model>_f<fold>/weights/best.pt")
    A(f"  그림(영문)     : {ROOT}/work/figures/en/")
    A(f"  그림(한글)     : {ROOT}/work/figures/ko/")
    A(f"  원시 결과 JSON : stage1_screen.json, stage2_cv.json, stage3b.json,")
    A(f"                   cv_report.json, error_analysis.json")
    A("")

    txt = "\n".join(out)
    p = f"{ROOT}/work/REPORT.txt"
    open(p, "w").write(txt)
    print(txt)
    print(f"\n[저장] {p}")


if __name__ == "__main__":
    main()
