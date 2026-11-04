"""예시 이미지를 저장소 업로드용으로 최적화한다.

원본 54 MB 는 git 본문에 과하다. GitHub README 에서 보는 용도이므로
가로 폭을 줄이고 **JPEG 품질 95** 로 변환한다.
윤곽선 시각화에는 품질 95 가 충분하며(색상 윤곽 보존 확인), PNG 대비 ~1/5.
격자·파이프라인은 글자 가독성이 중요해 폭을 더 크게 유지한다.
"""
import os
import pathlib
import shutil
import sys
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import ROOT

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else f"{ROOT}/work/figures/examples")
DST = pathlib.Path(sys.argv[2] if len(sys.argv) > 2
                   else os.environ.get("EXAMPLES_OUT", "/tmp/examples_opt"))
# grid 는 5열 x 2행이라 셀당 폭이 좁다 → 넉넉히 둬야 윤곽이 보인다
MAXW = {"grid": 2400, "pipeline": 1400, "errors": 1300, "default": 1100}
QUALITY = 95


def target_width(name):
    for k, v in MAXW.items():
        if name.startswith(k):
            return v
    return MAXW["default"]


def main():
    shutil.rmtree(DST, ignore_errors=True)
    tot_in = tot_out = 0
    for p in sorted(SRC.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(SRC)
        if p.suffix.lower() != ".png":
            out = DST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, out)
            continue
        out = (DST / rel).with_suffix(".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(p))
        assert img is not None, p
        tw = target_width(rel.name)
        if img.shape[1] > tw:
            sc = tw / img.shape[1]
            img = cv2.resize(img, None, fx=sc, fy=sc,
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        a, b = p.stat().st_size, out.stat().st_size
        tot_in += a
        tot_out += b
        print(f"  {out.relative_to(DST)}  {a/1e6:.1f}MB -> {b/1e6:.2f}MB "
              f"({img.shape[1]}x{img.shape[0]})")
    print(f"\n합계 {tot_in/1e6:.1f}MB -> {tot_out/1e6:.1f}MB "
          f"({tot_out/tot_in*100:.0f}%)")


if __name__ == "__main__":
    main()
