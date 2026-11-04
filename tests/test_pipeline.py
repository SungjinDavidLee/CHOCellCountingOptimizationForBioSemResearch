"""파이프라인 불변식 검증. 데이터 유무에 따라 범위가 자동으로 좁혀진다.

    pytest -q tests/                 # 또는 pytest 없이
    python tests/test_pipeline.py

클론 직후(데이터 없음)에는 이식성·기하 테스트만 돌고, 결과 JSON 이나
dataset 이 있으면 정합성 테스트까지 돈다.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT_DIR / "code"
sys.path.insert(0, str(CODE))

import config  # noqa: E402

WORK = pathlib.Path(config.ROOT) / "work"
SENTINEL = "/tmp/sentinel-root"


class Skip(Exception):
    """전제 조건(데이터·산출물) 부재. pytest 에서도 skip 으로 변환된다."""


def need(name):
    """결과 JSON 을 work/ 또는 results/ 에서 읽는다. 없으면 skip."""
    for p in (WORK / name, ROOT_DIR / "results" / name):
        if p.exists():
            return json.load(open(p))
    raise Skip(f"{name} 없음")


def need_dir(path, label):
    if not pathlib.Path(path).is_dir():
        raise Skip(f"{label} 없음")
    return pathlib.Path(path)


def imread(path, flag=1):
    """cv2.imread 는 실패 시 None 을 돌려준다 — 즉시 잡는다 (flag 1=color, 0=grey)."""
    import cv2
    img = cv2.imread(str(path), flag)
    assert img is not None, f"읽기 실패: {path}"
    return img


def _probe(code, **env):
    """하위 프로세스에서 code/ 를 import 해 실행한다."""
    pre = f"import sys;sys.path.insert(0,{str(CODE)!r});"
    return subprocess.run([sys.executable, "-c", pre + code],
                          env=dict(os.environ, **env),
                          capture_output=True, text=True)


# ── 이식성 (데이터 불필요) ────────────────────────────────────────────────

def test_root_follows_env():
    """CHO_ROOT 로 ROOT 와 파생 경로를 모두 바꿀 수 있어야 한다."""
    code = "import config;print(config.ROOT, config.IMD, config.INST)"
    out = _probe(code, CHO_ROOT=SENTINEL).stdout.split()
    assert out and all(p.startswith(SENTINEL) for p in out), out

    env = {k: v for k, v in os.environ.items() if k != "CHO_ROOT"}
    dflt = subprocess.run(
        [sys.executable, "-c", f"import sys;sys.path.insert(0,{str(CODE)!r});"
         "import config;print(config.ROOT)"],
        env=env, capture_output=True, text=True).stdout.strip()
    assert dflt == "/home/user/CHO_CELL_Project", dflt


def test_repo_code_matches_workspace():
    """저장소 code/ 와 작업 디렉터리 work/ 의 사본이 갈라지지 않아야 한다.

    실제로 발생한 사고: 저장소 클론에서만 버그를 고치고 work/ 에 반영하지
    않았더니, 다음 턴에 work/ 에서 복사하면서 수정이 되돌아갔다(regression).
    같은 파일을 두 곳에 두는 구조 자체의 위험이므로 테스트로 막는다.
    """
    work = WORK
    if not work.is_dir():
        raise Skip("work/ 없음 (클론 전용 환경)")
    diverged = []
    for p in sorted(CODE.glob("*.py")):
        w = work / p.name
        if w.exists() and w.read_text() != p.read_text():
            diverged.append(p.name)
    assert not diverged, (
        "저장소와 work/ 사본이 다름 — 한쪽만 고치면 되돌아간다: "
        + ", ".join(diverged))


def test_no_hardcoded_paths():
    """ROOT 하드코딩은 config.py 기본값 한 곳만 허용 (클론 이식성)."""
    bad = [p.name for p in CODE.glob("*.py")
           if p.name != "config.py" and "/home/user/" in p.read_text()]
    assert not bad, bad


def test_all_modules_import():
    """데이터 없이도 전 모듈 import 가능해야 한다.

    모듈 최상위에서 파일을 읽으면 클론 직후 import 가 깨진다
    (실제로 inspect_errors.py 가 이 버그를 갖고 있었다).
    """
    broken = {p.name: r.stderr.strip().splitlines()[-1][:120]
              for p in sorted(CODE.glob("*.py"))
              if (r := _probe(f"import importlib;importlib.import_module({p.stem!r})",
                              CHO_ROOT=SENTINEL)).returncode}
    assert not broken, broken


def test_no_dead_or_undefined_names():
    """pyflakes 로 undefined name / 미사용 import 를 막는다.

    일괄 치환(sed·정규식)이 코드를 주석 뒤로 합치거나 import 를 중복
    삽입하는 사고가 실제로 있었다. import 테스트는 그 중 실행 경로에
    닿지 않는 함수 내부 오류를 놓치므로 정적 검사를 함께 둔다.
    """
    r = subprocess.run([sys.executable, "-m", "pyflakes",
                        *map(str, CODE.glob("*.py")),
                        *map(str, pathlib.Path(__file__).parent.glob("*.py"))],
                       capture_output=True, text=True)
    if "No module named" in r.stderr:
        raise Skip("pyflakes 미설치")
    # f-string placeholder 경고는 의도된 서식이므로 제외
    bad = [ln for ln in r.stdout.splitlines()
           if ln.strip() and "f-string is missing placeholders" not in ln]
    assert not bad, "\n" + "\n".join(bad)


def test_tiling_geometry():
    """타일 격자가 영상을 빠짐없이 덮고, 오버랩이 세포 직경 이상이어야 한다."""
    xs, ys = config.tile_origins(1376, 1032)
    assert (len(xs), len(ys)) == (3, 3), (xs, ys)
    assert xs[0] == ys[0] == 0
    assert xs[-1] + config.TILE == 1376 and ys[-1] + config.TILE == 1032
    assert config.TILE - config.STRIDE >= 53.8, "오버랩 < 세포 중간직경"


def test_folds_partition_files():
    """5-fold 가 겹치지 않고 전체를 덮어야 한다 (세포 단위 누출 금지)."""
    files = [f"cho{i:02d}.png" for i in range(1, 61)]
    flat = [f for fold in config.folds(files) for f in fold]
    assert len(config.folds(files)) == config.NFOLD
    assert sorted(flat) == files and len(flat) == len(set(flat))


# ── 결과 정합성 ──────────────────────────────────────────────────────────

def test_single_table_matches_raw():
    """성능표 수치가 원시 JSON 과 일치하고 등급 규칙을 지켜야 한다."""
    tbl, cvr, s1 = need("single_model_table.json"), need("cv_report.json"), \
        need("stage1_screen.json")
    assert len(tbl) == 10, len(tbl)
    for r in tbl:
        exp = (cvr[r["model"]]["nested"] if r["grade"] == "A"
               else s1[r["model"]]["best"])["count_acc"]
        assert abs(r["acc"] - exp) < 1e-9, (r["model"], r["acc"], exp)
        # CI 는 교차검증(A)에만 존재해야 한다
        assert (r["ci"] is not None) == (r["grade"] == "A"), r["model"]


def test_nested_cv_not_optimistic():
    """nested 는 oracle 을 넘을 수 없다 (낙관 편향 역전 탐지)."""
    for k, v in need("cv_report.json").items():
        assert v["nested"]["count_acc"] <= v["oracle"]["count_acc"] + 1e-9, k
        assert len(v["nested"]["per_fold"]) == config.NFOLD, k


def test_latency_consistency():
    """FP16 이 FP32 보다 느리면 측정 오류다."""
    need_keys = {"gpu_ms", "gpu_fp16_ms", "cpu_ms", "params", "ckpt_mb", "tiles"}
    for k, v in need("latency.json").items():
        assert need_keys <= set(v), (k, need_keys - set(v))
        assert v["tiles"] == 9, k
        assert v["gpu_fp16_ms"] <= v["gpu_ms"], k


def test_gt_fill_matches_published():
    """GT 윤곽 채움 결과가 공개 수치와 일치해야 한다.

    GT 는 1px 외곽선이므로 채우지 않으면 직경이 16.0px 로 오산된다.
    """
    import numpy as np
    npys = sorted(need_dir(config.INST, "instances_filled").glob("*.npy"))
    assert len(npys) == 60, len(npys)
    areas, cov = [], []
    for p in npys:
        a = np.load(p).astype(np.int32)
        cov.append((a > 0).mean())
        areas += [s for v in range(1, int(a.max()) + 1)
                  if (s := int((a == v).sum()))]
    diam = float(np.median(2 * (np.array(areas) / np.pi) ** 0.5))
    assert len(areas) == 743, len(areas)
    assert abs(diam - 53.8) < 0.1, diam
    assert abs(float(np.mean(cov)) * 100 - 2.14) < 0.02, float(np.mean(cov)) * 100


def test_gt_is_outline_not_fill():
    """GT 가 외곽선임을 ring/filled 면적비로 재확인 (전처리 전제)."""
    import cv2
    import numpy as np
    gtd = need_dir(config.GTD, "dataset")
    ring = (imread(sorted(gtd.glob("*.png"))[0], 0) > 127).astype(np.uint8)
    cs, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(ring)
    cv2.drawContours(filled, cs, -1, 1, cv2.FILLED)
    assert ring.sum() / filled.sum() < 0.35, "GT 가 이미 채워져 있다 — 전처리 재검토"


def test_calib_set_integrity():
    """INT8 calibration 세트가 배포 추론 입력과 동일 규격이어야 한다."""
    d = need_dir(WORK / "calib_tiles", "calib 세트")
    m = json.load(open(d / "manifest.json"))
    pngs = list(d.glob("*.png"))
    assert m["n_tiles"] == len(pngs), (m["n_tiles"], len(pngs))
    assert {r["file"] for r in m["tiles"]} == {p.name for p in pngs}
    assert m["n_with_cells"] + m["n_empty"] == m["n_tiles"]
    assert m["n_empty"] / m["n_tiles"] < 0.20, "빈 타일 과다 → activation 왜곡"
    shape = imread(pngs[0]).shape
    assert shape == (config.TILE, config.TILE, 3), shape
    assert m["total_cells"] > 743, "오버랩 중복이 반영되지 않았다"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    fails = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as e:
            print(f"SKIP  {name}  [{e}]")
        except Exception as e:  # noqa: BLE001
            fails.append(name)
            print(f"FAIL  {name}  [{type(e).__name__}: {e}]")
    print(f"\n{len(tests) - len(fails)} pass / {len(fails)} fail")
    sys.exit(1 if fails else 0)
