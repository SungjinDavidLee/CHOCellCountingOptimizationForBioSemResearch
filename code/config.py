"""CHO YOLO-seg 파이프라인 공통 설정 (학습/추론 간 불일치 방지).

세포 등가직경 실측 53.8 px (채운 GT 기준) → 업스케일 불필요.
1376x1032 를 512 타일/스트라이드 432 로 자르면 3x3=9 타일, 오버랩 80 px.
세포 최대 직경 167 px 이므로 오버랩만으로는 경계 세포가 한 타일에
온전히 담기지 않을 수 있어, 추론 시 병합 IoU 로 중복을 제거한다.
"""
import os

# 작업 루트. 재현 시 CHO_ROOT 환경변수로 덮어쓸 수 있다.
#   export CHO_ROOT=/path/to/your/workspace
# 기대 구조:
#   $CHO_ROOT/dataset/images/images/*.png
#   $CHO_ROOT/dataset/ground_truth/ground_truth/*.png
#   $CHO_ROOT/work/   <- 산출물이 쌓이는 곳
ROOT = os.environ.get("CHO_ROOT", "/home/user/CHO_CELL_Project")
IMD = f"{ROOT}/dataset/images/images"
GTD = f"{ROOT}/dataset/ground_truth/ground_truth"
INST = f"{ROOT}/work/instances_filled"

TILE = 512
STRIDE = 432
UPSCALE = 1
IMGSZ = TILE * UPSCALE

NFOLD = 5
SEED = 0

# 타일 경계에서 잘린 인스턴스: 가시 비율이 이보다 작으면 라벨에서 제외
MIN_VISIBLE_FRAC = 0.55
MIN_INST_AREA = 150
EMPTY_KEEP = 0.2


def tile_origins(w, h):
    def ax(n):
        v, x = [], 0
        while True:
            v.append(min(x, n - TILE))
            if x + TILE >= n:
                break
            x += STRIDE
        return sorted(set(v))
    return ax(w), ax(h)


def folds(files):
    """파일 단위 K-fold (세포 단위 분할은 누출이므로 금지)."""
    return [files[i::NFOLD] for i in range(NFOLD)]
