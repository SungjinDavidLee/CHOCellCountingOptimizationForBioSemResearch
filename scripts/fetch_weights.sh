#!/usr/bin/env bash
# GitHub Releases 에서 학습 가중치를 내려받는다.
set -euo pipefail
REPO="SungjinDavidLee/CHOCellCountingOptimizationForBioSemResearch"
TAG="${1:-weights-v1}"
DEST="${2:-weights}"
mkdir -p "$DEST"
echo "Downloading $TAG -> $DEST"
gh release download "$TAG" --repo "$REPO" --dir "$DEST" --pattern '*.pt' --clobber
ls -lh "$DEST"
