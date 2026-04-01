#!/usr/bin/env bash
# Project Update 2 — run after CheXpert images + CheXlocalize files are under datasets/chexlocalize/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; fi

mkdir -p artifacts

echo "== 1) Baseline classification (val + test) =="
$PY scripts/run_pred_baseline.py \
  --csv datasets/chexlocalize/CheXpert/val_labels.csv \
  --images-root datasets/chexlocalize/CheXpert \
  --split-name val \
  --batch-size 32 --num-workers 4

$PY scripts/run_pred_baseline.py \
  --csv datasets/chexlocalize/CheXpert/test_labels.csv \
  --images-root datasets/chexlocalize/CheXpert \
  --split-name test \
  --batch-size 32 --num-workers 4

echo "== 2) Per-class F1 threshold tuning (val -> test) =="
$PY scripts/tune_f1_thresholds.py \
  --val-preds artifacts/chexpert_val_preds.npz \
  --test-preds artifacts/chexpert_test_preds.npz \
  --out-dir artifacts

echo "== 3) Temperature scaling / calibration =="
$PY scripts/temperature_scaling.py \
  --preds artifacts/chexpert_val_preds.npz \
  --out artifacts/chexpert_val_calibration.csv

echo "== 4) Risk–coverage curve + defer table =="
$PY scripts/risk_coverage.py \
  --preds artifacts/chexpert_val_preds.npz \
  --out artifacts/risk_coverage_val.csv

$PY scripts/defer_policy_table.py \
  --preds artifacts/chexpert_val_preds.npz \
  --out artifacts/defer_policy_val.csv

echo "== 5) Grad-CAM maps (val + test) =="
$PY scripts/generate_gradcam_pkls.py \
  --csv datasets/chexlocalize/CheXpert/val_labels.csv \
  --images-root datasets/chexlocalize/CheXpert \
  --out-dir artifacts/gradcam_maps_val \
  --split-name val --batch-size 8

$PY scripts/generate_gradcam_pkls.py \
  --csv datasets/chexlocalize/CheXpert/test_labels.csv \
  --images-root datasets/chexlocalize/CheXpert \
  --out-dir artifacts/gradcam_maps_test \
  --split-name test --batch-size 8

echo "== 6) Heatmaps -> segmentations + IoU eval (Grad-CAM) =="
$PY externals/chexlocalize/heatmap_to_segmentation.py \
  --map_dir artifacts/gradcam_maps_val \
  --output_path artifacts/my_gradcam_seg_val.json

$PY scripts/pad_pred_segmentation_json.py \
  --gt-path datasets/chexlocalize/CheXlocalize/gt_segmentations_val.json \
  --pred-path artifacts/my_gradcam_seg_val.json

$PY externals/chexlocalize/heatmap_to_segmentation.py \
  --map_dir artifacts/gradcam_maps_test \
  --output_path artifacts/my_gradcam_seg_test.json

$PY scripts/pad_pred_segmentation_json.py \
  --gt-path datasets/chexlocalize/CheXlocalize/gt_segmentations_test.json \
  --pred-path artifacts/my_gradcam_seg_test.json

$PY externals/chexlocalize/eval.py --metric iou \
  --gt_path datasets/chexlocalize/CheXlocalize/gt_segmentations_val.json \
  --pred_path artifacts/my_gradcam_seg_val.json \
  --save_dir artifacts/my_gradcam_eval_val \
  --true_pos_only True --if_human_benchmark False

$PY externals/chexlocalize/eval.py --metric iou \
  --gt_path datasets/chexlocalize/CheXlocalize/gt_segmentations_test.json \
  --pred_path artifacts/my_gradcam_seg_test.json \
  --save_dir artifacts/my_gradcam_eval_test \
  --true_pos_only True --if_human_benchmark False

echo "== 7) Explanation threshold sweep (validation) =="
$PY scripts/sweep_explanation_threshold.py \
  --map-dir artifacts/gradcam_maps_val \
  --gt-path datasets/chexlocalize/CheXlocalize/gt_segmentations_val.json \
  --out-dir artifacts/threshold_sweep_gradcam

echo "== 8) Integrated Gradients (slow; same downstream as Grad-CAM) =="
$PY scripts/generate_integrated_gradients.py \
  --csv datasets/chexlocalize/CheXpert/val_labels.csv \
  --images-root datasets/chexlocalize/CheXpert \
  --out-dir artifacts/ig_maps_val \
  --split-name val --batch-size 1 --ig-steps 32

echo "Done. See artifacts/ for CSV/JSON outputs."
