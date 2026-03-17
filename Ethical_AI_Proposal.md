# Explainable AI for Healthcare Diagnostics: Improving Human Oversight in Chest X-Ray Classification

**Authors:** Vipul Bodhani, Ademola John (Purdue University Fort Wayne)  
**Contact:** {bodhvm01, ademjo01}@purdue.edu

## Abstract
Deep CNNs for chest X-ray diagnosis achieve strong accuracy but are opaque and can be overconfident. We compare a CheXpert-pretrained DenseNet with post-hoc explanations (Grad-CAM, Integrated Gradients) against an interpretable prototype model, and add a predict-or-defer policy that routes low-confidence or poorly explained cases to humans. Using CheXlocalize for pixel-level evaluation, we measure diagnostic performance, calibration, explanation alignment, and oversight effectiveness.

## 1. Introduction
Black-box models limit clinician trust and can cause harm when confidently wrong. Ethical deployment requires transparency, calibration, and human oversight. We test whether explanation quality and abstention improve safety for chest X-ray classification.

## 2. Motivation
Key risks: non-transparent reasoning, overconfident errors, and misleading saliency maps. Goal: quantify when explanations help or fail and integrate that signal into deferral.

## 3. Ethical Question
Can explainability plus calibrated abstention improve human oversight and safety compared with black-box predictions alone?

## 4. Datasets
- **CheXpert (train/val/test):** multi-label CXR benchmark with uncertainty labels.  
- **CheXlocalize (val/test):** pixel-level radiologist segmentations + baseline Grad-CAM maps/segmentations for localization metrics.  
- *(Optional)* **MIMIC-CXR-JPG:** external shift check.

## 5. Methods
### 5.1 Black-box baseline
CheXpert-pretrained DenseNet121 (TorchXRayVision). Post-hoc: Grad-CAM, Integrated Gradients.

### 5.2 Interpretable model
Prototype-based network to point to similar learned visual prototypes.

### 5.3 Human oversight (predict-or-defer)
- Uncertainty gate: calibrated confidence threshold.  
- Explanation gate: localization quality threshold (mIoU/hit).  
- Defer to clinician when either gate fails; evaluate risk–coverage and AURC.

## 6. Evaluation Metrics
- **Diagnostic:** AUROC, AUPRC, F1, sensitivity/specificity.  
- **Calibration:** ECE, Brier; before/after temperature scaling.  
- **Explanation quality:** mIoU, hit rate (CheXlocalize); sanity checks (randomization), faithfulness (deletion/insertion AUC on a subset).  
- **Oversight:** risk–coverage curves, AURC, error reduction under deferral.

## 7. Planned Experiments / Timeline (16 weeks)
- **Weeks 1–2 (done/ongoing):** set up env, download CheXlocalize/CheXpert; reproduce baseline localization with provided Grad-CAM segmentations; run XRV DenseNet val/test AUROC/AUPRC/F1.  
- **Weeks 3–5:** add temperature scaling; generate own Grad-CAM + IG maps; convert to segmentations with official scripts; compare vs baseline.  
- **Weeks 6–8:** implement predict-or-defer (uncertainty vs explanation vs combined); plot risk–coverage/AURC.  
- **Weeks 9–12:** prototype-based model; compare to post-hoc explanations.  
- **Weeks 13–14:** faithfulness + sanity checks; small deletion/insertion study.  
- **Weeks 15–16:** external shift (optional MIMIC-CXR), ablation summary, final report.

## 8. Progress Update 1 (Week 10)
`instructions.md` asks for proof the project is running. Current status:
- [x] Data downloaded: CheXlocalize (with baseline Grad-CAM/segmentations) + CheXpert val/test (`datasets/chexlocalize`).
- [x] Baseline localization (CheXlocalize official eval on provided Grad-CAM maps, val set):
  - mIoU (mean) per task (95% CI):
    - Cardiomegaly 0.455 (0.420–0.491)
    - Enlarged Cardiomediastinum 0.409 (0.379–0.437)
    - Consolidation 0.384 (0.326–0.440)
    - Airspace Opacity 0.231 (0.201–0.262)
    - Edema 0.327 (0.288–0.362)
    - Atelectasis 0.197 (0.166–0.228)
    - Pleural Effusion 0.159 (0.125–0.195)
    - Pneumothorax 0.221 (0.109–0.337)
    - Support Devices 0.190 (0.173–0.209)
    - Lung Lesion 0.136 (0.136–na)
  - Artifacts: `artifacts/baseline_seg_val.json`, `artifacts/baseline_eval_val/iou_summary_results.csv`.
- [x] Baseline classification (TorchXRayVision DenseNet121, MPS):
  - Val AUROC/AUPRC/F1 saved to `artifacts/chexpert_val_metrics.csv` (table printed during run).
  - Test AUROC/AUPRC/F1 saved to `artifacts/chexpert_test_metrics.csv`.
- [ ] Calibration: temperature scaling and ECE/Brier deltas.
- [ ] Own explanations: generate Grad-CAM + IG maps, convert to segmentations, compare vs baseline.

## 9. Next steps (Weeks 11–12)
1) Add temperature scaling on val; report ECE/Brier before/after.  
2) Generate Grad-CAM + IG maps with current model; evaluate mIoU/hit vs baseline.  
3) Build predict-or-defer thresholds (uncertainty + explanation); plot risk–coverage and AURC.  
4) Summarize baseline tables in Update 1 write-up with links to artifacts.

## 10. Ethical Statement
Uses de-identified public datasets; goal is to enhance transparency and human oversight. Results will be reported with limitations and without clinical claims.
