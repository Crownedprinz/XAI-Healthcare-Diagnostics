# Week-10 Project Plan

## Purpose
Give you a concrete, low-lift path to produce the “current results” and set up the next experiments, while clarifying what I can and cannot run in this environment.

## What you need to do locally
- Download data: CheXlocalize (includes CheXpert val/test, labels, baseline Grad-CAM/segmentations). Optional: full CheXpert train set; optional: MIMIC-CXR for shift tests.
- Set up env: create a Python 3.10 conda/env; install PyTorch with your CUDA version, plus `torchxrayvision`, `captum`, `pytorch-grad-cam`, `scikit-learn`, `pycocotools`, `opencv-python`, `tqdm`, `pillow`, `numpy`, `pandas`, `matplotlib`.
- Run baseline eval: in the CheXlocalize repo, execute `heatmap_to_segmentation.py` and `eval.py` on the provided baseline maps to get mIoU/hit-rate tables (this is your Week-10 “current results”).
- Run model inference: use the XRV CheXpert DenseNet (`densenet121-res224-chex`) on CheXlocalize val/test to produce AUROC/AUPRC/F1 + calibration metrics.
- Save artifacts: keep the `*_summary_results.csv` from eval, your model metric CSV, and a short log of commands/commit hash.

## What I can do for you here
- Write ready-to-run scripts:
  - `run_pred_baseline.py` to load XRV model and score val/test.
  - `generate_gradcam_pkls.py` (and IG variant) to emit CheXlocalize-format map pickles.
  - A small calibration + temperature-scaling helper; risk–coverage evaluation for predict-or-defer.
- Adapt dataset loaders once you show your CheXlocalize folder layout.
- Draft the Week-10 update text using your outputs (tables/plots).
- Review/modify evaluation commands and figure out any errors from your logs.

## What I cannot do in this environment
- Download CheXlocalize/CheXpert/MIMIC data (requires your credentials and network access).
- Execute GPU-heavy training/inference; large downloads are blocked.
- Access expired uploads or your private portals.

## What to send me next
- OS + GPU info, whether you use conda.
- `tree -L 2` (or similar) of your CheXlocalize directory.
- The CSV outputs from `eval.py` and any model metrics you compute.
- Any error logs if commands fail.

Note that the dataset download link cannot be used directly in a browser
How do I use a download link for an entire dataset?
A download link for an entire dataset provides the location of the dataset in Azure as well as a special time-limited key that allows you to download the entire dataset. This link can be used with tools that can copy files from Azure, like the following:

AzCopy - a command-line tool for Windows or Linux that copies files to and from Azure.
Azure Storage Explorer - a utility that is used to manage Azure storage.

https://aimistanforddatasets01.blob.core.windows.net/chexlocalize?sv=2019-02-02&sr=c&sig=CQLBxruUoZFW2fE3fqQzTkSnj%2FQ26R8TqsVmvFbE9A4%3D&st=2026-03-15T05%3A52%3A47Z&se=2026-04-14T05%3A57%3A47Z&sp=rl
