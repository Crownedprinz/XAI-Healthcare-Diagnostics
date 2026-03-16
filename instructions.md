
# Next Steps for Your Week‑10 Project Update and Experiment Roadmap

## What you should do next

Your screenshot (“Project Update 1 (Due Week 10)”) is essentially asking for  **evidence that your project is already running** , plus a **credible plan** for the next phase. The fastest way to produce real results (without waiting weeks of training) is to  **reproduce baseline evaluation first** , then plug in your own refined models.

A strong Week‑10 update can be built around these concrete deliverables:

* **Current results (by end of this week):**
  * Run the **official CheXlocalize evaluation code** on the **provided baseline Grad‑CAM maps/segmentations** included with the CheXlocalize dataset package (this quickly yields mIoU/hit‑rate summaries). The official repo is designed for exactly this workflow.
  * Run a **TorchXRayVision (XRV) CheXpert pretrained DenseNet** on CheXpert val/test from the CheXlocalize download and produce AUROC/AUPRC/F1 + calibration metrics (ECE/Brier). XRV provides ready‑to‑use CheXpert‑trained weights (e.g., `densenet121-res224-chex`).
* **Upcoming results (next 1–2 weeks):**
  * Add **temperature scaling** (calibration) and show “before vs after” ECE/Brier.
  * Generate your  **own Grad‑CAM + Integrated Gradients heatmaps** , convert them to segmentations with the CheXlocalize scripts, and compare to the provided baseline.
  * Implement a **predict‑or‑defer** policy: defer when (a) calibrated uncertainty is high or (b) explanation quality is low; evaluate via **risk–coverage curves** and  **AURC** .
* **Problems/risks to report (it’s good to have them):**
  * Explanation reliability is known to be imperfect in CXR; your job is to quantify and mitigate (via deferral), not to assume heatmaps “solve trust.”
  * Human‑AI interaction is heterogeneous; showing deferral is ethically relevant because help can sometimes hurt.

If you do only the “current results” items above, you will already have enough for a credible Week‑10 update.

## Papers you should use and why

These are strong, widely‑cited, and directly usable for your narrative and experiments:

| Paper / resource                                                                      | Why it belongs in your project                                                                | How it affects your experiments                                                  |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **CheXpert** (Irvin et al., AAAI 2019)                                          | Defines the dataset + uncertainty labels; establishes CXR diagnostic benchmark                | Justifies label handling/calibration; gives you a canonical target list          |
| **CheXlocalize / saliency benchmarking** (Saporta et al., Nat Mach Intell 2022) | Shows saliency methods can underperform humans; provides evaluation protocol (mIoU, hit rate) | You’ll compute the same metrics; anchors your “explanation quality gate”      |
| **Grad‑CAM** (Selvaraju et al., ICCV 2017)                                     | Standard post‑hoc visual explanation for CNNs                                                | Your baseline explanation method; output heatmaps evaluated by CheXlocalize      |
| **Integrated Gradients** (Sundararajan et al., ICML 2017)                       | Axiomatic attribution; common alternative to CAM                                              | Second explanation method; compare localization + faithfulness                   |
| **Sanity Checks for Saliency Maps** (Adebayo et al., NeurIPS 2018)              | Warns that some saliency maps can be visually plausible but not model‑dependent              | Add a sanity‑check ablation (randomize weights → compare saliency similarity)  |
| **RISE + deletion/insertion** (Petsiuk et al., BMVC 2018)                       | Provides automated “faithfulness” metrics (deletion/insertion AUC)                          | Use as explanation faithfulness scoring (in addition to localization)            |
| **Learning to Defer** (Mozannar & Sontag, ICML 2020)                            | Formalizes predict‑or‑defer systems                                                         | Justifies your deferral training/thresholding design                             |
| **Selective classification** (Geifman & El‑Yaniv, NeurIPS 2017)                | Risk–coverage framing for abstention                                                         | Provides your key oversight evaluation plots                                     |
| **Radiologist‑AI assistance heterogeneity** (Yu et al., Nat Med 2024)          | Shows AI assistance can help some radiologists and harm others                                | Motivates human oversight + careful presentation of uncertainty and explanations |
| **TorchXRayVision** (Cohen et al., MIDL 2022 + docs)                            | Reproducible CXR datasets/models + pretrained weights                                         | Lets you run strong baselines quickly; reduces engineering overhead              |

This set is “published and good,” and it directly maps to your planned experiments.

## Datasets to use and what to download

You already picked the right real‑world datasets:

| Dataset                              | What you’ll use it for                                                        | Size / access                                      | Privacy/terms                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------ | -------------------------------------------------- | ---------------------------------------------------------------- |
| **CheXpert**                   | Training/finetuning (optional early); main pathology classification benchmark  | 224,316 CXRs / 65,240 pts; Stanford AIMI portal    | Research terms; do not redistribute                              |
| **CheXlocalize**               | Explanation evaluation using pixel‑level radiologist segmentations and points | 234 val (200 pts), 668 test (500 pts); AIMI portal | De‑identified research release; use for evaluation; keep secure |
| **MIMIC‑CXR‑JPG** (optional) | External validation / distribution shift                                       | 377,110 JPG images; PhysioNet credentialed access  | HIPAA Safe Harbor de‑identified; must follow PhysioNet DUA      |

Critical shortcut for Week‑10: the **cheXlocalize GitHub repo** documents that the CheXlocalize download includes  **CheXpert val/test images + labels and baseline Grad‑CAM heatmaps/segmentations** , and the repo ships scripts to evaluate them. That means you can generate real localization metrics immediately.

If full CheXpert is heavy for your machine right now, AIMI also provides a **CheXpert demo data** download you can use to sanity‑check your pipeline first.

## How to run experiments that produce real results

A practical, week‑by‑week experiment sequence that matches your ethical goal (oversight) is:

**Phase A: Baseline reproduction (fast)**

* Download CheXlocalize.
* Run CheXlocalize’s evaluation scripts on the included baseline segmentations/heatmaps to produce your first mIoU/hit‑rate tables.
* Run XRV pretrained CheXpert DenseNet on CheXpert val/test (from CheXlocalize package) and compute AUROC/AUPRC/F1.

This gives you “current results” for the update.

**Phase B: Refine existing model outputs**

* Add **temperature scaling** on the CheXlocalize validation set; report ECE/Brier drop.
* Generate your own Grad‑CAM and IG maps for the same images.
* Convert maps → segmentations and evaluate using the official scripts.

**Phase C: Human oversight via predict‑or‑defer**

* Define a deferral policy:
  * Defer if **uncertainty** above threshold (after calibration).
  * Defer if **explanation quality** below threshold (mIoU/hit on validation).
* Evaluate:
  * Risk–coverage curves (uncertainty only vs explanation only vs combined).
  * AURC and “system error” reduction (assume deferred cases are handled correctly by a human/ground truth oracle—standard evaluation for abstention).

**Phase D: Explanation reliability checks**

* Run **sanity checks** (randomize weights / labels → saliency similarity) to avoid “pretty heatmaps” failure modes.
* Run **deletion/insertion AUC** on a small subset (e.g., 100 images) to measure faithfulness.

This is enough for a strong final project and directly addresses ethical oversight.

## Code starter kit you can run locally

I cannot download and run CheXpert/CheXlocalize in this chat environment (dataset size + licensing), so I can’t truthfully generate results here. What I *can* do is give you **copy‑paste‑ready code** that will produce results on your machine. If you paste the outputs (tables/CSVs/logs) back to me, I’ll help you interpret them and write your Week‑10 update narrative.

### Environment setup

<pre node="[object Object]" class="my-4 overflow-auto whitespace-pre-wrap"><div class="bg-token-main-surface-secondary group relative overflow-hidden rounded-2xl _codeBlock_zqek1_1"><div class="text-token-text-secondary flex items-center justify-between px-4 py-2 text-xs font-normal tracking-wide"><span>bash</span><div class="flex items-center gap-2"><button type="button" class="text-token-text-secondary hover:text-token-text-primary focus-visible:outline-token-border-strong rounded-full px-3 py-1 text-[11px] font-medium opacity-0 transition group-hover:opacity-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2" aria-live="polite">Copy</button></div></div><pre class="m-0 overflow-x-auto p-4 whitespace-pre _codeContent_zqek1_76"><code class="block text-[10pt] leading-relaxed whitespace-pre language-bash"># create env
conda create -n cxr_xai python=3.10 -y
conda activate cxr_xai

# core libs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# xrv + explainability + eval
pip install torchxrayvision captum grad-cam interpret shap
pip install numpy pandas scikit-learn matplotlib tqdm pillow opencv-python pycocotools
</code></pre></div></pre>

Key tool docs:

* TorchXRayVision models/datasets and CheXpert weights
* Captum Integrated Gradients
* pytorch-grad-cam package
* InterpretML EBM docs
* SHAP docs
* Opacus (DP‑SGD) and TensorFlow Privacy (optional)

### Script 1: Load XRV CheXpert pretrained model and compute predictions

<pre node="[object Object]" class="my-4 overflow-auto whitespace-pre-wrap"><div class="bg-token-main-surface-secondary group relative overflow-hidden rounded-2xl _codeBlock_zqek1_1"><div class="text-token-text-secondary flex items-center justify-between px-4 py-2 text-xs font-normal tracking-wide"><span>python</span><div class="flex items-center gap-2"><button type="button" class="text-token-text-secondary hover:text-token-text-primary focus-visible:outline-token-border-strong rounded-full px-3 py-1 text-[11px] font-medium opacity-0 transition group-hover:opacity-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2" aria-live="polite">Copy</button></div></div><pre class="m-0 overflow-x-auto p-4 whitespace-pre _codeContent_zqek1_76"><code class="block text-[10pt] leading-relaxed whitespace-pre language-python"># file: run_pred_baseline.py
import torch
import torchxrayvision as xrv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pick the pretrained CheXpert weights (224x224 DenseNet)
model = xrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=False)
model = model.to(DEVICE).eval()

# Your dataset loader here:
# Option A (recommended): use CheXlocalize download structure and their CSV labels.
# You will define a PyTorch Dataset that returns:
#   img: torch.FloatTensor [1, H, W] scaled like XRV expects
#   y:   float labels for your target tasks

def sigmoid(x):
    return 1/(1+np.exp(-x))

def evaluate(y_true, y_logit):
    y_prob = sigmoid(y_logit)
    aurocs, auprcs = {}, {}
    for j in range(y_true.shape[1]):
        aurocs[j] = roc_auc_score(y_true[:, j], y_prob[:, j])
        auprcs[j] = average_precision_score(y_true[:, j], y_prob[:, j])
    return aurocs, auprcs

# TODO: implement dataset + dataloader, then run:
# y_true_all, y_logit_all = [], []
# for batch in tqdm(loader):
#     img = batch["img"].to(DEVICE)  # [B, 1, H, W]
#     y   = batch["y"].cpu().numpy()
#     with torch.no_grad():
#         logits = model(img).cpu().numpy()
#     y_true_all.append(y)
#     y_logit_all.append(logits)
# y_true_all = np.concatenate(y_true_all)
# y_logit_all = np.concatenate(y_logit_all)
# print(evaluate(y_true_all, y_logit_all))
</code></pre></div></pre>

This script is intentionally minimal because the biggest variance is  **your local file paths** . Once you tell me your CheXlocalize directory layout, I can give you the exact dataset loader that matches it.

### Script 2: Generate Grad‑CAM maps in CheXlocalize’s expected pickle format

This produces `*_map.pkl` files exactly like CheXlocalize expects for `heatmap_to_segmentation.py` and `eval.py`. The required pickle keys are visible in the official scripts.

<pre node="[object Object]" class="my-4 overflow-auto whitespace-pre-wrap"><div class="bg-token-main-surface-secondary group relative overflow-hidden rounded-2xl _codeBlock_zqek1_1"><div class="text-token-text-secondary flex items-center justify-between px-4 py-2 text-xs font-normal tracking-wide"><span>python</span><div class="flex items-center gap-2"><button type="button" class="text-token-text-secondary hover:text-token-text-primary focus-visible:outline-token-border-strong rounded-full px-3 py-1 text-[11px] font-medium opacity-0 transition group-hover:opacity-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2" aria-live="polite">Copy</button></div></div><pre class="m-0 overflow-x-auto p-4 whitespace-pre _codeContent_zqek1_76"><code class="block text-[10pt] leading-relaxed whitespace-pre language-python"># file: generate_gradcam_pkls.py
import os, pickle
import torch
import numpy as np
import torchxrayvision as xrv
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# XRV DenseNet + CheXpert weights
model = xrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=False).to(DEVICE).eval()

# IMPORTANT: For XRV DenseNet, many examples use model.features[-1] as CAM target layer.
# (This matches common practice in CXR tutorials using XRV.)
target_layers = [model.features[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

# CheXlocalize tasks (their code uses "Airspace Opacity" naming) citeturn12view0
LOCALIZATION_TASKS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Airspace Opacity",
    "Edema", "Consolidation", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Support Devices"
]

# Map CheXlocalize "Airspace Opacity" to XRV "Lung Opacity"
TASK_TO_XRV = {t: t for t in LOCALIZATION_TASKS}
TASK_TO_XRV["Airspace Opacity"] = "Lung Opacity"

# XRV model output targets list (documented) citeturn15view0
xrv_targets = list(model.targets)

def save_pkl(out_dir, img_id, task, heatmap_hw, cxr_dims_wh, prob_scalar):
    os.makedirs(out_dir, exist_ok=True)
    # CheXlocalize filename convention: {img_id}_{task}_map.pkl citeturn11view0
    pkl_path = os.path.join(out_dir, f"{img_id}_{task}_map.pkl")
    info = {
        "map": torch.tensor(heatmap_hw, dtype=torch.float32)[None, None, :, :],
        "cxr_dims": cxr_dims_wh,     # (W, H)
        "prob": float(prob_scalar),  # scalar is acceptable citeturn10view0
        "task": task
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(info, f)

# TODO: implement your dataloader that yields:
#   img_id: string (e.g., patientXXX_studyY_viewZ_frontal)
#   img: torch tensor [1, H, W] (XRV normalized), and original dims (W,H) if you resize
#
# for sample in tqdm(dataset):
#     img = sample["img"].to(DEVICE)[None, ...]   # [1,1,H,W]
#     img_id = sample["img_id"]
#     W,H = sample["orig_wh"]                     # original
#     with torch.no_grad():
#         logits = model(img)                     # [1, num_targets]
#         probs  = torch.sigmoid(logits)[0].cpu().numpy()
#
#     for task in LOCALIZATION_TASKS:
#         xrv_name = TASK_TO_XRV[task]
#         class_idx = xrv_targets.index(xrv_name)
#         heatmap = cam(input_tensor=img, targets=[ClassifierOutputTarget(class_idx)])[0]
#         prob_scalar = probs[class_idx]
#         save_pkl("maps_gradcam", img_id, task, heatmap, (W,H), prob_scalar)
</code></pre></div></pre>

### Script 3: Use the official CheXlocalize conversion + evaluation (this produces your “results”)

Once you generate map pkls (or use their provided ones), run:

<pre node="[object Object]" class="my-4 overflow-auto whitespace-pre-wrap"><div class="bg-token-main-surface-secondary group relative overflow-hidden rounded-2xl _codeBlock_zqek1_1"><div class="text-token-text-secondary flex items-center justify-between px-4 py-2 text-xs font-normal tracking-wide"><span>bash</span><div class="flex items-center gap-2"><button type="button" class="text-token-text-secondary hover:text-token-text-primary focus-visible:outline-token-border-strong rounded-full px-3 py-1 text-[11px] font-medium opacity-0 transition group-hover:opacity-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2" aria-live="polite">Copy</button></div></div><pre class="m-0 overflow-x-auto p-4 whitespace-pre _codeContent_zqek1_76"><code class="block text-[10pt] leading-relaxed whitespace-pre language-bash"># inside the cheXlocalize repo
python heatmap_to_segmentation.py \
  --map_dir /path/to/maps_gradcam \
  --output_path ./my_gradcam_segmentations.json

python eval.py \
  --metric iou \
  --gt_path /path/to/CheXlocalize/gt_segmentations_test.json \
  --pred_path ./my_gradcam_segmentations.json \
  --save_dir ./results_my_gradcam \
  --true_pos_only True \
  --if_human_benchmark False
</code></pre></div></pre>

This is the cleanest way to produce **mIoU with bootstrap CIs** using the same protocol as the benchmark paper.

## What I need from you to “complete the results” with you

Since you offered to download datasets, here’s exactly what to do next:

1. Download **CheXlocalize** from Stanford AIMI.
2. Clone the CheXlocalize repo and run the evaluation on the included sample (so you confirm everything works).
3. Tell me:

* Your OS (Windows/macOS/Linux) and whether you have an NVIDIA GPU
* The folder structure of your CheXlocalize download (just `tree -L 2` or screenshots)
* The output CSV printed by `eval.py` (the `*_summary_results.csv` table)

Then, in the next message, I will:

* Interpret your numbers (what’s good/bad relative to known benchmarks),
* Help you write the “current results / analysis done” section for Project Update 1,
* Decide the best next ablation (uncertainty threshold vs explanation threshold vs combined).

One note: earlier PDFs you uploaded in this chat are no longer accessible (they expired). If you want me to align to a specific template or rubric inside those PDFs, re‑upload a screenshot of the relevant page(s).
