# A Novel Framework for End-Diastolic and End-Systolic Frame Localization in Contrast and Non-Contrast Echocardiography Without Manual Annotations  
**Sahaj Patel, BS, MS, PhD; Neeraj Kummaragunta, BS; Krishin Yerabolu, N/A; Abdulla Shahid, BS; Priyank Baria, BSN; Cynthia Li, N/A; Nehal Vekariya, MS; Akhil Pampana, MS; Pankaj Arora, MD; Garima Arora, MD**  

**Paper (AHA / Circulation Abstract 4373195):** https://www.ahajournals.org/doi/10.1161/circ.152.suppl_3.4373195

A fully automated pipeline for localizing **end-diastolic (ED)** and **end-systolic (ES)** frames in echocardiography cine loops using **optional LV ROI detection (YOLO)** and **Robust PCA + SVD** motion signal extraction—**without requiring manually annotated ED/ES reference frames**.

---

## Table of Contents
- [Project Description](#project-description)
- [Why This Project](#why-this-project)
- [Method Overview (Theory)](#method-overview-theory)
  - [1) ROI / Cropping](#1-roi--cropping)
  - [2) Robust PCA Decomposition](#2-robust-pca-decomposition)
  - [3) Motion Signal Extraction via SVD](#3-motion-signal-extraction-via-svd)
  - [4) Cycle Selection (Spectral Dominance Ratio)](#4-cycle-selection-spectral-dominance-ratio)
  - [5) ED/ES Localization (Peak/Valley Detection)](#5-edes-localization-peakvalley-detection)
  - [Exclusions / Failure Cases](#exclusions--failure-cases)
- [Results Summary](#results-summary)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [Supported Input Modes](#supported-input-modes)
  - [Crop Modes](#crop-modes)
  - [Quick Start: Single AVI (Inference-only)](#quick-start-single-avi-inference-only)
  - [Quick Start: Batch from Filename List (Inference-only)](#quick-start-batch-from-filename-list-inference-only)
  - [Quick Start: EchoNet CSV + AVI (Labeled)](#quick-start-echonet-csv--avi-labeled)
  - [Quick Start: UAB NPY + PKL (Labeled)](#quick-start-uab-npy--pkl-labeled)
  - [Column Mapping for PKL Tables](#column-mapping-for-pkl-tables)
  - [Batch Slicing](#batch-slicing)
- [Outputs](#outputs)
- [Tips](#tips)
- [Citations / Acknowledgements](#citations--acknowledgements)

---

## Project Description

This repository implements a **generalizable ED/ES frame localization framework** for echocardiography cine loops that works on **contrast** and **non-contrast** images. The approach uses:

- **Optional LV ROI detection** via a YOLO (v12) object detector (or fixed crop / full-frame),
- **Robust principal component analysis (RobustPCA)** to separate low-rank structure from sparse variations,
- **SVD-derived temporal signals** to capture pseudo-periodic cardiac motion,
- **Spectral dominance + zero-crossing stability** to select the best motion signal,
- **Peak/valley detection** to infer ED and ES frames.

The method is designed to avoid reliance on **manually annotated ED/ES training labels**, which can be costly, variable, and dataset-specific.

---

## Why This Project

End-diastolic (ED) and end-systolic (ES) frames are critical for left ventricular (LV) volume measurements in echocardiography, but manual selection can have **high inter- and intra-observer variability**.

While deep learning methods have advanced ED/ES localization, many approaches require manually labeled ED/ES reference frames and may fail to generalize across different image types and acquisition conditions (especially **contrast vs non-contrast**).

This project focuses on a **fully automated, annotation-free ED/ES localization method** that can operate with or without a DL-based ROI detector.

---

## Method Overview (Theory)

### 1) ROI / Cropping

The pipeline supports three crop modes:

- `yolo`: **YOLO crop** (select the **single best-confidence LV box across all frames**, and crop the entire cine loop to that ROI)
- `full`: **no crop** (use full frame)
- `fixed`: **fixed crop rectangle** (X1,Y1,X2,Y2)

Cropping helps reduce background clutter and stabilize the extracted motion signal, especially for contrast studies.

### 2) Robust PCA Decomposition

After cropping (or using full frame), the cine loop is reshaped into a matrix representation and decomposed using **Robust PCA** into:

- **Low-rank (L)**: dominant structural content (stable anatomy/appearance)
- **Sparse (S)**: transient variations / motion / outliers

This decomposition yields a more stable basis for extracting pseudo-periodic cardiac motion.

### 3) Motion Signal Extraction via SVD

SVD is applied to the low-rank matrix. The pipeline extracts the **top three left singular vectors** (U1, U2, U3), each serving as a candidate 1D temporal motion signal.

### 4) Cycle Selection (Spectral Dominance Ratio)

Each candidate U signal is analyzed to identify pseudo-periodic cardiac cycles using a **Spectral Dominance Ratio** (how strongly a dominant frequency stands out).

The algorithm then:
- detects cycles,
- computes **zero-crossings** and their variance,
- selects the **U with lowest zero-crossing variance** (and at least **two cycles**) as the representative cardiac motion signal.

### 5) ED/ES Localization (Peak/Valley Detection)

A peak detection step identifies local extrema in the selected motion signal:
- one extremum corresponds to **ED** (typically maximal filling),
- the opposite extremum corresponds to **ES** (typically maximal contraction).

This yields predicted ED/ES frame indices for each cine loop.

### Exclusions / Failure Cases

If the selected motion signal contains only **one cardiac cycle**, the case is excluded (insufficient periodic structure for stable ED/ES localization). This is reported in the outputs so excluded cases are traceable.

---

## Results Summary

Validated on:
- **UAB dataset**: N=984 (912 contrast, 72 non-contrast)
- **EchoNet-Dynamic**: N=10,030 (non-contrast) for external validation

YOLO model (trained exclusively on UAB data: 1394 train / 298 val / 300 test):
- **mAP50 = 0.994**
- **mAP50–95 = 0.717**

Frame localization mean absolute error (MAE):
- **UAB**: ED = 2.65 ± 2.95 frames (median 2), ES = 1.58 ± 1.49 frames (median 1)
- **EchoNet**: ED = 3.75 ± 4.02 frames (median 2), ES = 2.72 ± 2.81 frames (median 2)

Excluded due to only one cardiac cycle in the selected motion signal:
- 5 UAB cases
- 115 EchoNet cases

---

## Repository Layout

Typical structure:
```
.
├── scripts/
│   └── run_edvesv.py
├── requirements.txt
└── README.md
```

> Your local repository may include additional modules (RobustPCA utils, YOLO wrappers, helpers, etc.).

---

## Installation

```bash
pip install -e .
```

If you run on GPU, ensure a CUDA-enabled PyTorch build is installed.

---

# How to Run

The main entry point is:

```bash
python scripts/run_edvesv.py [args...]
```

## Supported Input Modes

This repo supports **labeled evaluation** and **inference-only** workflows:

### Inference-only (no ED/ES required)
- `--dataset single_avi`  
  Run on a **single `.avi`** file and output predicted ED/ES frames.

- `--dataset echonet_list`  
  Run on a **batch of videos** specified by a **text file** of filenames (one per line).  
  This is useful when you only have filenames and want predictions without any ED/ES labels.

### Labeled evaluation (ED/ES provided)
- `--dataset echonet_csv`  
  AVI videos + a CSV that includes `FileName` (and optionally `EDV`, `ESV` for evaluation).

- `--dataset uab_pkl`  
  NPY arrays + a PKL table (column names can be mapped).

> Note: `--output_csv` is always required. Even in inference-only mode, results are written to a CSV.

---

## Crop Modes

- `--crop_mode yolo`  
  Uses YOLO weights from `--yolo_weights` and crops to the best LV ROI.

- `--crop_mode full`  
  Uses full frame (no cropping).

- `--crop_mode fixed --fixed_crop X1 Y1 X2 Y2`  
  Uses a constant rectangle crop.

---

## Quick Start: Single AVI (Inference-only)

```bash
python scripts/run_edvesv.py   --dataset single_avi   --video_path /data/project/arora_lab_imaging/Dataset/Videos/0X4EFB94EA8F9FC7C2.avi   --crop_mode yolo   --yolo_weights /data/user/nk7/EigenValue_EDV_ESV/EchoNet-Dynamic2/UABWeight/best.pt   --output_csv out_single.csv
```

Optional: set a custom identifier for the output row:
```bash
python scripts/run_edvesv.py   --dataset single_avi   --video_path /path/to/video.avi   --case_id MyCase123   --crop_mode full   --output_csv out_single.csv
```

---

## Quick Start: Batch from Filename List (Inference-only)

Create a text file (e.g., `filelist.txt`) with **one filename per line** (with or without `.avi`).  
Blank lines are ignored. Lines starting with `#` are treated as comments.

Example `filelist.txt`:
```
# EchoNet examples
0X4EFB94EA8F9FC7C2
0X211D307253ACBEE7.avi
0XD00B14807A0FA2B
```

Run:

```bash
python scripts/run_edvesv.py   --dataset echonet_list   --video_dir /data/project/arora_lab_imaging/Dataset/Videos   --file_list /path/to/filelist.txt   --crop_mode full   --output_csv out_batch.csv
```

---

## Quick Start: EchoNet CSV + AVI (Labeled)

### 1) YOLO crop
```bash
python scripts/run_edvesv.py   --dataset echonet_csv   --video_dir /data/project/arora_lab_imaging/Dataset/Videos   --input_csv /data/project/arora_lab_imaging/edv_esv.csv   --yolo_weights /data/user/nk7/EigenValue_EDV_ESV/EchoNet-Dynamic2/UABWeight/best.pt   --crop_mode yolo   --output_csv out_yolo.csv
```

### 2) Full-frame (no crop)
```bash
python scripts/run_edvesv.py   --dataset echonet_csv   --video_dir /data/project/arora_lab_imaging/Dataset/Videos   --input_csv /data/project/arora_lab_imaging/edv_esv.csv   --crop_mode full   --output_csv out_full.csv
```

### 3) Fixed crop
Matches notebook constants:
`X1,Y1,X2,Y2 = 24,14,88,94`

```bash
python scripts/run_edvesv.py   --dataset echonet_csv   --video_dir /data/project/arora_lab_imaging/Dataset/Videos   --input_csv /data/project/arora_lab_imaging/edv_esv.csv   --crop_mode fixed   --fixed_crop 24 14 88 94   --output_csv out_fixed.csv
```

---

## Quick Start: UAB NPY + PKL (Labeled)

```bash
python scripts/run_edvesv.py   --dataset uab_pkl   --npy_base_dir /data/project/arora_lab_imaging/C_NC_EF/RAW_Dataset   --input_pkl /data/project/arora_lab_imaging/final_LV_GT_log_with_array_v6.pkl   --yolo_weights /data/user/nk7/EigenValue_EDV_ESV/EchoNet-Dynamic2/UABWeight/best.pt   --crop_mode yolo   --output_csv UAB_RobutsPCA_All_v2.csv
```

---

## Column Mapping for PKL Tables

If your PKL column names differ, map them with:

- `--id_col` (case identifier; default tries `FileName`, `filename`, `Patient`)
- `--ed_col` (default tries `EDV` / `ED`)
- `--es_col` (default tries `ESV` / `ES`)
- `--npy_col` (relative `.npy` path; default tries `npy_path`, `Path`, `FilePath`)

Example:

```bash
python scripts/run_edvesv.py   --dataset uab_pkl   --input_pkl /path/to/table.pkl   --npy_base_dir /path/to/npy_base   --id_col PatientFolder   --ed_col ED   --es_col ES   --npy_col NPYRelPath   --crop_mode full   --output_csv out.csv
```

---

## Batch Slicing

To process only a subset of rows (useful for batching on HPC):

```bash
python scripts/run_edvesv.py ... --start_idx 5000 --end_idx 7500
```

> For `single_avi`, slicing is typically unnecessary.

---

## Outputs

The pipeline writes a CSV specified by `--output_csv`.

Typical output includes:
- identifiers (e.g., `FileName` / `Patient` / ID)
- predicted ED frame index
- predicted ES frame index
- crop mode and crop coordinates used (if applicable)
- status flags (e.g., excluded due to single-cycle, YOLO failure fallback, read failure, etc.)

> Exact column names depend on `scripts/run_edvesv.py`. If you paste your current output CSV header here, I can document each field precisely.

---

## Tips

- **Use `crop_mode=yolo`** for best robustness when backgrounds vary or contrast introduces clutter.
- **Use `crop_mode=full`** as a baseline or if YOLO weights are unavailable.
- **Use `crop_mode=fixed`** when you need strict reproducibility across experiments or your dataset is consistently centered.

---

## Citations / Acknowledgements

- EchoNet-Dynamic dataset is used for external validation.
- YOLO (v12) is used for optional LV ROI detection.
- Robust PCA + SVD is used for extracting a stable pseudo-periodic cardiac motion signal for ED/ES localization.
