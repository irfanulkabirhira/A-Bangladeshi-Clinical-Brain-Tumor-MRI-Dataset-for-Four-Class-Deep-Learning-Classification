# BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification

BDNeuro-MRI is a curated, leakage-free brain tumor MRI dataset sourced from Bangladeshi
clinical settings (Epic & CSCR Hospital), organized for four-class classification —
**Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. This repository contains the
end-to-end preprocessing pipeline used to build the dataset, plus a baseline model
benchmark confirming its quality and learnability.

> **Dataset DOI:** [10.17632/zwr4ntf94j.7](https://doi.org/10.17632/zwr4ntf94j.7) — Mendeley Data, V7
> **Dataset link:** <https://data.mendeley.com/datasets/zwr4ntf94j/7>

---

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Directory Structure](#directory-structure)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Data Integrity](#data-integrity)
- [Baseline Model Benchmark](#baseline-model-benchmark)
- [Repository Contents](#repository-contents)
- [How to Reproduce](#how-to-reproduce)
- [Requirements](#requirements)
- [Known Limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Publicly available brain tumor MRI datasets frequently suffer from unreported
train/test duplication and near-duplicate leakage, which inflates reported model
performance and undermines reproducibility. **BDNeuro-MRI** addresses this by rebuilding
the source imagery through an explicit, auditable pipeline: exact and near-duplicate
removal, per-class stratified splitting, and a hard programmatic assertion that zero
images overlap across splits.

- **Task**: Multi-class brain tumor classification from MRI images
- **Classes (4)**: Glioma, Meningioma, Pituitary, No Tumor
- **Total images**: 5,941
- **Split**: 70% Train (4,160) / 15% Validation (892) / 15% Test (889) — stratified per class
- **Source**: Epic & CSCR Hospital, Bangladesh — rebuilt from the [Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)](https://data.mendeley.com/datasets/zwr4ntf94j/6) on Mendeley Data (DOI: [10.17632/zwr4ntf94j.6](https://doi.org/10.17632/zwr4ntf94j.6))
- **This dataset (BDNeuro-MRI) is published on**: Mendeley Data, V7 — DOI: [10.17632/zwr4ntf94j.7](https://doi.org/10.17632/zwr4ntf94j.7) — <https://data.mendeley.com/datasets/zwr4ntf94j/7>
- **Integrity**: Verified zero content-level overlap between Train/Val/Test

---

## Dataset Description

| Property | Detail |
|---|---|
| Modality | Brain MRI (T1/T2-weighted clinical scans) |
| Classes | Glioma, Meningioma, Pituitary, No Tumor |
| Total images | 5,941 |
| Split ratio | 70% Train (4,160) / 15% Validation (892) / 15% Test (889) |
| Split method | Per-class stratified — every split mirrors the overall class distribution |
| Deduplication | Exact (MD5) + near-duplicate (perceptual hash, `imagehash.phash`) |
| Filenames | Anonymized, sequential (`<class>_<split>_<index>.<ext>`) — no original/patient identifiers retained |
| Leakage check | Programmatically asserted zero image-content overlap across splits |
| Dataset DOI | [10.17632/zwr4ntf94j.7](https://doi.org/10.17632/zwr4ntf94j.7) |

| Class | Train | Val | Test | Total | Overall % |
|---|---|---|---|---|---|
| Glioma | 1534 | 329 | 328 | 2191 | 36.9% |
| Meningioma | 1035 | 222 | 221 | 1478 | 24.9% |
| Pituitary | 1040 | 223 | 222 | 1485 | 25.0% |
| No Tumor | 551 | 118 | 118 | 787 | 13.2% |
| **Total** | **4160** | **892** | **889** | **5941** | **100%** |

- **Total images**: 5,941
- **Train**: 4,160 (70.0%)
- **Validation**: 892 (15.0%)
- **Test**: 889 (15.0%)

---

## Directory Structure

```
BDNeuro-MRI/
├── Brain_Tumor_MRI_Dataset_Rebuild_with_Benchmark.ipynb   # full pipeline notebook
├── README.md
└── Brain_Tumor_MRI_Dataset_Final/          # generated dataset (not tracked in git — see below)
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── no_tumor/
    ├── val/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── no_tumor/
    ├── test/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── no_tumor/
    ├── README.md                # auto-generated dataset card
    ├── class_distribution.csv
    ├── model_benchmark.csv
    ├── class_distribution_final.png
    └── model_benchmark.png
```

> **Note on git**: MRI image datasets are large and generally shouldn't be committed
> directly to a git repository. This repo hosts the **code**; the full dataset is
> distributed via Mendeley Data — see [Citation](#citation) for the DOI and link.

---

## Preprocessing Pipeline

The notebook (`Brain_Tumor_MRI_Dataset_Rebuild_with_Benchmark.ipynb`) implements the
full pipeline, runnable end-to-end in Google Colab:

1. **Ingest** — upload and extract the original source ZIP(s); old Train/Test folders
   are merged into a single pool before re-splitting.
2. **Class discovery** — images are assigned to one of the four canonical classes by
   folder name, with alias matching for common naming variants.
3. **Exact-duplicate removal** — MD5 content hashing drops byte-identical files; any
   exact duplicate found under conflicting class labels is flagged for manual review.
4. **Near-duplicate removal** — perceptual hashing (`imagehash.phash`, per class,
   configurable Hamming-distance threshold) removes re-compressed, re-exported, or
   minimally cropped copies of the same scan.
5. **(Optional) Patient-level grouping** — if filenames encode a patient/study ID, all
   images from the same patient can be constrained to a single split for the strictest
   possible leakage protection.
6. **Stratified 70/15/15 split** — each class is shuffled (fixed seed) and sliced
   independently so every split preserves the overall class balance.
7. **Leakage verification** — hard assertions confirm zero image-content overlap
   between Train/Val/Test; the notebook halts if this check fails.
8. **Anonymization & packaging** — files are copied into the final folder layout with
   sequential, anonymized filenames, then zipped for release.

---

## Data Integrity

- **Zero train/val/test leakage**, verified by content hash (MD5) over the final,
  exported dataset — re-checked independently of the split-generation step.
- **No duplicate images** within the released set (exact or near-duplicate).
- **Anonymized filenames** — no original filenames or embedded patient identifiers are
  present in the released files.

---

## Baseline Model Benchmark

To confirm the dataset is clean and learnable, seven architectures were trained on the
final 70/15/15 split and evaluated on the held-out Test set:

| Architecture | Paradigm | Test Acc. | Test F1 |
|---|---|---|---|
| CNN | From scratch | 0.8470 | 0.8457 |
| ViT | From scratch | 0.8256 | 0.8240 |
| Hybrid CNN–ViT | From scratch | 0.8189 | 0.8201 |
| ResNet50 | Frozen pretrained | 0.8031 | 0.8022 |
| DenseNet121 | Frozen pretrained | 0.8706 | 0.8703 |
| ResNet50 | Fine-tuned | 0.9708 | 0.9708 |
| **DenseNet121** | **Fine-tuned** | **0.9831** | **0.9831** |

The expected performance ordering holds — fine-tuned pretrained models outperform
frozen-pretrained models, which outperform from-scratch models — supporting that the
reported scores reflect genuine learnability rather than residual data leakage.

Full training code, hyperparameters, and per-epoch logs for all seven models are in the
benchmarking section of the notebook (Steps 9–12).

---

## Repository Contents

| File | Description |
|---|---|
| `Brain_Tumor_MRI_Dataset_Rebuild_with_Benchmark.ipynb` | End-to-end Colab notebook: dataset rebuild, dedup, stratified split, leakage verification, and 7-model benchmark |
| `README.md` | This file |

---

## How to Reproduce

1. Open `Brain_Tumor_MRI_Dataset_Rebuild_with_Benchmark.ipynb` in
   [Google Colab](https://colab.research.google.com/).
2. Set the runtime to a GPU: `Runtime > Change runtime type > T4 GPU`.
3. Run all cells (`Runtime > Run all`).
4. When prompted, upload the source dataset ZIP(s).
5. The notebook will rebuild, verify, benchmark, and export the final dataset as
   `Brain_Tumor_MRI_Dataset_Final.zip`.

Alternatively, the pre-built dataset can be downloaded directly from Mendeley Data:
<https://data.mendeley.com/datasets/zwr4ntf94j/7> (DOI: [10.17632/zwr4ntf94j.7](https://doi.org/10.17632/zwr4ntf94j.7)).

---

## Requirements

The notebook installs its own dependencies on first run (`pip install imagehash tqdm
timm -q`). Core libraries used:

- Python 3.10+
- `torch`, `torchvision`
- `timm`
- `imagehash`, `Pillow`
- `pandas`, `numpy`, `matplotlib`
- `scikit-learn`
- `tqdm`

A GPU (e.g. Colab's T4) is strongly recommended for the model benchmark section.

---

## Known Limitations

- Duplicate/near-duplicate removal is **image-level**. If multiple images originate from
  the same patient/study and are not visually near-identical, patient-level leakage is
  still possible unless the optional patient-level split was used when building this
  release.
- Class balance reflects the source clinical distribution and was not artificially
  equalized beyond proportional stratification across splits.

---

## Citation

If you use BDNeuro-MRI in your research, please cite both this dataset and the
original source data it was rebuilt from.

**This work:**

```bibtex
@dataset{bdneuro_mri,
  title        = {BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification},
  author       = {Hira, Md Irfanul Kabir and Hossain, Md Sohag and Bithee, Mst Moriom Akter and Sara, Umme Sara and Hasan, Md Mahmudul and Towsif, Abdullah Al and Ahmed, Md Kowsar},
  year         = {2026},
  publisher    = {Mendeley Data},
  version      = {V7},
  doi          = {10.17632/zwr4ntf94j.7},
  url          = {https://data.mendeley.com/datasets/zwr4ntf94j/7}
}
```

> HIRA, MD IRFANUL KABIR; HOSSAIN, MD SOHAG; BITHEE, MST MORIOM AKTER; Sara, Umme Sara;
> HASAN, MD MAHMUDUL; Towsif, Abdullah Al; Ahmed, Md Kowsar (2026), "BDNeuro-MRI: A
> Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning
> Classification", Mendeley Data, V7, doi: [10.17632/zwr4ntf94j.7](https://doi.org/10.17632/zwr4ntf94j.7)

**Source dataset:**

```bibtex
@dataset{hira2026braintumor,
  title        = {Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)},
  author       = {Hira, Md Irfanul Kabir and Hossain, Md Sohag and Bithee, Mst Moriom Akter and Sara, Umme Sara and Hasan, Md Mahmudul and Towsif, Abdullah Al and Ahmed, Md Kowsar},
  year         = {2026},
  publisher    = {Mendeley Data},
  version      = {V6},
  doi          = {10.17632/zwr4ntf94j.6},
  url          = {https://data.mendeley.com/datasets/zwr4ntf94j/6}
}
```

---

## License

[FILL IN — e.g. CC BY-NC 4.0. Confirm this is compatible with the data usage/consent
terms from Epic & CSCR Hospital before publishing.]

---

## Acknowledgements

Source imagery rebuilt from the **Brain Tumor MRI Dataset (Glioma, Meningioma,
Pituitary, No Tumor)**, originally sourced from Epic & CSCR Hospital, Bangladesh, and
published on Mendeley Data:

> HIRA, MD IRFANUL KABIR; HOSSAIN, MD SOHAG; BITHEE, MST MORIOM AKTER; Sara, Umme Sara;
> HASAN, MD MAHMUDUL; Towsif, Abdullah Al; Ahmed, Md Kowsar (2026), "Brain Tumor MRI
> Dataset (Glioma, Meningioma, Pituitary, No Tumor)", Mendeley Data, V6,
> doi: [10.17632/zwr4ntf94j.6](https://doi.org/10.17632/zwr4ntf94j.6)
>
> Dataset link: <https://data.mendeley.com/datasets/zwr4ntf94j/6>

BDNeuro-MRI rebuilds this source data with exact/near-duplicate removal, a stratified
70/15/15 split, and verified zero data leakage across splits, and adds a baseline
model benchmark. The rebuilt dataset itself is published on Mendeley Data:

> HIRA, MD IRFANUL KABIR; HOSSAIN, MD SOHAG; BITHEE, MST MORIOM AKTER; Sara, Umme Sara;
> HASAN, MD MAHMUDUL; Towsif, Abdullah Al; Ahmed, Md Kowsar (2026), "BDNeuro-MRI: A
> Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning
> Classification", Mendeley Data, V7, doi: [10.17632/zwr4ntf94j.7](https://doi.org/10.17632/zwr4ntf94j.7)
>
> Dataset link: <https://data.mendeley.com/datasets/zwr4ntf94j/7>

[Add any additional acknowledgements, funding sources, or ethical approval/IRB
statements here.]
