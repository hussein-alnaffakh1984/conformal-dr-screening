# ConformalDR — Trustworthy Cross-Population Diabetic Retinopathy Screening

Code, cached scores, and interactive demo for the manuscript:

**Improving Per-Grade Conformal Coverage in Cross-Population Diabetic Retinopathy
Screening via Class-Conditional Recalibration**

> ⚠️ **PRIVATE — manuscript under review.** Keep this repository **private** until
> publication. For peer review, share a read-only/anonymized link with the editor
> only. A public release with the final DOI will follow acceptance.

## What this repository contains

```
code/   reproduction scripts (run on Kaggle with the public datasets)
data/   cached model softmax scores + labels (reproduce every table/figure & the demo)
app/    self-contained interactive demo (single HTML file)
```

## Reproducing the results

All scripts are written for a Kaggle GPU notebook with the three public datasets
attached: **APTOS 2019**, **Messidor-2**, and **IDRiD**. Dataset paths are set at
the top of each script.

| Script | Reproduces |
| --- | --- |
| `code/dr_full_kaggle.py` | EfficientNet-B0 training; in-distribution / naive / local-recalibration coverage; per-grade (Mondrian) and class-conditional recalibration (Tables 2, 4; Figs 2–5) |
| `code/dr_robust_kaggle.py` | Weighted CP with softmax and 1280-d embedding features; label-prior TV distance (Tables 2, 3) |
| `code/dr_supp_kaggle.py` | ECE, temperature scaling, and α-sensitivity (Tables 6, 7) |
| `code/dr_qualitative_kaggle.py` | Qualitative input→prediction-set examples (Fig 7) |
| `code/dr_secondbackbone_kaggle.py` | ResNet-50 replication + shift-magnitude measures: domain-classifier AUC, MMD, TV (Tables 8, 9) |
| `code/dr_export_app_kaggle.py` | Exports `data/dr_app_data.json` used by the demo |

Setup: `pip install -r requirements.txt`

## Interactive demo

`app/index.html` is fully self-contained (no server, no build step, no data leaves
the browser). Open it locally by double-clicking, or deploy it:

1. Go to **netlify.com → Add new site → Deploy manually**.
2. Drag `app/index.html` (or the `app/` folder) and drop it.

The demo lets a user choose a deployment scenario (site, risk level α, local-label
budget, recalibration strategy) and see live coverage, per-grade safety, and
per-patient prediction sets, all computed from `data/dr_app_data.json`.

## Author

**Hussein Ali Hussein Al Naffakh**
Department of Medical Laboratory Techniques, University of Alkafeel, Najaf, Iraq
Email: <hussein.alnaffakh@alkafeel.edu.iq>

## Citation

Manuscript under review (2026); full citation and DOI to be added upon publication.
See `CITATION.cff`.

## License

Code is released under the MIT License (see `LICENSE`). The public datasets
(APTOS 2019, Messidor-2, IDRiD) remain under the terms of their original providers.
