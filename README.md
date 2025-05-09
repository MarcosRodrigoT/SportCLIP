# Text‑Guided Sports Highlights <br>*(SportCLIP — official code release)*

This repository contains the code, example data and reproducibility scripts for the paper  
**“Text‑Guided Sports Highlights: A CLIP‑Based Framework for Automatic Video Summarization”**.

Our framework turns **any** sports video into a concise highlight reel by:

1. Embedding every video frame with **CLIP ViT‑B/32**.  
2. Comparing the frame embedding against *multiple* highlight / non‑highlight sentence prompts.  
3. Filtering out low‐quality prompt pairs with distribution‑ and area‑based criteria.  
4. Averaging the remaining predictions and applying a light post‑process to deliver robust, frame‑level highlight masks.

The code reproduces all experiments on **MATDAT** (martial‑arts tricking), **YouTube Highlights** and our new **SportClip** benchmark covering diving, long‑jump, pole‑vault and tumbling datasets (details in Section V‑A of the paper).

---

## Project layout

```bash
.
├── extractor.py        # sample video frames & save CLIP embeddings
├── multi_sentences.py  # grid-search every (H, NH) prompt pair
├── crossing_point.py    # optional – find KDE intersections
├── pair_ranker.py      # rank pairs with simple histogram metrics
├── median_pipeline.py  # fuse best pairs → final events & metrics
├── utils.py            # shared helpers (I/O, plotting, metrics…)
├── requirements.txt
└── README.md           # you are here
```


`data/` (videos & CSVs) and `results/` are created on the fly.

---

## Quick start

```bash
# 1 Install dependencies (Python ≥ 3.9)
pip install -r requirements.txt        # choose a CUDA torch build if you have a GPU

# 2 Place assets under ./data
#      e.g.  data/V1.mpeg   data/V1.csv   …

# 3 Extract frame embeddings
python extractor.py --dataset-dir data

# 4 Score every highlight / non-highlight prompt pair
python highlight_pairs.py              # edit Config inside to change prompts & video list

# 5 (Optional) extra KDE diagnostics
python kde_crossings.py

# 6 Rank pairs, fuse their curves, detect events, and evaluate
python pair_ranker.py                  # writes intermediate variables
python median_pipeline.py              # final metrics & plots
```

---

## Script cheat-sheet

| Script  | Purpose |
|---------|---------|
|extractor.py	| Saves evenly spaced PNG frames and their 512-D CLIP vectors.|
|highlight_pairs.py|Produces a probability curve for every highlight vs. non-highlight prompt pair.|
|kde_crossings.py|Finds the intersection of highlight / non-highlight KDE curves (debug helper).|
|pair_ranker.py|Calculates separation / range / AUC metrics and filters poor pairs.|
|median_pipeline.py|Merges the surviving curves (median), detects highlight events, and scores them.|
|utils.py|Ground-truth loading, smoothing, morphology, metrics, plotting, colours.|

---

## Citing

If you find SportCLIP useful in your research, please cite:

```bibtex
@article{rodrigo2025sportclip,
  title   = {Text-Guided Sports Highlights: A CLIP-Based Framework for Automatic Video Summarization},
  author  = {Marcos Rodrigo and Carlos Cuevas and Narciso García},
  journal = {IEEE Transactions on Consumer Electronics},
  year    = {2025}
}
```

---

## License

The code is released under the MIT License (see `LICENSE`).
MATDAT, SportClip and YouTube Highlights have their own licenses – please respect each dataset’s terms of use.

Happy summarizing! 🎬🏅