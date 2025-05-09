# Text‑Guided Sports Highlights <br>*(SportCLIP — official code release)*

This repository contains the code, example data and reproducibility scripts for the paper  
**“Text‑Guided Sports Highlights: A CLIP‑Based Framework for Automatic Video Summarization”**.

Our framework turns **any** sports video into a concise highlight reel by leveraging [OpenAI CLIP](https://github.com/openai/CLIP) image–text embeddings. The workflow is broken into three clear stages:

1. **Frame & embedding extraction** (`extractor.py`)
2. **Prompt engineering & event detection** (`multi_sentences.py`)
3. **Post‑processing & metric computation** (`entire_pipeline.py`)

---

## Table of Contents

* [Installation](#installation)
* [Quick‑start](#quick-start)
* [Directory Structure](#directory-structure)
* [Configuration](#configuration)
* [Outputs & Results](#outputs--results)
* [Citation](#citation)
* [License](#license)

---

## Installation

> Tested with **Python ≥ 3.10** on Linux/macOS and with an NVIDIA GPU + CUDA (CPU also works, but slower).

```bash
# 1. Clone the repository
$ git clone https://github.com/<your‑username>/<repo‑name>.git
$ cd <repo‑name>

# 2. Create an isolated environment (recommended)
$ python3 -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install Python dependencies
$ pip install --upgrade pip
$ pip install -r requirements.txt

# 4. Install and test FFmpeg / FFprobe (needed by extractor.py)
$ ffmpeg -version && ffprobe -version  # should print version info
```

---

## Quick‑start

Assuming you have a video **`data/long_jump.mp4`** and its frame‑level annotation file **`data/long_jump.csv`**:

```bash
# 1. Extract frames & CLIP embeddings (≈ real‑time length, GPU makes this faster)
$ python extractor.py \
        --video data/long_jump.mp4 \
        --annotations data/long_jump.csv \
        --frames-per-second 29.97

# 2. Run multiple sentence prompts to discover the best highlight / non‑highlight pairs
$ python multi_sentences.py
# ▸ results are written to results/<video_name>/

# 3. Generate final predictions, plots & evaluation metrics
$ python entire_pipeline.py
# ▸ key outputs appear in results/<video_name>/Final result.png
```

All default hyper‑parameters are hard‑coded in the corresponding scripts and can be overridden via command‑line flags or by editing the classes at the top of each file.

---

## Directory Structure

```text
.
├── extractor.py
├── multi_sentences.py
├── entire_pipeline.py
├── utils.py
├── requirements.txt
├── data/                 # ─► videos (.mp4) & CSV annotations
├── imgs/                 # ─► one sub‑folder per video with extracted frames     (generated)
├── img_embeddings/       # ─► one sub‑folder per video with CLIP embeddings      (generated)
└── results/              # ─► per‑video plots, logs & metrics                    (generated)
```

---

## Configuration

Most hyper‑parameters live at the top of each script. &#x20;

* **Dataset layout**  Place each `*.mp4` next to its `*.csv` annotation in the `data/` folder.
  The CSV must contain at least the columns `First frame`, `Last frame` and `Event type`.
* **GPU vs CPU**  All scripts will automatically switch to CUDA if available.
* **Custom prompts**  Edit the `highlight_sentences` and `not_highlight_sentences` lists in `multi_sentences.py`.

---

## Outputs & Results

After running the full pipeline you will find:

| Path                               | Description                                       |
| ---------------------------------- | ------------------------------------------------- |
| `results/<video>/Final result.png` | Summary plot of ground truth vs predictions       |
| `results/<video>/Pairs used.txt`   | List of prompt pairs kept after filtering         |
| `results/<video>/*.pkl`            | Pickled KDE curves, scores & auxiliary stats      |
| `results/<video>/*.png`            | Histograms and stitched diagnostic images         |

---

## Citation

If you find our work useful in your research, please cite:

```bibtex
@article{rodrigo2025sportclip,
  title   = {Text-Guided Sports Highlights: A CLIP-Based Framework for Automatic Video Summarization},
  author  = {Marcos Rodrigo, Carlos Cuevas, and Narciso García},
  journal = {IEEE Transactions on Consumer Electronics},
  year    = {2025}
}
```

---

## License

This project is released under the **MIT License**. &#x20;
See [`LICENSE`](LICENSE) for details.

Happy summarizing! 🎬🏅