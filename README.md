# Text-Guided Sports Video Summarizer <br>*(SportCLIP — official code release)*

This repository contains the code, example data and reproducibility scripts for the paper  
**“Text-Guided Sports Highlights: A CLIP-Based Framework for Automatic Video Summarization”**.

[[`Webpage`]](http://gti.ssr.upm.es/data/sportclip) | [[`Paper`]](#citation)

![Architecture](assets/Architecture.jpg)

Our framework turns **any** sports video into a concise highlight reel by leveraging [OpenAI CLIP](https://github.com/openai/CLIP) image–text embeddings. The workflow is broken into three clear stages:

1. **Frame & embedding extraction** (`extractor.py`)
2. **Prompt engineering** (`multi_sentences.py`)
3. **Highlight extraction & evaluation** (`summarize.py`)

---

## Table of Contents

* [Installation](#installation)
* [Quick-start](#quick-start)
* [Directory Structure](#directory-structure)
* [Configuration](#configuration)
* [Outputs & Results](#outputs--results)
* [Citation](#citation)

---

## Installation

> Tested with **Python ≥ 3.10** on Linux/macOS and with an NVIDIA GPU + CUDA (CPU also works, but slower).

```bash
# 1. Clone the repository
$ git clone https://github.com/MarcosRodrigoT/SportCLIP.git
$ cd SportCLIP

# 2. Create an isolated environment (recommended)
$ python3 -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install Python dependencies
$ pip install --upgrade pip
$ pip install -r requirements.txt

# 4. Install and test FFmpeg / FFprobe (needed by extractor.py)
$ sudo apt install -y ffmpeg
$ ffmpeg -version && ffprobe -version  # should print version info
```

---

## Quick-start

Assuming you have a video (e.g., **`data/long_jump.mp4`**) and its frame-level annotation file (e.g., **`data/long_jump.csv`**):

```bash
# 1. Extract frames & CLIP embeddings
$ python extractor.py \
        --video data/long_jump.mp4 \
        --annotations data/long_jump.csv \
        --frames-per-second 29.97

# 2. Run multiple sentence prompts to discover the best highlight / non-highlight pairs
$ python multi_sentences.py \
        --video_name long_jump \
        --sentences_file data/sentences/long_jump.json
# ▸ results are written to results/<video_name>/

# 3. Generate final video summary (highlight reel), plots & evaluation metrics
$ python summarize.py \
        --video_name long_jump
# ▸ results are written to results/<video_name>/
```

All default hyper-parameters can be overridden via command-line flags. Run each script with `--help` to see all available options.

---

## Directory Structure

```text
.
├── extractor.py
├── multi_sentences.py
├── summarize.py
├── utils.py
├── requirements.txt
├── data/
│   ├── sentences/        # ─► JSON files with highlight/non-highlight sentences  (provided)
│   └── ...               # ─► videos & ground-truth annotations
├── imgs/                 # ─► one sub-folder per video with extracted frames     (generated)
├── img_embeddings/       # ─► one sub-folder per video with CLIP embeddings      (generated)
└── results/              # ─► per-video plots, logs & metrics                    (generated)
```

---

## Configuration

### Dataset layout

Place all **videos** and their **ground-truth annotation files** in the `data/` folder:

* Videos: `*.mp4`, `*.mpeg`, `*.avi`, …
* CSV annotations: `*.csv` (must include the columns `Event type`, `First frame`, `Last frame`, `Num. frames`).

When trying out [MATDAT](http://gti.ssr.upm.es/data/matdat) and [SportCLIP](http://gti.ssr.upm.es/data/sportclip) datasets, extract all videos and ground-truth annotation files to this directory:

```text
data/
├── V1.mpeg
├── V1.csv
├── V2.mpeg
├── V2.csv
├── V3.mpeg
├── V3.csv
├── diving.mp4
├── diving.csv
├── long_jump.mp4
├── long_jump.csv
├── pole_vault.mp4
├── pole_vault.csv
├── tumbling.mp4
└── tumbling.csv
```

---

### Trying out new sentences or sports

Sentence prompts are stored as JSON files in the **`data/sentences/`** directory. Each sport has its own JSON file containing `highlight_sentences` and `not_highlight_sentences`. To craft sport-specific prompts:

1. Create a new JSON file in `data/sentences/` (e.g., `long_jump.json`)
2. Use the template below as a guide:

```json
{
    "highlight_sentences": [
        "An athlete sprinting down the runway before launching into the air, reaching for maximum distance",
        "A long jumper executing a well-timed takeoff, soaring through the air before landing in the sand pit",
        "A person accelerating down the track, generating momentum for an explosive jump",
        "An athlete gliding through the air with extended arms and legs, preparing for a controlled landing",
        "A competitor demonstrating strength and precision in a long jump attempt",
        "A long jumper executing a perfect flight phase, reaching their peak height before descent",
        "An athlete pushing off the ground with powerful force, achieving an impressive airborne moment",
        "Athlete running, jumping into the air and landing in the sand pit"
    ],
    "not_highlight_sentences": [
        "A long jumper adjusting their starting position on the runway",
        "A person discussing jump techniques with a coach",
        "An athlete waiting for their turn while observing competitors",
        "A group of athletes standing near the sand pit, preparing for their jumps",
        "A long jumper walking back after a completed attempt",
        "A judge measuring the distance of a jump while athletes watch",
        "A competitor stretching and warming up before their jump",
        "Athlete relaxed, greeting judges, celebrating"
    ]
}
```

3. Run `multi_sentences.py` with your custom sentences file:

```bash
$ python multi_sentences.py \
        --root_dir data \
        --video_name long_jump \
        --sentences_file data/sentences/long_jump.json \
        --context_window 600 \
        --min_duration 15 \
        --min_area 15 \
        --hist_sharey True \
        --hist_scale_y True \
        --draw_individual_plots True \
        --frames_to_plot 0 7500
```

**Key parameters:**
- `--root_dir`: Root directory containing video data and annotations
- `--video_name`: Name of the video (without extension) to process
- `--sentences_file`: Path to JSON file with highlight/non-highlight sentences
- `--context_window`: Context window size for rolling average (default: 600)
- `--min_duration`: Minimum duration (in frames) for event filtering (default: 15)
- `--min_area`: Minimum area for event filtering (default: 15)
- `--hist_sharey`: Share y-axis among histplots when drawing multiple plots (default: True)
- `--hist_scale_y`: Dynamically scale y-axis vs. fixed to frame count (default: True)
- `--draw_individual_plots`: Whether to draw individual diagnostic plots (default: True)
- `--frames_to_plot`: Frame range to plot [start, end] (default: 0 7500)

Pre-configured sentence files are provided for: **diving**, **long_jump**, **pole_vault**, **tumbling**, **tricking**, **100_meters**, **javelin**, and **high_jump**.

Run `python multi_sentences.py --help` to see all available options.

---

### Generating the final highlight reel

The **`summarize.py`** script accepts command-line arguments to customize the highlight extraction process. All parameters have sensible defaults but can be overridden:

```bash
$ python summarize.py \
        --dataset_dir data \
        --video_name long_jump \
        --results_dir results \
        --context_window 600 \
        --min_duration 15 \
        --min_area dynamic \
        --filter_separation 0.1 \
        --filter_range 0.4 \
        --filter_auc 0.4 \
        --hist_div 2 \
        --num_steps 10 \
        --export_highlight_reel \
        --export_mode individual \
        --fps 30 \
        --frame_root data/imgs \
        --frame_ext png \
        --out_filename highlight.mp4
```

**Key parameters:**
- `--dataset_dir`: Root directory containing video data and annotations (default: data)
- `--video_name`: Name of the video (without extension) to process (default: long_jump)
- `--results_dir`: Results directory containing intermediate outputs (default: results)
- `--context_window`: Context window size for rolling average (default: 600)
- `--min_duration`: Minimum duration (in frames) for event filtering (default: 15)
- `--min_area`: Use `"dynamic"` for automatic threshold or a numeric value (e.g., `15`)
- `--filter_separation`: Minimum separation threshold for histogram filtering (default: 0.1)
- `--filter_range`: Minimum range threshold for histogram filtering (default: 0.4)
- `--filter_auc`: Maximum AUC threshold for histogram filtering (default: 0.4)
- `--hist_div`: Histogram division factor for area filtering (default: 2)
- `--num_steps`: Step size for ablation metrics thresholds (default: 10)
- `--export_highlight_reel`: Export the highlight reel video (disabled by default)
- `--export_mode`: Export mode: `"individual"` for separate videos per highlight, `"combined"` for single video (default: individual)
- `--fps`: Frames per second for exported highlight reel (default: 30)
- `--frame_root`: Root directory containing extracted frames (default: data/imgs)
- `--frame_ext`: Frame image file extension (default: png)
- `--out_filename`: Output filename for highlight reel (default: highlight.mp4)

Run `python summarize.py --help` to see all available options.

---

## Outputs & Results

After running the full pipeline you will find:

| Path                               | Description                                       |
| ---------------------------------- | ------------------------------------------------- |
| `results/<video>/Final result.png` | Summary plot of ground truth vs predictions       |
| `results/<video>/Pairs used.txt`   | List of prompt pairs kept after filtering         |
| `results/<video>/*.pkl`            | Pickled KDE curves, scores & auxiliary stats      |
| `results/<video>/*.png`            | Histograms and stitched diagnostic images         |
| `results/<video>/highlight.mp4`    | Final highlight reel                              |

---

## Citation

If you find our work useful in your research, please cite:

```bibtex
@article{rodrigo2025sportclip,
  title   = {Text-Guided Sports Highlights: A CLIP-Based Framework for Automatic Video Summarization},
  author  = {Marcos Rodrigo, Carlos Cuevas, and Narciso García},
  journal = {IEEE Access},
  year    = {2025}
}
```

Happy summarizing! 🎬🏅