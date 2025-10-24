#!/bin/bash

# Script to run ablation experiments for different sentence sets
# This script processes multiple sports with different sentence variations

# Set up paths
DATA_DIR="/mnt/Data/mrt/MATDAT"
SENTENCES_DIR="data/sentences/ablation"
RESULTS_BASE="results/ablation"

# Create results directory
mkdir -p "$RESULTS_BASE"

# Define sports and their video names
declare -A SPORTS_VIDEOS
SPORTS_VIDEOS[diving]="diving"
SPORTS_VIDEOS[long_jump]="long_jump"
SPORTS_VIDEOS[pole_vault]="pole_vault"
SPORTS_VIDEOS[tumbling]="tumbling"
SPORTS_VIDEOS[tricking]="V1 V2 V3"

# Sport-specific parameters
declare -A CONTEXT_WINDOW
CONTEXT_WINDOW[diving]=600
CONTEXT_WINDOW[long_jump]=600
CONTEXT_WINDOW[pole_vault]=600
CONTEXT_WINDOW[tumbling]=600
CONTEXT_WINDOW[tricking]=600

declare -A MIN_DURATION
MIN_DURATION[diving]=15
MIN_DURATION[long_jump]=15
MIN_DURATION[pole_vault]=15
MIN_DURATION[tumbling]=15
MIN_DURATION[tricking]=15

declare -A MIN_AREA
MIN_AREA[diving]=15
MIN_AREA[long_jump]=15
MIN_AREA[pole_vault]=15
MIN_AREA[tumbling]=15
MIN_AREA[tricking]=15

echo "=========================================="
echo "Starting Ablation Experiments"
echo "=========================================="

# Iterate over each sport
for sport in diving long_jump pole_vault tumbling tricking; do
    echo ""
    echo "=========================================="
    echo "Processing sport: $sport"
    echo "=========================================="

    # Get video names for this sport
    videos="${SPORTS_VIDEOS[$sport]}"

    # Automatically detect available sentence sets for this sport
    # Extract set numbers from filenames, sort them numerically
    available_sets=$(ls "$SENTENCES_DIR/${sport}_set_"*.json 2>/dev/null | \
                     sed "s|.*${sport}_set_\([0-9]*\)\.json|\1|" | \
                     sort -n)

    if [ -z "$available_sets" ]; then
        echo "WARNING: No sentence files found for sport: $sport"
        echo "Skipping this sport..."
        continue
    fi

    echo "Found sentence sets: $available_sets"

    # Iterate over each available sentence set
    for set_num in $available_sets; do
        echo ""
        echo "---------- Sentence Set $set_num ----------"

        # Set the sentence file path
        sentences_file="$SENTENCES_DIR/${sport}_set_${set_num}.json"

        # Process each video for this sport
        for video in $videos; do
            echo ""
            echo ">>> Processing video: $video with set $set_num"

            # Define result directory for this experiment
            result_dir="${RESULTS_BASE}/${sport}_${video}_set${set_num}"

            # Check if result directory already exists
            if [ -d "$result_dir" ]; then
                echo "SKIPPING: Result directory already exists: $result_dir"
                continue
            fi

            echo "Step 1/2: Running multi_sentences.py..."
            python multi_sentences.py \
                --root_dir "$DATA_DIR" \
                --video_name "$video" \
                --sentences_file "$sentences_file" \
                --output_dir "$RESULTS_BASE" \
                --context_window "${CONTEXT_WINDOW[$sport]}" \
                --min_duration "${MIN_DURATION[$sport]}" \
                --min_area "${MIN_AREA[$sport]}" \
                --hist_sharey True \
                --hist_scale_y True \
                --draw_individual_plots False \
                --frames_to_plot 0 7500

            if [ $? -ne 0 ]; then
                echo "ERROR: multi_sentences.py failed for $video with set $set_num"
                continue
            fi

            # Rename results to specific experiment directory
            if [ -d "$RESULTS_BASE/$video" ]; then
                mv "$RESULTS_BASE/$video" "$result_dir"
                echo "Results saved to: $result_dir"
            fi

            echo "Step 2/2: Running summarize.py..."
            python summarize.py \
                --dataset_dir "$DATA_DIR" \
                --video_name "$video" \
                --results_dir "$result_dir" \
                --context_window "${CONTEXT_WINDOW[$sport]}" \
                --min_duration "${MIN_DURATION[$sport]}" \
                --min_area dynamic \
                --filter_separation 0.1 \
                --filter_range 0.4 \
                --filter_auc 0.4 \
                --hist_div 2 \
                --num_steps 10 \
                --frame_root "$DATA_DIR/imgs" \
                --frame_ext png

            if [ $? -ne 0 ]; then
                echo "ERROR: summarize.py failed for $video with set $set_num"
                continue
            fi

            echo ">>> Completed: $video with set $set_num"
        done
    done

    echo ""
    echo "Completed all sentence sets for sport: $sport"
done

echo ""
echo "=========================================="
echo "All Ablation Experiments Completed!"
echo "Results saved in: $RESULTS_BASE"
echo "=========================================="

echo ""
echo "=========================================="
echo "Generating Timing Summary..."
echo "=========================================="

# Generate comprehensive timing summary
python -c "
from timing_utils import aggregate_timing_logs
aggregate_timing_logs(
    log_file='$RESULTS_BASE/timing_log.json',
    output_file='$RESULTS_BASE/timing_summary.txt'
)
"

echo ""
echo "Timing summary saved to: $RESULTS_BASE/timing_summary.txt"
echo "Timing log saved to: $RESULTS_BASE/timing_log.json"
