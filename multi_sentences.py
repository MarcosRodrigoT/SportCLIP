import argparse
import json
import os
import pickle
import numpy as np
from utils import (
    Color,
    createGrounTruth,
    createClassEmbeddings,
    collectPredictions,
    groundTruth_Dict2List,
    predictions_Dict2List,
    computeRollingAverages,
    closingOperation,
    detectEvents,
    filterEvents,
    computeFrameLevelResults,
    plotGroundTruthVSPredictions,
    stitch_images,
    print_events,
    print_frame_level_results,
    draw_histograms,
    compute_crossing_point,
    save_logs,
)
from timing_utils import get_tracker


def main():
    parser = argparse.ArgumentParser(description="Process video highlights using sentence-based CLIP embeddings")

    # Main parameters
    # TODO: Remember to change below lines to for the final submission:
    # default root_dir should be "data"
    # default video_name should be "long_jump"
    parser.add_argument("--root_dir", type=str, default="/mnt/Data/mrt/SportCLIP-OlympicHighlights", help="Root directory containing video data")
    parser.add_argument("--video_name", type=str, default="longjump_video1", help="Name of the video (without extension) to process")
    parser.add_argument("--sentences_file", type=str, default="data/sentences/long_jump.json", help="Path to JSON file containing highlight and non-highlight sentences")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results (default: results)")

    # Window and filter parameters
    parser.add_argument("--context_window", type=int, default=600, help="Context window size for rolling average")
    parser.add_argument("--min_duration", type=int, default=15, help="Minimum duration for event filtering")
    parser.add_argument("--min_area", type=int, default=15, help="Minimum area for event filtering")

    # Plotting parameters
    parser.add_argument("--hist_sharey", type=bool, default=True, help="Share y axis among histplots when drawing multiple in a single figure")
    parser.add_argument("--hist_scale_y", type=bool, default=True, help="True -> maximum y-axis set dynamically. False -> set to the number of video frames")
    parser.add_argument("--draw_individual_plots", type=bool, default=True, help="Whether to draw individual plots")
    parser.add_argument("--frames_to_plot", type=int, nargs=2, default=[0, 7500], help="Frame range to plot [start, end]")

    args = parser.parse_args()

    # Initialize timing tracker
    tracker = get_tracker(log_file=f"{args.output_dir}/timing_log.json")

    # Extract sport name from sentences file for metadata
    sport_name = os.path.basename(args.sentences_file).replace("_set_", "/set").replace(".json", "")

    with tracker.track("multi_sentences.py - Setup", {"video_name": args.video_name, "sport": sport_name}):
        # Compute derived parameters
        instant_window = int(args.context_window / 10)
        closing_kernel = int(args.context_window / 10 + 1)

        # Load sentences from JSON file
        with open(args.sentences_file, "r") as f:
            sentences_data = json.load(f)
        highlight_sentences = sentences_data["highlight_sentences"]
        not_highlight_sentences = sentences_data["not_highlight_sentences"]

        # Create results folder
        results_folder = f"{args.output_dir}/{args.video_name}"
        os.makedirs(results_folder, exist_ok=True)

    # Get ground truth and empty predictions
    with tracker.track("Load ground truth annotations", {"video_name": args.video_name, "sport": sport_name}):
        ground_truth, predictions = createGrounTruth(annotations_file=os.path.join(args.root_dir, f"{args.video_name}.csv"))

    # Loop over the different sentences forming pairs
    pair_num = 0
    total_pairs = len(highlight_sentences) * len(not_highlight_sentences)
    for h_sent_idx, h_sentence in enumerate(highlight_sentences):
        for nh_sent_idx, nh_sentence in enumerate(not_highlight_sentences):
            print(f"\n{Color.ORANGE}{'='*80}")
            print(f"PAIR {pair_num}/{total_pairs} (H{h_sent_idx} x NH{nh_sent_idx})")
            print(f"{'='*80}{Color.RESET}")
            print(f"{Color.ORANGE}----------- Highlight sentence {h_sent_idx}:{' ': >6} -----------{Color.RESET}\n{h_sentence}")
            print(f"{Color.ORANGE}----------- Not a highlight sentence {nh_sent_idx}: -----------{Color.RESET}\n{nh_sentence}")

            metadata = {"video_name": args.video_name, "sport": sport_name, "pair_num": pair_num}

            # Get class (text) embeddings and the clip model
            with tracker.track("Create class embeddings (sentences)", metadata):
                classes = [h_sentence, nh_sentence]
                class_embeddings, clip_model = createClassEmbeddings(classes=classes)

            # Get predictions
            with tracker.track("Collect predictions (compute similarities)", metadata):
                predictions = collectPredictions(
                    root_dir=args.root_dir,
                    video_name=args.video_name,
                    class_embeddings=class_embeddings,
                    model=clip_model,
                    predictions=predictions,
                )

            # Drop any frames that have no predictions (e.g., off-by-one last frame)
            if isinstance(predictions, dict):
                missing_keys = [k for k, v in predictions.items() if v is None]
                for k in missing_keys:
                    predictions.pop(k, None)
                    # Keep ground truth aligned with predictions
                    if isinstance(ground_truth, dict):
                        ground_truth.pop(k, None)

            # Convert ground truth and predictions to lists
            with tracker.track("Convert predictions to lists", metadata):
                ground_truth_list = groundTruth_Dict2List(ground_truth_dict=ground_truth, skip_uncertainty=False)
                predictions_list, sentences_score_hist = predictions_Dict2List(predictions_dict=predictions)

            # Compute the rolling average for the predictions (instant & context)
            with tracker.track("Compute rolling averages", metadata):
                predictions_instant, predictions_context = computeRollingAverages(predictions_list, instant_window, args.context_window)

            # Compute coarse final predictions (those where instant predictions are above the context)
            with tracker.track("Compute coarse predictions", metadata):
                coarse_final_predictions = [1 if pred_inst > pred_cont else 0 for pred_inst, pred_cont in zip(predictions_instant, predictions_context)]

            # Closing operation (dilate/erode) of the coarse final predictions
            with tracker.track("Closing operation (dilate/erode)", metadata):
                refined_final_predictions = closingOperation(coarse_predictions=coarse_final_predictions, kernel_size=closing_kernel)

            # Compute areas enclosed between the instant and context predictions
            with tracker.track("Compute prediction areas", metadata):
                areas = [max(pred_inst - pred_cont, 0) for pred_inst, pred_cont in zip(predictions_instant, predictions_context)]

            # Collect events
            with tracker.track("Detect events", metadata):
                events_detected = detectEvents(predictions=areas, masks=refined_final_predictions)
                print_events(events_detected, color=Color.RED, message="Events detected")

            # Filter events by duration
            with tracker.track("Filter events by duration", metadata):
                events_filtered_by_duration = filterEvents(events=events_detected, min_duration=args.min_duration, min_area=0, reorder_by_relevance=False)
                print_events(events_detected, color=Color.GREEN, message="Events after filtering by duration")

            # Compute detected events' statistics
            with tracker.track("Compute event statistics", metadata):
                mean_area = np.mean([d["area"] for d in events_filtered_by_duration.values()])
                std_area = np.std([d["area"] for d in events_filtered_by_duration.values()])

                # Save the mean event area
                with open(f"{results_folder}/Mean area - Pair{pair_num}.pkl", "wb") as f:
                    pickle.dump(mean_area, f)

            # Filter events by area
            with tracker.track("Filter events by area", metadata):
                events_filtered = filterEvents(events=events_filtered_by_duration, min_duration=0, min_area=args.min_area, reorder_by_relevance=False)
                print_events(events_detected, color=Color.GREEN, message="Events after filtering by area")

            # Obtain frame level results
            with tracker.track("Compute frame-level results", metadata):
                recall, precision, fscore = computeFrameLevelResults(ground_truth=ground_truth_list, events_detected=events_filtered)
                print_frame_level_results(recall, precision, fscore, color=Color.CYAN)

            # Plot predictions against the ground truth
            if args.draw_individual_plots:
                with tracker.track("Plot predictions vs ground truth", metadata):
                    plotGroundTruthVSPredictions(
                        frames_to_plot=args.frames_to_plot,
                        ground_truth=ground_truth_list,
                        predictions=predictions_list,
                        predictions_instant=predictions_instant,
                        predictions_context=predictions_context,
                        coarse_final_predictions=coarse_final_predictions,
                        refined_final_predictions=refined_final_predictions,
                        areas=areas,
                        events_filtered=events_filtered,
                        fig_name=f"{'/'.join(results_folder.split('/')[1:])}/{args.video_name} - binary - tmp.png",
                        recall=recall,
                        precision=precision,
                        fscore=fscore,
                        mean_area=mean_area,
                        mean_std=std_area,
                    )

            # Draw histograms as individual distribution plots with KDE
            with tracker.track("Draw histograms", metadata):
                draw_histograms(sentences_score_hist, pair_num, args.hist_scale_y, results_folder, args.video_name)

            # Compute the crossing point between the two distributions
            with tracker.track("Compute crossing point", metadata):
                compute_crossing_point(pair_num, results_folder)

            # Stitch the histograms and predictions together
            with tracker.track("Stitch images", metadata):
                stitch_images(pair_num=pair_num, highlight_idx=h_sent_idx, non_highlight_idx=nh_sent_idx, results_folder=results_folder, video_name=args.video_name)

            # Save a log of the experiments
            with tracker.track("Save logs", metadata):
                save_logs(pair_num, h_sentence, nh_sentence, recall, precision, fscore, sentences_score_hist, results_folder, args.video_name)

            # Move on to the next pair
            pair_num += 1

    # Save timing data and print summary
    tracker.save()
    tracker.print_summary("multi_sentences.py Timing Summary")


if __name__ == "__main__":
    main()
