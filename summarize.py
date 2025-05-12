import os
import cv2
import pickle
import numpy as np
import pandas as pd
from utils import Color
from itertools import accumulate
from utils import (
    Color,
    createGrounTruth,
    groundTruth_Dict2List,
    closingOperation,
    detectEvents,
    filterEvents,
    computeFrameLevelResults,
    plotGroundTruthVSPredictions,
    print_frame_level_results,
    computeRollingAverages,
)


def create_histogram(scores):
    bin_edges = np.linspace(0, 1, 50)
    hist_values, _ = np.histogram(scores, bins=bin_edges)
    return hist_values


def compute_histogram_auc(hist_values):
    # Convert to numpy array if it isn't already
    hist_values = np.asarray(hist_values, dtype=float)

    # Normalize the histogram so that the highest bin = 1
    max_val = hist_values.max()
    normalized_vals = hist_values / max_val

    # Each bin covers an equal fraction of the [0,1] interval:
    n_bins = len(normalized_vals)
    bin_width = 1.0 / n_bins

    # Sum of heights times bin width
    auc = np.sum(normalized_vals) * bin_width
    return auc


def compute_intermediate_vars(vid_name):
    intermediate_vars_dict = {x: None for x in range(0, 64)}
    f_scores = {x: None for x in range(0, 64)}

    for pair_num in range(0, 64):
        # Compute separation
        h_mean = None
        # Use the maximum of the KDE curves instead of the means
        with open(f"results/{vid_name}/KDE - Pair{pair_num} - H.pkl", "rb") as f:
            h_kde = pickle.load(f)
        with open(f"results/{vid_name}/KDE - Pair{pair_num} - NH.pkl", "rb") as f:
            nh_kde = pickle.load(f)
        h_mean = h_kde["x_coord"][np.argmax(h_kde["y_coord"])]
        separation = 0.5 - h_mean

        # Compute AUC
        with open(f"results/{vid_name}/Scores - Pair{pair_num} - H.pkl", "rb") as f:
            h_scores = pickle.load(f)
        hist_values = create_histogram(h_scores)
        auc = compute_histogram_auc(hist_values)

        # Retrieve F-scores
        with open(f"results/{vid_name}/{vid_name} - log.txt", "r") as f:
            log_data = f.read()
        sections = log_data.split("########################################################")
        for idx, section in enumerate(sections):
            if f"Pair {pair_num}" in section:
                lines = sections[idx + 1].split("\n")
                for line in lines:
                    if "Fscore:" in line:
                        f_score = line.split(": ")[-1]
                        f_scores[pair_num] = f_score
                        break
                break

        # Compute the dynamic range as a percentage of the cumulative sum
        with open(f"results/{vid_name}/KDE - Pair{pair_num} - H.pkl", "rb") as f:
            h_kde = pickle.load(f)
        with open(f"results/{vid_name}/KDE - Pair{pair_num} - NH.pkl", "rb") as f:
            nh_kde = pickle.load(f)

        cum_sum_h_score = list(accumulate(h_kde["y_coord"]))
        cum_sum_nh_score = list(accumulate(nh_kde["y_coord"]))

        RANGE_THRESHOLD = 0.90
        for idx, value in enumerate(cum_sum_h_score):
            if value > cum_sum_h_score[-1] * RANGE_THRESHOLD:
                h_range = [min(h_kde["x_coord"]), h_kde["x_coord"][idx]]
                break
        for idx, value in enumerate(cum_sum_nh_score):
            if value > cum_sum_nh_score[-1] * (1 - RANGE_THRESHOLD):
                nh_range = [nh_kde["x_coord"][idx], max(nh_kde["x_coord"])]
                break

        h_dyn_range = h_range[1] - h_range[0]
        nh_dyn_range = nh_range[1] - nh_range[0]
        range_ = (h_dyn_range + nh_dyn_range) / 2

        # Store results in a dictionary
        f_score = float(f_scores[pair_num].split("%")[0]) / 100
        intermediate_vars_dict[pair_num] = {"separation": separation, "range": range_, "auc": auc, "f_score": f_score}

    return intermediate_vars_dict


def filter_by_histogram(pairs_data):
    # Filter by separation, range, and AUC
    events_to_remove = [
        pair_num for pair_num, vars in pairs_data.items() if np.abs(vars["separation"]) < FILTER_SEPARATION or vars["range"] < FILTER_RANGE or vars["auc"] > FILTER_AUC
    ]

    return events_to_remove


def filter_by_area(sport, hist_div=2, num_bins=50):
    # Initialize dictionary to store the mean area of all pairs
    mean_areas = {}

    for pair_num in range(64):
        with open(f"results/{sport}/Mean area - Pair{pair_num}.pkl", "rb") as f:
            mean_area = pickle.load(f)
            mean_areas[pair_num] = mean_area

    # Compute histogram of the mean area values
    bins = np.linspace(0, np.max(list(mean_areas.values())), num_bins)
    hist_values, bin_values = np.histogram(list(mean_areas.values()), bins=bins)

    # Get the minimum and maximum area values in the histogram
    for hist_value, bin_value in zip(hist_values, bin_values):
        if hist_value > 0:
            min_area_value = bin_value
            break
    for hist_value, bin_value in zip(hist_values[::-1], bin_values[::-1]):
        if hist_value > 0:
            max_area_value = bin_value
            break

    area_filter = min_area_value + (max_area_value - min_area_value) / hist_div

    # Filter by area
    pairs_to_remove = [pair_num for pair_num, area in mean_areas.items() if area < area_filter]

    return pairs_to_remove, area_filter


def get_pairs(sport, pairs_data, hist_div=2):
    pairs_to_remove = set()

    # Filter by histogram
    pairs_filtered_by_histogram = filter_by_histogram(pairs_data)
    pairs_to_remove.update(pairs_filtered_by_histogram)

    # Filter by area
    pairs_filtered_by_area, area_filter = filter_by_area(sport, hist_div=hist_div)
    pairs_to_remove.update(pairs_filtered_by_area)

    # Return pairs
    return [pair_num for pair_num in range(64) if pair_num not in pairs_to_remove], area_filter


def compute_frame_level_metrics(predictions, ground_truth, steps=10):
    # Define the thresholds
    thresholds = range(0, 101, steps)  # 0%, 5%, 10%, 15%, ..., 100%

    # Order predictions by relevance (from highest to lowest)
    sorted_events = dict(sorted(predictions.items(), key=lambda x: x[1]["area"], reverse=True))
    total_events = len(sorted_events)

    # Iterate over the different thresholds
    for threshold in thresholds:
        # Retrieve the top events that pass the threshold
        num_events = int(round(total_events * threshold / 100, 0))
        events = dict(list(sorted_events.items())[:num_events])

        # Obtain frame level results
        recall, precision, fscore = computeFrameLevelResults(ground_truth=ground_truth, events_detected=events)
        print(f"{Color.YELLOW}{'':->10} Frame Level Results for threshold {threshold}% ({num_events} top events) {'':->10}")
        print(f"RECALL:    {recall*100:.2f}%")
        print(f"PRECISION: {precision*100:.2f}%")
        print(f"FSCORE:    {fscore*100:.2f}%{Color.RESET}\n")


def compute_event_level_metrics(predictions, ground_truth, steps=10):
    # Define the thresholds
    thresholds = range(0, 101, steps)  # 0%, 5%, 10%, 15%, ..., 100%

    # Order predictions by relevance (from highest to lowest)
    sorted_events = dict(sorted(predictions.items(), key=lambda x: x[1]["area"], reverse=True))
    num_pred_events = len(sorted_events)

    # Read the ground truth CSV file
    df_gt = pd.read_csv(ground_truth, header=0)

    # First of all, associate predicted events with ground truth events (in case one event multiple associations)
    # Extract ground truth highlight events
    gt_events = {}
    event_number = 0
    for _, row in df_gt.iterrows():
        if row["Event type"] == "Highlight":
            first_frame = int(row["First frame"])
            last_frame = int(row["Last frame"])

            gt_events[event_number] = np.arange(first_frame, last_frame + 1)
            event_number += 1
    num_gt_events = len(gt_events.keys())

    # Associate each prediction with the ground truth event that has the highest overlap
    for event_number, event_content in sorted_events.items():
        first_frame = event_content["frames"][0]
        last_frame = event_content["frames"][-1]

        event_content["associated_gt_events"] = []

        # Associate events which intersection between the prediction and each ground truth event is greather than 30 % of the ground truth event
        threshold = 0.3
        for gt_event_number, gt_event_frames in gt_events.items():
            intersection = len(set(event_content["frames"]).intersection(gt_event_frames))
            if intersection > len(gt_event_frames) * threshold:
                event_content["associated_gt_events"].append(gt_event_number)

    # Compute recall, precision, and f1-score metrics for different ranges of the top predictions
    for threshold in thresholds:
        # Retrieve the top events that pass the threshold
        num_events = int(round(num_pred_events * threshold / 100, 0))
        events = dict(list(sorted_events.items())[:num_events])

        tp = 0
        fp = 0
        gt_events_associated = []
        for event_number, event_content in list(events.items()):
            if event_content["associated_gt_events"]:
                for e in event_content["associated_gt_events"]:
                    if e not in gt_events_associated:
                        # Only count each ground truth event once as a TP
                        tp += 1
                        gt_events_associated.append(e)
            else:
                fp += 1

        recall = tp / num_gt_events
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        fscore = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        print(f"{Color.PURPLE}{'':->10} Event Level Results for threshold {threshold}% ({num_events} top events) {'':->10}")
        print(f"- TPs:       {tp}")
        print(f"- FPs:       {fp}")
        print(f"- GT events: {num_gt_events}")
        print(f"- Recall:    {recall*100:.2f}")
        print(f"- Precision: {precision*100:.2f}")
        print(f"- F1-score:  {fscore*100:.2f}")
        print(f"{Color.RESET}")


def export_highlight_reel(events_detected, video_name, fps=30, frame_root="data/imgs", frame_ext="png", out_filename="highlight.mp4"):
    # Make sure there are events detected
    if not events_detected:
        raise ValueError("events_detected is empty - nothing to export.")

    # Output path
    out_dir = os.path.join("results", video_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)

    # Determine frame size from the very first frame
    first_event = min(events_detected.values(), key=lambda e: e["frames"][0])
    sample_idx = first_event["frames"][0]
    sample_path = os.path.join(frame_root, video_name, f"frame{sample_idx:05d}.{frame_ext}")
    frame_sample = cv2.imread(sample_path)
    if frame_sample is None:
        raise FileNotFoundError(f"Cannot read sample frame: {sample_path}")
    height, width = frame_sample.shape[:2]

    # OpenCV writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # safe cross-platform codec
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Write frames in chronological order
    for event in sorted(events_detected.values(), key=lambda e: e["frames"][0]):
        for idx in event["frames"]:
            img_path = os.path.join(frame_root, video_name, f"frame{idx:05d}.{frame_ext}")
            frame = cv2.imread(img_path)
            if frame is not None:
                writer.write(frame)
    writer.release()

    return


if __name__ == "__main__":
    # Videos
    DATASET_DIR = "data"
    VIDEO = "long_jump"
    GROUND_TRUTH = f"{DATASET_DIR}/{VIDEO}.csv"

    # Constants
    CONTEXT_WINDOW = 600
    INSTANT_WINDOW = int(CONTEXT_WINDOW / 10)
    CLOSING_KERNEL = int(CONTEXT_WINDOW / 10 + 1)
    MIN_DURATION = 15
    MIN_AREA = "dynamic"  # "dynamic" or float (i.e., 15)

    # Filters
    FILTER_SEPARATION = 0.1
    FILTER_RANGE = 0.4
    FILTER_AUC = 0.4
    HIST_DIV = 2

    # Ablation
    NUM_STEPS = 10

    # Load pairs' data and compute intermediate variables
    pairs_data = compute_intermediate_vars(VIDEO)

    # Get pairs that pass the filters
    pairs, area_filter = get_pairs(VIDEO, pairs_data, HIST_DIV)
    min_area = area_filter if MIN_AREA == "dynamic" else MIN_AREA

    # Pretty print
    print(f"\n{'':->50} VIDEO: {VIDEO.upper()} {'':->50}")

    # Load highlight score predictions curve
    for pair in pairs_data.keys():
        with open(f"results/{VIDEO}/Scores - Pair{pair} - H.pkl", "rb") as file:
            pairs_data[pair]["h_scores"] = pickle.load(file)

    # Get ground truth and empty predictions
    ground_truth, _ = createGrounTruth(annotations_file=GROUND_TRUTH)

    # Convert ground truth to list
    ground_truth_list = groundTruth_Dict2List(ground_truth_dict=ground_truth, skip_uncertainty=False)

    # Compute median highlight score predictions
    median_predictions = np.median([pairs_data[pair]["h_scores"] for pair in pairs], axis=0).tolist()

    # Compute the rolling average for the predictions (instant & context)
    predictions_instant, predictions_context = computeRollingAverages(median_predictions, INSTANT_WINDOW, CONTEXT_WINDOW)

    # Compute coarse final predictions (those where instant predictions are above the context)
    coarse_final_predictions = [1 if pred_inst > pred_cont else 0 for pred_inst, pred_cont in zip(predictions_instant, predictions_context)]

    # Closing operation (dilate/erode) of the coarse final predictions
    refined_final_predictions = closingOperation(coarse_predictions=coarse_final_predictions, kernel_size=CLOSING_KERNEL)

    # Compute areas enclosed between the instant and context predictions
    areas = [max(pred_inst - pred_cont, 0) for pred_inst, pred_cont in zip(predictions_instant, predictions_context)]

    # Collect events
    events_detected = detectEvents(predictions=areas, masks=refined_final_predictions)
    print(f"{Color.RED}----------- Events detected -----------")
    print(list(events_detected.keys()))

    # Filter events by duration
    events_filtered_by_duration = filterEvents(events=events_detected, min_duration=MIN_DURATION, min_area=0, reorder_by_relevance=False)
    print(f"{Color.GREEN}----------- Events after filtering by duration -----------")
    print(f"{list(events_filtered_by_duration.keys())}{Color.RESET}")

    # Compute detected events' statistics
    mean_area = np.mean([d["area"] for d in events_filtered_by_duration.values()])
    std_area = np.std([d["area"] for d in events_filtered_by_duration.values()])

    # Get ablation metrics
    compute_frame_level_metrics(predictions=events_filtered_by_duration, ground_truth=ground_truth_list, steps=NUM_STEPS)
    compute_event_level_metrics(predictions=events_filtered_by_duration, ground_truth=GROUND_TRUTH, steps=NUM_STEPS)

    # Filter events by area
    events_filtered = filterEvents(events=events_filtered_by_duration, min_duration=0, min_area=min_area, reorder_by_relevance=False)
    print(f"{Color.GREEN}----------- Events after filtering by area -----------")
    print(f"{list(events_filtered.keys())}{Color.RESET}")

    # Obtain frame level results
    recall, precision, fscore = computeFrameLevelResults(ground_truth=ground_truth_list, events_detected=events_filtered)
    print_frame_level_results(recall, precision, fscore, color=Color.CYAN)

    # Plot predictions against the ground truth
    plotGroundTruthVSPredictions(
        frames_to_plot=[0, min(7500, list(ground_truth.keys())[-1])],
        ground_truth=ground_truth_list,
        predictions=median_predictions,
        predictions_instant=predictions_instant,
        predictions_context=predictions_context,
        coarse_final_predictions=coarse_final_predictions,
        refined_final_predictions=refined_final_predictions,
        areas=areas,
        events_filtered=events_filtered,
        fig_name=f"{VIDEO}/Final result.png",
        recall=recall,
        precision=precision,
        fscore=fscore,
        mean_area=mean_area,
        mean_std=std_area,
    )

    # Save pairs that were used
    with open(f"results/{VIDEO}/Pairs used.txt", "w") as file:
        file.write(", ".join(map(str, pairs)))

    # Save highlight reel
    export_highlight_reel(
        events_detected=events_filtered,
        video_name=VIDEO,
        fps=30,
        frame_root="data/imgs",
        frame_ext="png",
        out_filename="highlight.mp4",
    )
