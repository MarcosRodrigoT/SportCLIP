"""
Utility functions for highlight-detection experiments.

Key features
------------
* Build frame-level ground-truth maps from an annotation CSV.
* Encode text prompts with CLIP and score pre-computed frame embeddings.
* Convert predictions/ground truth to lists for analysis.
* Smooth scores, apply simple morphology, and group consecutive positives into “events”.
* Compute frame-level recall/precision/F-score, full ROC curves, and many threshold-swept minor metrics.
* Helper plots: timelines, ROC, per-threshold metrics.

Designed to be imported by higher-level scripts rather than run standalone.
"""


import os
import pandas as pd
import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.lines import Line2D
from scipy import ndimage


class Color:
    BLACK = "\x1b[30m"
    RED = "\033[31m"
    ORANGE = "\033[33m"
    GREEN = "\033[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\033[34m"
    MAGENTA = "\x1b[35m"
    PURPLE = "\033[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[97m"

    RESET = "\033[0m"


def createGrounTruth(annotations_file):
    ground_truth = {}
    predictions = {}
    annotations = pd.read_csv(annotations_file)

    for _, row in annotations.iterrows():
        for frame_num in range(row["First frame"], row["Last frame"] + 1):
            ground_truth[frame_num] = row["Event type"]
            predictions[frame_num] = None

    return ground_truth, predictions


def createClassEmbeddings(classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    classes_tokenized = clip.tokenize(classes).to(device)

    with torch.no_grad():
        class_embeddings = model.encode_text(classes_tokenized).cpu().numpy()

    return class_embeddings, model


def computeProbabilities(img_embeddings, class_embeddings, model):
    assert img_embeddings.shape == (1, 512), "Image embeddings should have shape (1, 512)"
    assert class_embeddings.shape[1] == 512, "Class embeddings should have shape (None, 512)"

    # Convert numpy to tensor
    img_embeddings = torch.from_numpy(img_embeddings).to("cuda")
    class_embeddings = torch.from_numpy(class_embeddings).to("cuda")

    # Normalized features
    img_embeddings = img_embeddings / img_embeddings.norm(dim=1, keepdim=True)
    class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)

    # Cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * img_embeddings @ class_embeddings.t()

    # Probabilities
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()

    return probs


def collectPredictions(root_dir, video_name, class_embeddings, model, predictions):
    img_embeddings_dir = os.path.join(root_dir, "img_embeddings", video_name)
    img_embeddings_files = [os.path.join(img_embeddings_dir, file) for file in sorted(os.listdir(img_embeddings_dir))]

    for img_embedding_file in img_embeddings_files:
        frame_num = int(img_embedding_file.split("/frame")[-1].split(".")[0])

        img_embeddings = np.load(img_embedding_file)

        probs = computeProbabilities(img_embeddings=img_embeddings, class_embeddings=class_embeddings, model=model)

        predictions[frame_num] = probs.astype("float")

    return predictions


def groundTruth_Dict2List(ground_truth_dict, skip_uncertainty=True):
    ground_truth_list = []

    for frame_num, val in ground_truth_dict.items():
        if val == "Uncertainty":
            if skip_uncertainty:
                continue
            else:
                ground_truth_list.append(0.5)
        elif val == "Not a highlight":
            ground_truth_list.append(0)
        elif val == "Highlight":
            ground_truth_list.append(1)

    return ground_truth_list


def predictions_Dict2List(predictions_dict):
    sentences_score_history = {x: [] for x in range(0, 2)}
    predictions_list = []

    for frame_num, val in predictions_dict.items():
        # Save the score history for each sentence to later draw their gaussians
        for sentence, score in enumerate(val[0]):
            sentences_score_history[sentence].append(score)

        # Obtain the H and NH sentences' scores
        h_pred = val[0][0]
        nh_pred = val[0][1]

        # Prediction is the positive prediction
        prediction = h_pred

        predictions_list.append(prediction)

    return predictions_list, sentences_score_history


def computeRollingAverages(predictions, instant_window, context_window):
    predictions_instant = [np.mean(predictions[max(0, i - int(instant_window / 2) - 1) : min(len(predictions) - 1, i + int(instant_window / 2))]) for i in range(len(predictions))]
    predictions_context = [np.mean(predictions[max(0, i - int(context_window / 2) - 1) : min(len(predictions) - 1, i + int(context_window / 2))]) for i in range(len(predictions))]

    return predictions_instant, predictions_context


def plotGroundTruthVSPredictions(
    ground_truth,
    predictions,
    fig_name,
    optimal_threshold=None,
    predictions_instant=None,
    predictions_context=None,
    optimal_threshold_instant=None,
):
    # Check if lists have the same length
    assert len(ground_truth) == len(predictions), "Lists must have the same length"

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot ground truth
    colors = {0: "red", 1: "green"}
    for frame_num, gt_value in enumerate(ground_truth):
        axs[0].axvline(frame_num, color=colors[gt_value], ymin=0, ymax=0.5)
    axs[0].set_ylabel("Ground Truth")
    axs[0].set_yticks([])
    legend_colors = [Line2D([0], [0], color="red"), Line2D([0], [0], color="blue"), Line2D([0], [0], color="green")]
    legend_names = ["Not a highlight", "Uncertainty", "Highlight"]
    axs[0].legend(legend_colors, legend_names, loc="upper right")

    # Plot predictions
    axs[1].plot(predictions, "r-", label="Predictions")
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("Predictions")
    axs[1].legend(loc="upper right")

    # Add instant and context rolling averages
    if predictions_instant is not None:
        axs[1].plot(predictions_instant, "b-", label="Instant")
        axs[1].legend(loc="upper right")

    if predictions_context is not None:
        axs[1].plot(predictions_context, "b--", label="Context")
        axs[1].legend(loc="upper right")

    # Add optimal threshold as a horizontal line
    if optimal_threshold is not None:
        axs[1].axhline(y=optimal_threshold, color="green", linestyle="-", label="Optimal threshold")
        axs[1].legend(loc="upper right")

    # Add optimal threshold for the instant predictions as a dashed horizontal line
    if optimal_threshold_instant is not None:
        axs[1].axhline(y=optimal_threshold_instant, color="green", linestyle="--", label="Optimal threshold instant")
        axs[1].legend(loc="upper right")

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"results/{fig_name}")


def plotGroundTruthVSPredictionsTFM(
    frames_to_plot,
    ground_truth,
    predictions,
    predictions_instant,
    predictions_context,
    coarse_final_predictions,
    refined_final_predictions,
    areas,
    events_filtered,
    fig_name,
    recall,
    precision,
    fscore,
    mean_area,
    mean_std,
):
    # Check if lists have the same length
    assert len(ground_truth) == len(predictions), "Lists must have the same length"

    # Limit the number of frames to plot if they surpass the length of the video
    if frames_to_plot[-1] > len(ground_truth):
        frames_to_plot[-1] = len(ground_truth)

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(7, 1, figsize=(8, 10), sharex=True)
    colors = {0: "red", 0.5: "blue", 1: "green"}

    # Plot ground truth
    for frame_num, gt_value in zip(range(frames_to_plot[0], frames_to_plot[-1]), ground_truth[frames_to_plot[0] : frames_to_plot[-1]]):
        axs[0].axvline(frame_num, color=colors[gt_value], ymin=0, ymax=1)
    axs[0].set_ylabel("Ground Truth")
    axs[0].set_yticks([])
    legend_colors = [Line2D([0], [0], color="red"), Line2D([0], [0], color="blue"), Line2D([0], [0], color="green")]
    legend_names = ["Not a highlight", "Uncertainty", "Highlight"]
    axs[0].legend(legend_colors, legend_names, loc="upper right")

    # Plot predictions
    axs[1].plot(
        list(range(frames_to_plot[0], frames_to_plot[-1])),
        predictions[frames_to_plot[0] : frames_to_plot[-1]],
        "r-",
        label="Predictions",
    )
    axs[1].set_ylabel("Predictions")
    axs[1].legend(loc="upper right")
    axs[1].set_ylim(0, 1)

    # Plot instant and context rolling averages
    axs[1].plot(
        list(range(frames_to_plot[0], frames_to_plot[-1])),
        predictions_instant[frames_to_plot[0] : frames_to_plot[-1]],
        "b-",
        label="Instant",
    )
    axs[1].legend(loc="upper right")

    axs[1].plot(
        list(range(frames_to_plot[0], frames_to_plot[-1])),
        predictions_context[frames_to_plot[0] : frames_to_plot[-1]],
        "b--",
        label="Context",
    )
    axs[1].legend(loc="upper right")

    # Plot coarse final predictions
    for frame_num, coarse_pred in zip(range(frames_to_plot[0], frames_to_plot[-1]), coarse_final_predictions[frames_to_plot[0] : frames_to_plot[-1]]):
        axs[2].axvline(frame_num, color=colors[coarse_pred], ymin=0, ymax=1)
    axs[2].set_ylabel("Coarse Preds")
    axs[2].set_yticks([])
    legend_colors = [Line2D([0], [0], color="red"), Line2D([0], [0], color="green")]
    legend_names = ["Not a highlight", "Highlight"]
    axs[2].legend(legend_colors, legend_names, loc="upper right")

    # Plot refined final predictions
    for frame_num, refine_pred in zip(range(frames_to_plot[0], frames_to_plot[-1]), refined_final_predictions[frames_to_plot[0] : frames_to_plot[-1]]):
        axs[3].axvline(frame_num, color=colors[refine_pred], ymin=0, ymax=1)
    axs[3].set_ylabel("Refined Preds")
    axs[3].set_yticks([])
    legend_colors = [Line2D([0], [0], color="red"), Line2D([0], [0], color="green")]
    legend_names = ["Not a highlight", "Highlight"]
    axs[3].legend(legend_colors, legend_names, loc="upper right")

    # Plot areas
    for frame_num, area_value in zip(range(frames_to_plot[0], frames_to_plot[-1]), areas[frames_to_plot[0] : frames_to_plot[-1]]):
        axs[4].axvline(frame_num, color="blue", ymin=0, ymax=(area_value / max(areas)))
    axs[4].set_ylabel("Area")
    legend_colors = [Line2D([0], [0], color="blue")]
    legend_names = ["Enclosed area"]
    axs[4].legend(legend_colors, legend_names, loc="upper right")

    # Plot refined predictions modeled by areas
    for frame_num, (area_value, refine_pred) in zip(
        range(frames_to_plot[0], frames_to_plot[-1]),
        zip(areas[frames_to_plot[0] : frames_to_plot[-1]], refined_final_predictions[frames_to_plot[0] : frames_to_plot[-1]]),
    ):
        ymax = (area_value / max(areas)) if refine_pred else 1
        axs[5].axvline(frame_num, color=colors[refine_pred], ymin=0, ymax=ymax)
    axs[5].set_ylabel("Modeled Preds")
    legend_colors = [Line2D([0], [0], color="red"), Line2D([0], [0], color="green")]
    legend_names = ["Not a highlight", "Highlight"]
    axs[5].legend(legend_colors, legend_names, loc="upper right")

    # Plot final filtered events
    color = "red"  # default in case all events are filtered out
    ymax = 1  # default in case all events are filtered out
    for frame_num in range(frames_to_plot[0], frames_to_plot[-1]):
        for event_content in events_filtered.values():
            if frame_num in event_content["frames"]:
                color = "green"
                ymax = event_content["areas"][event_content["frames"].index(frame_num)]
                break
            else:
                color = "red"
                ymax = 1

        axs[6].axvline(frame_num, color=color, ymin=0, ymax=ymax)
        axs[6].set_ylabel("Final Preds")
        axs[6].set_xlabel("Frame")
    legend_colors = [Line2D([0], [0], color="red"), Line2D([0], [0], color="green")]
    legend_names = ["Not a highlight", "Highlight"]
    axs[6].legend(legend_colors, legend_names, loc="upper right")

    # Adjust layout to prevent clipping of labels
    fig.suptitle(f"Recall: {recall*100:.2f}% - Precision: {precision*100:.2f}% - F-score: {fscore*100:.2f}%\nMean area: {mean_area:.2f} - Mean std: {mean_std:.2f}")
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"results/{fig_name}")
    plt.close()


def stitch_images(pair_num, highlight_idx, non_highlight_idx, results_folder, video_name):
    # Open the images
    image1 = Image.open(f"{results_folder}/{video_name} - histogram - H - tmp.png")
    image2 = Image.open(f"{results_folder}/{video_name} - histogram - NH - tmp.png")
    image3 = Image.open(f"{results_folder}/{video_name} - binary - tmp.png")

    # Create a new image with the dimensions required
    final_image = Image.new("RGB", (image1.width + image3.width, image3.height))

    # Paste the first two images onto the left side of the final image
    final_image.paste(image1, (0, 0))
    final_image.paste(image2, (0, image1.height))

    # Paste the resized third image onto the right side of the final image
    final_image.paste(image3, (image1.width, 0))

    # Save the final image
    final_image.save(f"{results_folder}/Pair{pair_num} - H{highlight_idx} NH{non_highlight_idx}.png")

    # Remove temporal images
    os.remove(f"{results_folder}/{video_name} - histogram - H - tmp.png")
    os.remove(f"{results_folder}/{video_name} - histogram - NH - tmp.png")
    os.remove(f"{results_folder}/{video_name} - binary - tmp.png")


def closingOperation(coarse_predictions, kernel_size=61):
    kernel = np.ones(kernel_size, dtype=bool)
    dilation = np.convolve([bool(coarse_pred) for coarse_pred in coarse_predictions], kernel, mode="same")
    erosion = ndimage.binary_erosion(dilation, structure=kernel).astype(int).tolist()

    return erosion


def compute_crossing_point(curve1, curve2):
    # Extract x and y coordinates from the dictionaries
    x1 = np.array(curve1["x_coord"])
    y1 = np.array(curve1["y_coord"])
    x2 = np.array(curve2["x_coord"])
    y2 = np.array(curve2["y_coord"])

    # Calculate distances between each pair of points from the two curves
    distances = np.sqrt((x1[:, np.newaxis] - x2) ** 2 + (y1[:, np.newaxis] - y2) ** 2)

    # Find the minimum distance and corresponding indices
    min_index = np.unravel_index(np.argmin(distances), distances.shape)

    # Check if the curves do not cross (i.e., minimum distance is too large)
    if distances[min_index] > 0.1:  # You can adjust this threshold as needed
        raise ValueError("These 2 curves do not cross.")

    # Return the coordinates of the crossing point
    crossing_x = x1[min_index[0]]
    crossing_y = y1[min_index[0]]

    return crossing_x, crossing_y


def detectEvents(predictions, masks):
    events = {}
    frames = []
    areas = []

    event_number = 0
    prev_active = False

    for idx, (pred, mask) in enumerate(zip(predictions, masks)):
        if mask:
            frames.append(idx)
            areas.append(pred)
            prev_active = True
        else:
            if prev_active:
                prev_active = False
                # events[event_number] = [frames, len(frames), sum(areas)]
                events[event_number] = {"frames": frames, "areas": areas, "length": len(frames), "area": sum(areas)}

                frames = []
                areas = []
                event_number += 1

    # In case the video ends with a highlight (although very rare)
    if prev_active:
        # events[event_number] = [frames, len(frames), sum(areas)]
        events[event_number] = {"frames": frames, "areas": areas, "length": len(frames), "area": sum(areas)}

    return events


def filterEvents(events, min_duration, min_area, reorder_by_relevance=True):
    events_to_delete = [event_number for event_number in events if events[event_number]["length"] < min_duration or events[event_number]["area"] < min_area]

    for event_to_delete in events_to_delete:
        events.pop(event_to_delete)

    # Reorder events, from highest to lowest area
    if reorder_by_relevance:
        events = OrderedDict(sorted(events.items(), key=lambda item: item[1]["area"], reverse=True))

    return events


def computeFrameLevelResults(ground_truth, events_detected):
    # Initialize variables
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tu = 0
    fu = 0

    # Convert detected events to a list
    events_frames = []
    for event_number, event_content in events_detected.items():
        events_frames.extend(event_content["frames"])

    events_list = [0 for _ in range(0, len(ground_truth))]
    for event_frame in events_frames:
        events_list[event_frame] = 1

    # Get minor metrics
    for gt_element, event_element in zip(ground_truth, events_list):
        # If ground truth is "H"
        if gt_element == 1:
            if event_element == 1:
                tp += 1
            elif event_element == 0:
                fn += 1
        # If ground truth is "NH"
        elif gt_element == 0:
            if event_element == 1:
                fp += 1
            elif event_element == 0:
                tn += 1
        # If ground truth is "U"
        elif gt_element == 0.5:
            if event_element == 1:
                tu += 1
            elif event_element == 0:
                fu += 1

    # Convert uncertainty to whatever increases results
    tp += tu
    tn += fu

    recall = tp / (tp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    fscore = (2 * tp) / (2 * tp + fp + fn + 1e-7)

    return recall, precision, fscore


def computeMinorMetrics(
    ground_truth,
    predictions,
    thresholds,
    optimal_threshold,
    fig_name="",
    predictions_instant=None,
    thresholds_instant=None,
    optimal_threshold_instant=None,
):
    # Auxiliary function to compute the minor metrics for a specific threshold
    def auxialiaryCalculations(ground_truth, predictions):
        tp_ = 0
        tn_ = 0
        fp_ = 0
        fn_ = 0
        for gt, pred in zip(ground_truth, predictions):
            if gt == 1 and pred == 1:
                tp_ += 1
            elif gt == 0 and pred == 0:
                tn_ += 1
            elif gt == 0 and pred == 1:
                fp_ += 1
            elif gt == 1 and pred == 0:
                fn_ += 1
        recall_ = tp_ / (tp_ + fn_ + 1e-7)
        precision_ = tp_ / (tp_ + fp_ + 1e-7)
        fscore_ = (2 * tp_) / (2 * tp_ + fp_ + fn_ + 1e-7)

        return tp_, tn_, fp_, fn_, recall_, precision_, fscore_

    # Initialize minor metrics
    tp = []
    tn = []
    fp = []
    fn = []
    recall = []
    precision = []
    fscore = []

    # Compute minor metrics for each threshold
    for threshold in thresholds:
        predictions_thresholded = np.where(predictions > threshold, 1, 0)
        tp_, tn_, fp_, fn_, recall_, precision_, fscore_ = auxialiaryCalculations(ground_truth=ground_truth, predictions=predictions_thresholded)

        tp.append(tp_)
        tn.append(tn_)
        fp.append(fp_)
        fn.append(fn_)
        recall.append(recall_)
        precision.append(precision_)
        fscore.append(fscore_)

        # Store the metrics corresponding to the optimal point
        if threshold == optimal_threshold:
            tp_eer, tn_eer, fp_eer, fn_eer, recall_eer, precision_eer, fscore_eer = (
                tp_,
                tn_,
                fp_,
                fn_,
                recall_,
                precision_,
                fscore_,
            )

    # Configure minor metrics for the plots
    tps = ["True Positives (TP)", "TP", tp, tp_eer]
    tns = ["True Negatives (TN)", "TN", tn, tn_eer]
    fps = ["False Positives (FP)", "FP", fp, fp_eer]
    fns = ["False Negatives (FN)", "FN", fn, fn_eer]
    recalls = ["Recall", "Recall", recall, recall_eer]
    precisions = ["Precision", "Precision", precision, precision_eer]
    fscores = ["F-Score", "F-Score", fscore, fscore_eer]

    if predictions_instant is not None:
        # Initialize minor metrics
        tp_instant = []
        tn_instant = []
        fp_instant = []
        fn_instant = []
        recall_instant = []
        precision_instant = []
        fscore_instant = []

        # Compute minor metrics for each threshold
        for threshold_instant in thresholds_instant:
            predictions_instant_thresholded = np.where(predictions_instant > threshold_instant, 1, 0)
            (
                tp_instant_,
                tn_instant_,
                fp_instant_,
                fn_instant_,
                recall_instant_,
                precision_instant_,
                fscore_instant_,
            ) = auxialiaryCalculations(ground_truth=ground_truth, predictions=predictions_instant_thresholded)

            tp_instant.append(tp_instant_)
            tn_instant.append(tn_instant_)
            fp_instant.append(fp_instant_)
            fn_instant.append(fn_instant_)
            recall_instant.append(recall_instant_)
            precision_instant.append(precision_instant_)
            fscore_instant.append(fscore_instant_)

            # Store the metrics corresponding to the optimal point
            if threshold_instant == optimal_threshold_instant:
                (
                    tp_instant_eer,
                    tn_instant_eer,
                    fp_instant_eer,
                    fn_instant_eer,
                    recall_instant_eer,
                    precision_instant_eer,
                    fscore_instant_eer,
                ) = (
                    tp_instant_,
                    tn_instant_,
                    fp_instant_,
                    fn_instant_,
                    recall_instant_,
                    precision_instant_,
                    fscore_instant_,
                )

        # Configure minor metrics for the plots
        tps_instant = ["True Positives (TP)", "TP instant", tp_instant, tp_instant_eer]
        tns_instant = ["True Negatives (TN)", "TN instant", tn_instant, tn_instant_eer]
        fps_instant = ["False Positives (FP)", "FP instant", fp_instant, fp_instant_eer]
        fns_instant = ["False Negatives (FN)", "FN instant", fn_instant, fn_instant_eer]
        recalls_instant = ["Recall", "Recall instant", recall_instant, recall_instant_eer]
        precisions_instant = ["Precision", "Precision instant", precision_instant, precision_instant_eer]
        fscores_instant = ["F-Score", "F-Score instant", fscore_instant, fscore_instant_eer]

        metrics_instant = [
            tps_instant,
            tns_instant,
            fps_instant,
            fns_instant,
            recalls_instant,
            precisions_instant,
            fscores_instant,
        ]

    # Plot each minor metric
    for idx, metric in enumerate([tps, tns, fps, fns, recalls, precisions, fscores]):
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        # Metrics from the predictions
        axs.plot(thresholds, metric[-2], color="blue", label=f"{metric[1]}={metric[-1]:.3f}")
        axs.scatter(optimal_threshold, metric[-1], color="blue")
        axs.legend()
        # Metrics from the averaged predictions
        axs.plot(
            thresholds_instant,
            metrics_instant[idx][-2],
            color="blue",
            linestyle="--",
            label=f"{metrics_instant[idx][1]}={metrics_instant[idx][-1]:.3f}",
        )
        axs.scatter(optimal_threshold_instant, metrics_instant[idx][-1], color="blue")
        axs.legend()

        axs.set_title(metric[0])
        axs.set_xlabel("Thresholds")
        axs.set_ylabel(metric[1])

        # Save the plot
        plt.savefig(f"results/{metric[0]}{fig_name}")
