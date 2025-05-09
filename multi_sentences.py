"""
Runs a “sentence-vs-sentence” experiment to see which CLIP text prompt best separates *highlight* moments from *non-highlight* moments in one video.

High-level flow
---------------
1. **Config** - pick a video (e.g. “V1”) and two lists of prompts: `highlight_sentences` (H) and `not_highlight_sentences` (NH).

2. **For every H-to-NH pair**  
   • Embed the two sentences with CLIP.  
   • For every frame, score similarity between the frame embedding (pre-computed elsewhere) and each sentence.  
   • Collapse the two scores into a single “highlight probability” (max/avg/binary).  
   • Smooth scores with short/long rolling windows and keep spikes where the short window rises above the long one.  
   • Post-process with a morphological closing and turn the result into *events* (contiguous highlight segments).

3. **Evaluation & visualisation**  
   • Compare detected events against ground-truth CSV, print recall / precision / F-score.  
   • Plot timelines, score histograms, KDEs, save stitched PNGs.  
   • Compute many distance metrics (Bhattacharyya, JS, KL, etc.) between the H and NH score distributions and save them to disk.

4. **Outputs** land in `results/` as logs, figures, pickled statistics and metric files - one set per sentence pair.

Run with `python multi_sentences.py` after editing the `Config` class.  All heavy lifting lives in `utils.py`; this script is mainly orchestration and reporting.
"""


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


class Config:
    # Main parameters
    root_dir = "data"
    video_name = "long_jump"
    results_folder = f"results/{video_name}"
    os.makedirs(results_folder, exist_ok=True)

    context_window = 500
    instant_window = int(context_window / 10)
    closing_kernel = int(context_window / 10 + 1)
    min_duration = 15
    min_area = 15

    """ Highlight and non-highlight sentences """
    # Tricking
    # highlight_sentences = [
    #     "A highlight",
    #     "A person doing something interesting",
    #     "Someone practicing dynamic movements",
    #     "Someone is performing martial arts tricking",
    #     "A skilled athlete executing kicks and spins",
    #     "A person is performing some type of acrobatics",
    #     "A person is clearly performing some type of acrobatics in a gym",
    #     "Tricker performing a pass including flips, transitions and kicks",
    # ]
    # not_highlight_sentences = [
    #     "Not a highlight",
    #     "A gym with people present",
    #     "A group of people chatting and relaxing",
    #     "Athletes preparing for their practice session",
    #     "A group of people in a gym, ready to perform acrobatics but not at the moment",
    #     "No one is currently performing acrobatics, people enjoy a moment of relaxed interaction",
    #     "Trickers form a circle, chatting and enjoying a break, with no one currently performing acrobatics",
    #     "Trickers relaxed in a gym, talking, walking, waiting for their turn to perform flips and tricks",
    # ]

    # Diving
    # highlight_sentences = [
    #     "A diver launching into the air, executing elegant flips and twists before entering the water seamlessly",
    #     "A person performing an intricate diving maneuver, maintaining perfect control and posture",
    #     "A high-speed, precise dive with controlled rotation and a smooth water entry",
    #     "A diver executing a complex aerial twist, demonstrating agility and technical skill",
    #     "A person in mid-air, rotating dynamically before breaking the water surface with minimal splash",
    #     "A skilled athlete performing a synchronized series of rotations and flips before entering the water",
    #     "A diver showcasing advanced aerial control, twisting and flipping effortlessly during descent",
    #     "Diver executing precise, high-speed flips and twists with controlled power while in the air, during the dive",
    # ]
    # not_highlight_sentences = [
    #     "A diver standing on the platform, preparing for their turn",
    #     "An athlete adjusting their stance and breathing before initiating the dive",
    #     "A group of divers discussing techniques near the pool",
    #     "A coach providing feedback while the diver listens attentively",
    #     "A person climbing the ladder to get to the diving board",
    #     "An athlete walking along the pool deck, stretching and preparing",
    #     "A diver surfacing after the dive, calmly swimming towards the pool's edge",
    #     "Diver relaxed, getting prepared to perform, close to the edge of the platform, greeting judges, walking, swimming, diving below the water",
    # ]

    # Long jump
    highlight_sentences = [
        "An athlete sprinting down the runway before launching into the air, reaching for maximum distance",
        "A long jumper executing a well-timed takeoff, soaring through the air before landing in the sand pit",
        "A person accelerating down the track, generating momentum for an explosive jump",
        "An athlete gliding through the air with extended arms and legs, preparing for a controlled landing",
        "A competitor demonstrating strength and precision in a long jump attempt",
        "A long jumper executing a perfect flight phase, reaching their peak height before descent",
        "An athlete pushing off the ground with powerful force, achieving an impressive airborne moment",
        "Athlete running, jumping into the air and landing in the sand pit",
    ]
    not_highlight_sentences = [
        "A long jumper adjusting their starting position on the runway",
        "A person discussing jump techniques with a coach",
        "An athlete waiting for their turn while observing competitors",
        "A group of athletes standing near the sand pit, preparing for their jumps",
        "A long jumper walking back after a completed attempt",
        "A judge measuring the distance of a jump while athletes watch",
        "A competitor stretching and warming up before their jump",
        "Athlete relaxed, greeting judges, celebrating",
    ]

    # Pole vault
    # highlight_sentences = [
    #     "A pole vaulter sprinting down the track, planting the pole, and propelling into the air",
    #     "An athlete executing a flawless pole vault, clearing the bar with perfect body control",
    #     "A vaulter swinging upward, rotating their body to clear the bar with precision",
    #     "A person vaulting high into the air, momentarily suspended before landing safely on the mat",
    #     "An athlete demonstrating strength and technique as they push off the pole, reaching great heights",
    #     "A pole vaulter soaring over the bar, arching their back for a clean clearance",
    #     "A competitor successfully clearing the bar, landing smoothly with an impressive technique",
    #     "Pole vaulter running, vaulting over the bar, jumping into the air and landing in a mat",
    # ]
    # not_highlight_sentences = [
    #     "A pole vaulter adjusting their grip on the pole before an attempt",
    #     "An athlete waiting near the track, watching competitors perform",
    #     "A coach giving last-minute instructions to a vaulter",
    #     "A competitor retrieving their pole and preparing for their next vault",
    #     "A person checking the pole's flexibility before beginning their approach",
    #     "An athlete sitting on the mat, reflecting on their previous attempt",
    #     "A pole vault official adjusting the bar height for the next attempt",
    #     "Pole vaulter relaxed, greeting judges, adjusting the pole, celebrating",
    # ]

    # Tumbling
    # highlight_sentences = [
    #     "An athlete executing a fast-paced series of flips and twists along the tumbling track",
    #     "A competitor demonstrating explosive power, transitioning seamlessly between acrobatic moves",
    #     "A person performing high-speed aerial rotations with perfect body control",
    #     "An athlete showcasing a dynamic tumbling pass, blending flips and twists effortlessly",
    #     "A tumbler reaching incredible heights during a complex flipping sequence",
    #     "A competitor executing multiple backflips in succession with remarkable precision",
    #     "An athlete performing an elegant combination of twisting and flipping movements at high velocity",
    #     "Athlete performing powerful, high-speed flips and twists with explosive energy along a straight tumbling track",
    # ]
    # not_highlight_sentences = [
    #     "A tumbler standing at the start of the track, preparing for their run",
    #     "A coach giving feedback to an athlete after their routine",
    #     "A competitor stretching before their turn on the tumbling track",
    #     "A group of athletes chatting and resting between routines",
    #     "A tumbler walking off the mat after completing a pass",
    #     "An athlete adjusting their uniform before stepping onto the track",
    #     "A competitor watching others perform while waiting for their turn",
    #     "Athlete relaxed, getting prepared to perform, greeting judges",
    # ]

    sentences = highlight_sentences + not_highlight_sentences

    # Plotting parameters
    hist_sharey = True  # Share y axis among histplots when drawing multiple in a single figure
    hist_scale_y = True  # "True" -> maximum y-axis set dinamycally. "False" -> set to the number of video frames
    draw_individual_plots = True
    frames_to_plot = [0, 7500]


def main():
    # Get ground truth and empty predictions
    ground_truth, predictions = createGrounTruth(annotations_file=os.path.join(Config.root_dir, f"{Config.video_name}.csv"))

    # Loop over the different sentences forming pairs
    pair_num = 0
    for h_sent_idx, h_sentence in enumerate(Config.highlight_sentences):
        for nh_sent_idx, nh_sentence in enumerate(Config.not_highlight_sentences):
            print(f"{Color.ORANGE}----------- Highlight sentence {h_sent_idx}:{' ': >6} -----------{Color.RESET}\n{h_sentence}")
            print(f"{Color.ORANGE}----------- Not a highlight sentence {nh_sent_idx}: -----------{Color.RESET}\n{nh_sentence}")

            # Get class (text) embeddings and the clip model
            classes = [h_sentence, nh_sentence]
            class_embeddings, clip_model = createClassEmbeddings(classes=classes)

            # Get predictions
            predictions = collectPredictions(
                root_dir=Config.root_dir,
                video_name=Config.video_name,
                class_embeddings=class_embeddings,
                model=clip_model,
                predictions=predictions,
            )

            # Convert ground truth and predictions to lists
            ground_truth_list = groundTruth_Dict2List(ground_truth_dict=ground_truth, skip_uncertainty=False)
            predictions_list, sentences_score_hist = predictions_Dict2List(predictions_dict=predictions)

            # Compute the rolling average for the predictions (instant & context)
            predictions_instant, predictions_context = computeRollingAverages(predictions_list, Config.instant_window, Config.context_window)

            # Compute coarse final predictions (those where instant predictions are above the context)
            coarse_final_predictions = [1 if pred_inst > pred_cont else 0 for pred_inst, pred_cont in zip(predictions_instant, predictions_context)]

            # Closing operation (dilate/erode) of the coarse final predictions
            refined_final_predictions = closingOperation(coarse_predictions=coarse_final_predictions, kernel_size=Config.closing_kernel)

            # Compute areas enclosed between the instant and context predictions
            areas = [max(pred_inst - pred_cont, 0) for pred_inst, pred_cont in zip(predictions_instant, predictions_context)]

            # Collect events
            events_detected = detectEvents(predictions=areas, masks=refined_final_predictions)
            print_events(events_detected, color=Color.RED, message="Events detected")

            # Filter events by duration
            events_filtered_by_duration = filterEvents(events=events_detected, min_duration=Config.min_duration, min_area=0, reorder_by_relevance=False)
            print_events(events_detected, color=Color.GREEN, message="Events after filtering by duration")

            # Compute detected events' statistics
            mean_area = np.mean([d["area"] for d in events_filtered_by_duration.values()])
            std_area = np.std([d["area"] for d in events_filtered_by_duration.values()])

            # Save the mean event area
            with open(f"{Config.results_folder}/Mean area - Pair{pair_num}.pkl", "wb") as f:
                pickle.dump(mean_area, f)

            # Filter events by area
            events_filtered = filterEvents(events=events_filtered_by_duration, min_duration=0, min_area=Config.min_area, reorder_by_relevance=False)
            print_events(events_detected, color=Color.GREEN, message="Events after filtering by area")

            # Obtain frame level results
            recall, precision, fscore = computeFrameLevelResults(ground_truth=ground_truth_list, events_detected=events_filtered)
            print_frame_level_results(recall, precision, fscore, color=Color.CYAN)

            # Plot predictions against the ground truth
            if Config.draw_individual_plots:
                plotGroundTruthVSPredictions(
                    frames_to_plot=Config.frames_to_plot,
                    ground_truth=ground_truth_list,
                    predictions=predictions_list,
                    predictions_instant=predictions_instant,
                    predictions_context=predictions_context,
                    coarse_final_predictions=coarse_final_predictions,
                    refined_final_predictions=refined_final_predictions,
                    areas=areas,
                    events_filtered=events_filtered,
                    fig_name=f"{'/'.join(Config.results_folder.split('/')[1:])}/{Config.video_name} - binary - tmp.png",
                    recall=recall,
                    precision=precision,
                    fscore=fscore,
                    mean_area=mean_area,
                    mean_std=std_area,
                )

            # Draw histograms as individual distribution plots with KDE
            draw_histograms(sentences_score_hist, pair_num, Config.hist_scale_y, Config.results_folder, Config.video_name)

            # Compute the crossing point between the two distributions
            compute_crossing_point(pair_num, Config.results_folder)

            # Stitch the histograms and predictions together
            stitch_images(pair_num=pair_num, highlight_idx=h_sent_idx, non_highlight_idx=nh_sent_idx, results_folder=Config.results_folder, video_name=Config.video_name)

            # Save a log of the experiments
            save_logs(pair_num, h_sentence, nh_sentence, recall, precision, fscore, sentences_score_hist, Config.results_folder, Config.video_name)

            # Move on to the next pair
            pair_num += 1


if __name__ == "__main__":
    main()
