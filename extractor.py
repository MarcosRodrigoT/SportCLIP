"""
Extracts evenly-spaced frames from several hard-coded sports videos, runs each frame through CLIP (ViT-B/32) to get a 512-D image vector, and saves:

    imgs/<video>/frameXXXXX.png
    img_embeddings/<video>/frameXXXXX.npy

Use --dataset-dir to point at the folder containing the videos/CSVs and --frames-per-second to control how densely frames are sampled.
"""


import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image


def get_fps(video_name):
    command = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {video_name}"
    fps = subprocess.check_output(command.split()).decode("utf-8").strip().split("/")
    return float(fps[0]) / float(fps[1])


def extract_frame(video_name, frame_idx, frame_name):
    command = f"ffmpeg -n -i {video_name} -vf select='eq(n\,{frame_idx})' -vframes 1 -vsync 0 {frame_name}"
    subprocess.call(command.split())


def get_clip_embeddings(device, model, preprocess, frame_name, img_embedding_name):
    image = preprocess(Image.open(frame_name)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()
        np.save(img_embedding_name, image_features)


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for video_file, ann_file in [
        ("V1.mpeg", "V1.csv"),
        ("V2.mpeg", "V2.csv"),
        ("V3.mpeg", "V3.csv"),
        ("diving.mp4", "diving.csv"),
        ("long_jump.mp4", "long_jump.csv"),
        ("pole_vault.mp4", "pole_vault.csv"),
        ("tumbling.mp4", "tumbling.csv"),
    ]:
        video_name = video_file.split(".")[0]
        video_file = os.path.join(config.dataset_dir, video_file)
        ann_file = os.path.join(config.dataset_dir, ann_file)

        annotations = pd.read_csv(ann_file)
        last_frame = annotations.iloc[-1]["Last frame"]

        imgs_dir = os.path.join(config.dataset_dir, "imgs", video_name)
        img_embs_dir = os.path.join(config.dataset_dir, "img_embeddings", video_name)
        video_framerate = get_fps(video_file)
        for frame_idx in range(0, last_frame + 1, round(video_framerate / config.frames_per_second)):
            frame_name = os.path.join(imgs_dir, f"frame{frame_idx:0>5}.png")
            img_embedding_name = os.path.join(img_embs_dir, f"frame{frame_idx:0>5}.npy")

            extract_frame(
                video_name=video_file,
                frame_idx=frame_idx,
                frame_name=frame_name,
            )

            get_clip_embeddings(
                device=device,
                model=clip_model,
                preprocess=preprocess,
                frame_name=frame_name,
                img_embedding_name=img_embedding_name,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="data", help="Dataset directory")
    parser.add_argument("--frames-per-second", type=float, default=29.97, help="Number of frames extracted per second")

    config = parser.parse_args()
    main(config)
