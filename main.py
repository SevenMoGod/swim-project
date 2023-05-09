import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms/GSA"))

import copy

import numpy as np
import json
import yaml
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from algorithms.GSA import grounded_sam_demo as gsa
from algorithms.utils.vis import show_mask, show_box_and_label, save_mask_data
from utils.statistics import print_runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_config_file = config.get("model_config")
    grounded_checkpoint = config.get("grounded_checkpoint")
    sam_checkpoint = config.get("sam_checkpoint")

    input_video_name = config.get("input_video_name")
    output_dir = config.get("output_dir")
    stride = config.get("every_n_frames")

    text_prompt = config.get("text_prompt")
    box_threshold = config.get("box_threshold")
    text_threshold = config.get("text_threshold")

    device = config.get("device")

    # Make output dir
    os.makedirs(output_dir, exist_ok=True)

    # Read video
    video = cv2.VideoCapture(input_video_name)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video read success, fps: {fps}, frames: {frame_count}")

    # output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_name = os.path.join(output_dir, os.path.basename(input_video_name))
    out = cv2.VideoWriter(out_video_name, fourcc, fps, (width, height))

    # Load model
    p1_start_time = time.time()
    # print(grounded_checkpoint, sam_checkpoint)
    model = gsa.load_model(model_config_file, grounded_checkpoint, device=device)
    predictor = gsa.init_predictor(sam_checkpoint)
    p1_end_time = time.time()
    p1_time = p1_end_time - p1_start_time
    print("model loaded.")


    # Loop through video frames
    for idx in range(frame_count):
        success, frame = video.read()
        if not success:
            print("input error, check it again.")
            break

        print(f"frame {idx}")
        if idx % stride == 0:
            # load image
            p2_start_time = time.time()
            image_pil, image = gsa.load_image(frame)
            p2_end_time = time.time()
            p2_time = p2_end_time - p2_start_time

            # visualize raw image
            # cv2.imwrite(os.path.join(output_dir, f"raw_frame_{idx}.jpg"), frame)

            # run grounding dino model
            p3_start_time = time.time()
            boxes_filt, pred_phrases = gsa.get_grounding_output(model, image, text_prompt,
                                                             box_threshold, text_threshold,
                                                             device=device)
            p3_end_time = time.time()
            p3_time = p3_end_time - p3_start_time

            if boxes_filt.size()[0] == 0:
                out.write(frame)
                continue

            # init SAM
            p4_start_time = time.time()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
            # print("shape: ", H, W, boxes_filt.size(), transformed_boxes.size())
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            p4_end_time = time.time()
            p4_time = p4_end_time - p4_start_time

            # draw output image
            p5_start_time = time.time()
            for mask in masks:
                # print("now input image and mask:", image.shape, mask.cpu().numpy().shape)
                # save_mask_data(output_dir, mask.cpu().numpy())
                image = show_mask(image, mask.cpu().numpy(), random_color=False)

            for box, label in zip(boxes_filt, pred_phrases):
                # print("now input box: ", box.numpy().shape)
                image = show_box_and_label(image, box.numpy(), label)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(image)
            p5_end_time = time.time()
            p5_time = p5_end_time - p5_start_time
            # print_runtime(p5_end_time - p1_start_time, p1_time, p2_time, p3_time, p4_time, p5_time)
            # cv2.imwrite(os.path.join(output_dir, f"output_{idx}.jpg"), image)
            # gsa.save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
            # break

    video.release()
    out.release()
