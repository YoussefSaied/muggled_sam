#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% imports
# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

from collections import defaultdict
import shutil
import cv2
import numpy as np
import torch
from lib.make_sam_v2 import make_samv2_from_original_state_dict
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults
from oscar.utils.data_utils import SimpleImagesDS
import os


# %% Define pathing & device usage
game = "Boxing-v4"
run = "run_30_01_17_32"
episode = "episode-0"
DATADIR = "/home/saied/atari_data"
raw_img_dir = f"{DATADIR}/{game}/{run}/raw"
video_path = os.path.join(raw_img_dir, episode)
SAM2DIR = "/home/saied/oscar/segment-anything-2"
model_path = f"{SAM2DIR}/checkpoints/sam2_hiera_tiny.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define image processing config (shared for all video frames)
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

# %% For demo purposes, we'll define all prompts ahead of time and store them per frame index & object
# -> First level key (e.g. 0, 30, 35) represents the frame index where the prompts should be applied
# -> Second level key (e.g. 'obj1', 'obj2') represents which 'object' the prompt belongs to for tracking purposes
obj_coords = {
    0: [3, 104],
    1: [49, 8],
    2: [77, 193],
    3: [117, 145],
    4: [43, 64],
    5: [113, 8],
    6: [65, 20],
    7: [81, 20],
    8: [89, 20],
}
H, W = 160, 210
obj_coords_normalized = {k: (v[0] / H, v[1] / W) for k, v in obj_coords.items()}

# Convert obj_coords_normalized to the same format as prompts_per_frame_index
obj_prompts_frame_0 = {
    "obj"
    + str(k): {
        "box_tlbr_norm_list": [],
        "fg_xy_norm_list": [v],
        "bg_xy_norm_list": [],
    }
    for k, v in obj_coords_normalized.items()
}
prompts_per_frame_index = {0: obj_prompts_frame_0}


# %%
enable_prompt_visualization = True
# *** These prompts are set up for a video of horses available from pexels.com ***
# https://www.pexels.com/video/horses-running-on-grassland-4215784/
# By: Adrian Hoparda

# Set up memory storage for tracked objects
# -> Assumes each object is represented by a unique dictionary key (e.g. 'obj1')
# -> This holds both the 'prompt' & 'recent' memory data needed for tracking!
memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)

# Read first frame to check that we can read from the video, then reset playback
image_ds = SimpleImagesDS(video_path)

# Set up model
print("Loading model...")
model_config_dict, sammodel = make_samv2_from_original_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)

# Process video frames
close_keycodes = {27, ord("q")}  # Esc or q to close
# Create directory for saving visualizations
visualization_dir = "tmp_visualisation"
if os.path.exists(visualization_dir):
    for filename in os.listdir(visualization_dir):
        file_path = os.path.join(visualization_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
os.makedirs(visualization_dir)

try:
    total_frames = len(image_ds)
    for frame_idx in range(total_frames):

        # Read frames
        frame = image_ds[frame_idx]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        # Encode frame data (shared for all objects)
        encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)

        # Generate & store prompt memory encodings for each object as needed
        prompts_dict = prompts_per_frame_index.get(frame_idx, None)
        if prompts_dict is not None:

            # Loop over all sets of prompts for the current frame
            for obj_key_name, obj_prompts in prompts_dict.items():
                print(
                    f"Generating prompt for object: {obj_key_name} (frame {frame_idx})"
                )
                init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(
                    encoded_imgs_list, **obj_prompts
                )
                memory_per_obj_dict[obj_key_name].store_prompt_result(
                    frame_idx, init_mem, init_ptr
                )

                # Draw prompts for debugging
                if enable_prompt_visualization:
                    prompt_vis_frame = cv2.resize(
                        frame_bgr,
                        dsize=None,
                        fx=0.5,
                        fy=0.5,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    norm_to_px_factor = np.float32(
                        (prompt_vis_frame.shape[1] - 1, prompt_vis_frame.shape[0] - 1)
                    )
                    for xy_norm in obj_prompts.get("fg_xy_norm_list", []):
                        xy_px = np.int32(xy_norm * norm_to_px_factor)
                        cv2.circle(prompt_vis_frame, xy_px, 1, (0, 255, 0), -1)
                    for xy_norm in obj_prompts.get("bg_xy_norm_list", []):
                        xy_px = np.int32(xy_norm * norm_to_px_factor)
                        cv2.circle(prompt_vis_frame, xy_px, 3, (0, 0, 255), -1)
                    for xy1_norm, xy2_norm in obj_prompts.get("box_tlbr_norm_list", []):
                        xy1_px = np.int32(xy1_norm * norm_to_px_factor)
                        xy2_px = np.int32(xy2_norm * norm_to_px_factor)
                        cv2.rectangle(
                            prompt_vis_frame, xy1_px, xy2_px, (0, 255, 255), 2
                        )

                    # Save prompt visualization frame
                    prompt_vis_path = os.path.join(
                        visualization_dir,
                        f"prompt_frame_{frame_idx}_{obj_key_name}.png",
                    )
                    cv2.imwrite(prompt_vis_path, prompt_vis_frame)

        # Update tracking using newest frame
        combined_mask_result = np.zeros(frame.shape[0:2], dtype=bool)
        for obj_key_name, obj_memory in memory_per_obj_dict.items():
            obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = (
                sammodel.step_video_masking(encoded_imgs_list, **obj_memory.to_dict())
            )

            # Skip storage for bad results (often due to occlusion)
            obj_score = obj_score.item()
            if obj_score < 0:
                print(
                    f"Bad object score for {obj_key_name}! Skipping memory storage..."
                )
                # continue

            # Store 'recent' memory encodings from current frame (helps track objects with changing appearance)
            # -> This can be commented out and tracking may still work, if object doesn't change much
            obj_memory.store_result(frame_idx, mem_enc, obj_ptr)

            # Add object mask prediction to 'combine' mask for display
            # -> This is just for visualization, not needed for tracking
            obj_mask = torch.nn.functional.interpolate(
                mask_preds[:, best_mask_idx, :, :],
                size=combined_mask_result.shape,
                mode="nearest",
                # align_corners=False,
            )
            obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
            combined_mask_result = np.bitwise_or(combined_mask_result, obj_mask_binary)

        print(f"Frame {frame_idx} processed!")
        # Combine original image & mask result side-by-side for display
        combined_mask_result_uint8 = combined_mask_result.astype(np.uint8) * 255
        disp_mask = cv2.cvtColor(combined_mask_result_uint8, cv2.COLOR_GRAY2BGR)
        sidebyside_frame = np.hstack((frame_bgr, disp_mask))
        sidebyside_frame = cv2.resize(
            sidebyside_frame,
            dsize=None,
            interpolation=cv2.INTER_NEAREST,
            fx=0.5,
            fy=0.5,
        )

        # Save result
        result_path = os.path.join(visualization_dir, f"result_frame_{frame_idx}.png")
        cv2.imwrite(result_path, sidebyside_frame)

except Exception as err:
    raise err

except KeyboardInterrupt:
    print("Closed by ctrl+c!")

finally:
    cv2.destroyAllWindows()
