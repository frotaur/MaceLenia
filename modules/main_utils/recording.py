import os, torch, cv2, numpy as np
from showtens import show_image, save_image
import pygame
import json


def launch_video(size, fps, save_folder="videos", save_name=None,fourcc="mp4v"):
    """
    Returns Videowriter, ready to record and save the video.

    Parameters:
    size: (H,W) 2-uple size of the video
    fps: int, frames per second
    save_folder: str, folder to save the video
    save_name: str, name of the video. If None, will save as "vid_<number>.mp4"
    fourcc : Encoder, must work with .mp4 videos
    """
    os.makedirs(save_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    if(save_name is None):
        numvids = len(os.listdir(save_folder))
        vid_loc = os.path.join(save_folder,f"vid_{numvids}.mp4")
    else:
        vid_loc = os.path.join(save_folder, f"{save_name}.mp4")
        
    return cv2.VideoWriter(vid_loc, fourcc, fps, (size[1], size[0]))


def add_frame(writer, worldsurface):
    worldmap = pygame.surfarray.array3d(worldsurface)  # (W,H,3)

    frame = worldmap.transpose(1, 0, 2)  # (H,W,3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame)


def print_screen(worldsurface, name=None):
    worldmap = pygame.surfarray.array3d(worldsurface)  # (W,H,3)
    os.makedirs("images", exist_ok=True)
    numimgs = len(os.listdir("images/"))
    img_name = f"img_{numimgs}" if name is None else name
    save_image(torch.tensor(worldmap, dtype=float).permute(2, 1, 0) / 255.0, folder="images", name=img_name)
