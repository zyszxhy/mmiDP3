import cv2
from termcolor import cprint
import pandas as pd
import shutil
from tqdm import tqdm
import visualizer
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/media/viewer/Image_Lab/embod_data/MarsMind_data")
parser.add_argument("--vis_save_path", type=str, default="/media/viewer/Image_Lab/embod_data/MarsMind_vis_data")


parser.add_argument("--use_img", type=int, default=1)
parser.add_argument("--vis_cloud", type=int, default=1)
parser.add_argument("--use_pc_color", type=int, default=0)
parser.add_argument("--downsample", type=int, default=0)

args = parser.parse_args()
use_img = args.use_img
dataset_path = args.dataset_path
vis_save_path = args.vis_save_path
vis_cloud = args.vis_cloud
use_pc_color = args.use_pc_color
downsample = args.downsample

episode_files = []

for dir_task in os.listdir(dataset_path):
    if dir_task == 'Sample_episode_12':
        continue
    for dir_episode in os.listdir(os.path.join(dataset_path, dir_task)):
        episode_files.append(os.path.join(dataset_path, dir_task, dir_episode))

    
# devide episodes by episode_ends
for episode_file in tqdm(episode_files, desc="Processing Episode"):
    i = 0
    save_dir = episode_file.replace(dataset_path, vis_save_path)
    if vis_cloud:
        os.makedirs(save_dir, exist_ok=True)
    # cprint(f"replay {episode_file.split('/')[-2]} {episode_file.split('/')[-1]}", "green")

    for dir_metaaction in tqdm(sorted(os.listdir(episode_file)), desc="Processing Metaaction"):
        if dir_metaaction == 'PHOTO' or 'README' in dir_metaaction or 'json' in dir_metaaction or 'pt' in dir_metaaction:
            continue
        
        for pc_file in sorted(os.listdir(os.path.join(episode_file, dir_metaaction, "pc"))):
            pc_data = pd.read_csv(os.path.join(episode_file, dir_metaaction, "pc", pc_file))    
            pc = pc_data[['x', 'y', 'z']].to_numpy()

            # downsample
            if downsample:
                num_points = 4096
                idx = np.random.choice(pc.shape[0], num_points, replace=False)
                pc = pc[idx]

            if use_img:
                shutil.copy(os.path.join(episode_file, dir_metaaction, "p0", "rgb", pc_file.replace('.csv', '.png')), os.path.join(save_dir, str(i)+'_rgb.png'))
            
            # if vis_cloud and i >= 50:
            if vis_cloud:
                if not use_pc_color:
                    pc = pc[:, :3]
                visualizer.visualize_pointcloud(pc, img_path=f"{save_dir}/{i}.png")
                i += 1
                # print(f"vis cloud saved to {save_dir}/{i}.png")
    
        if vis_cloud:
            # to video
            os.system(f"ffmpeg -r 10 -i {save_dir}/%d.png -vcodec mpeg4 -y {save_dir}/pc.mp4")

        
        
    




