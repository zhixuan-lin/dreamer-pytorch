import pandas as pd
import cv2 
from torch.utils.data import IterableDataset, DataLoader
import random
import torch
import numpy as np

class Logger:
    def __init__(self):
        self.log = {}
    def add_log(self,feature_name,value):
        self.log[feature_name] = value       
    def fill_missing_values(self,data: dict) -> dict:
        max_len = max([len(v) for v in data.values()])
        for key in data:
            if len(data[key]) < max_len:
                data[key] += [None] * (max_len - len(data[key]))
        return data
    def write_to_csv(self, file_name):
        filled_data = self.fill_missing_values(self.log)
        df = pd.DataFrame(filled_data)
        df.to_csv(file_name, index=False)
        
    def write_video(self,filepath,frames, fps=60):
        """ Write a video to disk using openCV
            filepath : the path to write the video to 
            frames : a numpy array with shape (time, height, width, channels)
            fps : the number of frames per second
        """
        height, width, channels = frames.shape[1:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

def set_seed(seed : int,device):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)