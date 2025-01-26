import torch
from torch.utils.data import Dataset, DataLoader
import os
from os.path import join as osp
import numpy as np
import pickle
import random
import cv2
import math

class GREW_frames(Dataset):
    """Need preprocessed pkl files for this"""
    def __init__(self, root, mode):
        super(GREW_frames, self).__init__()
        #train_dirs = ['BRS1', 'BRS11']
        train_dirs = []
        test_dirs = ['probe']
        
        assert mode in ['train', 'test']
        self.mode = mode
        self.root = root
        
        print(f"This dataloader will sample a frame from a video and randomly occlude it, even during test phase")

        self.portion_range = [0.4,0.6]      # Range of portion of half consistent occlusion, as % of height/width of frame 
        
        self.vid_paths = []
        # self.labels = []        #subject id
        # self.walk_typ = []      #rand, struct
        self.env = []           #controlled, field-distance
        
        if mode == 'train':
            protocols = train_dirs
        else:
            protocols = test_dirs
        
        
        #####################
        #"""img for silhouette"""
        # rgb_img for rgb without background
        #filename = 'img.pkl'    #change to rgb_img.pkl for rgb without background
        #####################
        if mode == 'train':
            for sub in sorted(os.listdir(self.root)):
                if sub[-5:] == 'train':
                    sub_path = osp(self.root, sub)
                    for angle in os.listdir(sub_path):
                        angle_path = osp(sub_path, angle)
                        for sensor in os.listdir(angle_path):
                            sensor_path = osp(angle_path, sensor)
                            for vid in os.listdir(sensor_path):
                                vid_path = osp(sensor_path, vid)
                                self.vid_paths.append(vid_path)
                                self.env.append(f"{vid}")
        
        
        elif mode == 'test':
            for dir in protocols:
                #dir is 'probe'
                probe_path = osp(self.root, dir)
                for angle in os.listdir(probe_path):
                    angle_path = osp(probe_path, angle)
                    for probe_id in os.listdir(angle_path):
                        probe_id_path = osp(angle_path, probe_id)
                        for vid in os.listdir(probe_id_path):
                            vid_path = osp(probe_id_path, vid)
                            self.vid_paths.append(vid_path)
                            self.env.append(f"{vid}")
        
        print(f"Mode {self.mode}, Number of videos = {len(self.vid_paths)}")
                             
        return
        
    

    def read_vid(self, index):
        """Reads a video from pkl file, given an index"""
        file_p = self.vid_paths[index]
        
        with open(file_p, 'rb') as f:
            vid = pickle.load(f)
        
        return vid
    
    def sample_frame(self, vid):
        assert np.max(vid) != 0, f"Max value in video is 0! {vid}"

        while True:
            idx = int(torch.randint(low=0, high=len(vid), size=(1,)))
            sampled_frame = vid[idx]
            if np.max(sampled_frame) > 0:
                break
        
        return sampled_frame
    
    def resize_frame(self, frame, h, w):
        max_old = np.max(frame)
        old_dtype = frame.dtype
        resized_frame = cv2.resize(frame, dsize=(w, h))
        resized_frame = max_old*((resized_frame > 0.5*max_old).astype(old_dtype))
        return resized_frame    
    
    def random_occlude(self, frame):
        ### Occludes the frame (h,w), returns occluded frame of same size as original one    
        h,w = frame.shape     #Original size
        
        ######### OCCLUSION KEY ################
        # 0 - No occlusion
        # 1 - Top only - Bottom occluded
        # 2 - Bottom only - Top occluded
        
        
        # To train on more occlusion types, add more types here. Refer transform.py for code to generate more occlusions.
        
        portion = random.uniform(0.4, 0.6)
        occ_type = random.randint(0, 2)  #0,1,2

        h, w = frame.shape

        if occ_type == 0:
            occ_frame = frame
            portion = 0
        elif occ_type in [1, 2]:
            int_portion = int(portion*h) 
            if occ_type == 1:
                #Top only
                occ_frame = frame[:int_portion,:]
            elif occ_type == 2:
                # Bottom only
                occ_frame = frame[(h-int_portion):,:]
            occ_frame = occ_frame[:, int_portion//2 : -(int_portion//2)]

        resized_frame = self.resize_frame(occ_frame, h, w)
        return resized_frame, occ_type, portion
    
    def __getitem__(self, index):
        
        vid = self.read_vid(index)
        frame = self.sample_frame(vid)
        occ_frame, occ_type, occ_portion = self.random_occlude(frame)
        occ_frame = np.expand_dims(occ_frame, axis=0)   #(Need channel dimension.)
        
        return occ_frame, occ_type, occ_portion 

        

    def __len__(self):
        return len(self.vid_paths)
    
    
    def collate_fn(self,batch):
        frames = []
        occ_types = []
        occ_portions = []
        
        for item in batch:
            frames.append(item[0])
            occ_types.append(item[1])
            occ_portions.append(item[2])
        
        frames = np.array(frames)
        occ_types = np.array(occ_types)
        occ_portions = np.array(occ_portions)
        
        frames = torch.from_numpy(frames).float()
        occ_types = torch.from_numpy(occ_types)
        occ_portions = torch.from_numpy(occ_portions).float()

        return frames, occ_types, occ_portions

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)
    return
    
if __name__ == '__main__':
    print(f"Starting...")
    
    
    dataset = GREW_frames(root='path/', mode='train')
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn, worker_init_fn=worker_init_fn)
    
    for i, (frames, occ_types, occ_portions) in enumerate(dataloader):
        print(frames.shape)
        print(occ_types)
        print(occ_portions)
        break
    
    print(f"Complete")