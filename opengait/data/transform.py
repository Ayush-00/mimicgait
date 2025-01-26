from typing import Any
import numpy as np
import random
import torchvision.transforms as T
import cv2
import math
from data import transform as base_transform
from utils import is_list, is_dict, get_valid_args
import pdb


class NoOperation():
    def __call__(self, x):
        return x

class BaseSilTransform():
    def __init__(self, divsor=255.0, img_shape=None):
        self.divsor = divsor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.divsor


class BaseSilCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        x = x[..., cutting:-cutting]
        return x / self.divsor


class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std


# **************** Data Agumentation ****************


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[..., ::-1]


class RandomErasing(object):
    def __init__(self, prob=0.5, sl=0.05, sh=0.2, r1=0.3, per_frame=False):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                for _ in range(100):
                    seq_size = seq.shape
                    area = seq_size[1] * seq_size[2]

                    target_area = random.uniform(self.sl, self.sh) * area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    if w < seq_size[2] and h < seq_size[1]:
                        x1 = random.randint(0, seq_size[1] - h)
                        y1 = random.randint(0, seq_size[2] - w)
                        seq[:, x1:x1+h, y1:y1+w] = 0.
                        return seq
            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...])
                   for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)


class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            return seq


class RandomPerspective(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, h, w = seq.shape
            cutting = int(w // 44) * 10
            x_left = list(range(0, cutting))
            x_right = list(range(w - cutting, w))
            TL = (random.choice(x_left), 0)
            TR = (random.choice(x_right), 0)
            BL = (random.choice(x_left), h)
            BR = (random.choice(x_right), h)
            srcPoints = np.float32([TL, TR, BR, BL])
            canvasPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            perspectiveMatrix = cv2.getPerspectiveTransform(
                np.array(srcPoints), np.array(canvasPoints))
            seq = [cv2.warpPerspective(_[0, ...], perspectiveMatrix, (w, h))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            return seq


class RandomAffine(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            # rotation
            max_shift = int(dh // 64 * 10)
            shift_range = list(range(0, max_shift))
            pts1 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            pts2 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            M1 = cv2.getAffineTransform(pts1, pts2)
            # affine
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            return seq
        

class MovingPatch(object):
    def __init__(self, prob=0.5, width_range=[0.4,0.6], height_range=[0.3,0.5], velocity_range=[0.5, 2]):
        self.prob = prob
        self.width_range = width_range
        self.height_range = height_range
        self.velocity_range = velocity_range
        return
    
    def black_patch(self, img, start_x, width=0, height=0):
        #start_x = pixel
        #img = image
        #width = width of patch
        new_img = img.copy()
        h,w = new_img.shape
        new_img[h-height:,start_x:start_x+width] = 0
        return new_img
    
    def __call__(self, seq):
        #seq is of shape (T, H, W)
        _, h, w = seq.shape
        
        width = int(random.uniform(*self.width_range)*w)
        height = int(random.uniform(*self.height_range)*h)
        velocity = random.uniform(*self.velocity_range)
        #pdb.set_trace()
        new_video = np.zeros_like(seq)
        
        for i, frame in enumerate(seq):
            new_video[i] = self.black_patch(frame, int(i*velocity), width, height)
            
        
        return new_video
        
        
class MovingPole(MovingPatch):
    def __init__(self, prob=0.5, width_range=[0.4,0.6], velocity_range=[0.5, 2]):
        super().__init__(prob, width_range, [1.0,1.0], velocity_range)
        return


class ConsistentRectanglePatch(object):
    def __init__(self, prob=0.5, width_range=[0.4,0.6], height_range=[0.3,0.5]):
        self.prob = prob
        self.width_range = width_range
        self.height_range = height_range
        return
    
    def __call__(self, seq):
        #seq is of shape (T, H, W)
        new_vid = seq.copy()
        f,h,w = new_vid.shape
        
        width = int(random.uniform(*self.width_range)*w)
        height = int(random.uniform(*self.height_range)*h)
        
        if random.uniform(0,1) > 0.5:
            start_x = 0 
        else:
            start_x = w - width

        if random.uniform(0,1) > 0.5:
            new_vid[:, h-height:,start_x:start_x+width] = 0
        else:
            new_vid[:, :height,start_x:start_x+width] = 0

        return new_vid


class ConsistentHalfPatch(object):
    def __init__(self, prob, portion_range=[0.4,0.6]):
        self.prob = prob
        self.portion_range = portion_range      
        return
    
    def __call__(self, seq):
        
        
        
        new_vid = seq.copy()
        f,h,w = new_vid.shape

        occ_typ = random.randint(0,3)

        if occ_typ == 0:
            portion = int(random.uniform(*self.portion_range)*h)
            new_vid[:, :portion, :] = 0
        elif occ_typ == 1:
            portion = int(random.uniform(*self.portion_range)*h)
            new_vid[:, (h-portion): , :] = 0
        elif occ_typ == 2:
            portion = int(random.uniform(*self.portion_range)*w)
            new_vid[:, :, :portion] = 0
        elif occ_typ == 3:
            portion = int(random.uniform(*self.portion_range)*w)
            new_vid[:,:,(w-portion):] = 0

        return new_vid


class ConsistentHalfPatchTwoTypesResized(object):
    def __init__(self, prob, portion_range=[0.4,0.6], img_size=64):
        self.prob = prob
        self.portion_range = portion_range      
        self.img_size = img_size
        return
    
    def resize_seq(self, seq):
        max_old = np.max(seq)       #Either 1 or 255.0 - depending on whether mask in binary or uint8
        old_dtype = seq.dtype
        resized_seq = np.array([cv2.resize(frame, dsize=(self.img_size, self.img_size)) for frame in seq])
        resized_seq = max_old*((resized_seq > 0.5*max_old).astype(old_dtype))
        return resized_seq
    
    def __call__(self, video):
        
        f, h, w = video.shape
        occ_portion = random.uniform(self.portion_range[0], self.portion_range[1])
        
        if random.random() < 0.5:
            # Top video is visible
            visible = video[:, :int((1-occ_portion)*h), :]
        else:
            # Bottom video is visible
            visible = video[:, int(occ_portion*h):, :]
        
        visible = self.resize_seq(visible)
        return visible


class ConsistentMiddlePatch(object):
    def __init__(self, prob, portion_range=[0.5,0.5], img_size=64):
        self.prob = prob
        self.portion_range = portion_range      
        self.img_size = img_size
        return  
    
    def resize_seq(self, seq):
        max_old = np.max(seq)       #Either 1 or 255.0 - depending on whether mask in binary or uint8
        old_dtype = seq.dtype
        resized_seq = np.array([cv2.resize(frame, dsize=(self.img_size, self.img_size)) for frame in seq])
        resized_seq = max_old*((resized_seq > 0.5*max_old).astype(old_dtype))
        return resized_seq
    
    def __call__(self, video):
        f, h, w = video.shape
        occ_portion = random.uniform(self.portion_range[0], self.portion_range[1])
        video[:, int((1-occ_portion)*h/2):int((1+occ_portion)*h/2), :] = 0
        visible = self.resize_seq(video)
        return visible


class ConsistentHalfPatchOneTypeResized(object):
    def __init__(self, prob, occluded_part, portion_range=[0.4,0.6], img_size=64):
        self.prob = prob
        self.occluded_part = occluded_part
        self.portion_range = portion_range      
        self.img_size = img_size
        
        assert self.occluded_part in ['top', 'bottom'], "occluded_part must be either 'top' or 'bottom'"
        
        return
    
    def resize_seq(self, seq):
        max_old = np.max(seq)       #Either 1 or 255.0 - depending on whether mask in binary or uint8
        old_dtype = seq.dtype
        resized_seq = np.array([cv2.resize(frame, dsize=(self.img_size, self.img_size)) for frame in seq])
        resized_seq = max_old*((resized_seq > 0.5*max_old).astype(old_dtype))
        return resized_seq
    
    def __call__(self, video):
        
        f, h, w = video.shape
        occ_portion = random.uniform(self.portion_range[0], self.portion_range[1])
        
        if self.occluded_part == 'bottom':
            # Top video is visible
            visible = video[:, :int((1-occ_portion)*h), :]
        else:
            # Bottom video is visible
            visible = video[:, int(occ_portion*h):, :]
        
        visible = self.resize_seq(visible)
        return visible
    
class ConsistentHalfPatchTwoTypesResizedOcclusionChangeInBetween(object):
    # In this transform, the occlusion type is changed at the midpoint of the video
    def __init__(self, prob, portion_range=[0.4,0.6], img_size=64):
        self.prob = prob
        self.portion_range = portion_range      
        self.img_size = img_size
        return
    
    def resize_seq(self, seq):
        max_old = np.max(seq)       #Either 1 or 255.0 - depending on whether mask in binary or uint8
        old_dtype = seq.dtype
        resized_seq = np.array([cv2.resize(frame, dsize=(self.img_size, self.img_size)) for frame in seq])
        resized_seq = max_old*((resized_seq > 0.5*max_old).astype(old_dtype))
        return resized_seq
    
    def __call__(self, video):
        
        f, h, w = video.shape
        occ_portion = random.uniform(self.portion_range[0], self.portion_range[1])
        
        first_half = video[:f//2]
        second_half = video[f//2:]
        if random.random() < 0.5:
            # Top video is visible for the first half of the video
            visible_first_half = first_half[:, :int((1-occ_portion)*h), :]
            visible_second_half = second_half[:, int(occ_portion*h):, :]
            
    
            # Old - Top video is visible
            #visible = video[:, :int((1-occ_portion)*h), :]
        else:
            # Bottom video is visible for the first half of the video
            visible_first_half = first_half[:, int(occ_portion*h):, :]
            visible_second_half = second_half[:, :int((1-occ_portion)*h), :]
            
            
            # Old - Bottom video is visible
            #visible = video[:, int(occ_portion*h):, :]
        
        #Old - resize visible
        #visible = self.resize_seq(visible)
        
        #Resize top and bottom videos separately
        visible_first_half = self.resize_seq(visible_first_half)
        visible_second_half = self.resize_seq(visible_second_half)
        #COncatenate the two videos
        visible = np.concatenate([visible_first_half, visible_second_half], axis=0)
        
        return visible

class ConsistentHalfPatchTwoTypes(object):
    def __init__(self, prob, portion_range=[0.4,0.6]):
        self.prob = prob
        self.portion_range = portion_range      
        return
    
    def __call__(self, seq):
        
        new_vid = seq.copy()
        f,h,w = new_vid.shape

        occ_typ = random.randint(0,1)

        if occ_typ == 0:
            portion = int(random.uniform(*self.portion_range)*h)
            new_vid[:, :portion, :] = 0
        elif occ_typ == 1:
            portion = int(random.uniform(*self.portion_range)*h)
            new_vid[:, (h-portion): , :] = 0

        return new_vid

class Occlusion(object):
    def __init__(self, shape_cfgs = []):
        total_prob = 0
        self.prob_cumulative = []
        for s in shape_cfgs:
            #print(s)
            total_prob += s['prob']
            self.prob_cumulative.append(total_prob)
            
        if total_prob > 1:
            raise ValueError('Total probability of transient occlusion shapes is greater than 1')
        if total_prob < 1:
            shape_cfgs.append({'type': 'NoOperation', 'prob': 1-total_prob})
            self.prob_cumulative.append(1.0)
        
        self.occs = [get_transform(s) for s in shape_cfgs]
        self.num_shapes = len(shape_cfgs)
        return
    
    
    def __call__(self, seq):
        #pdb.set_trace()
        coin_flip = random.uniform(0, 1)
        
        for shape_idx in range(self.num_shapes):
            if coin_flip <= self.prob_cumulative[shape_idx]:
                return self.occs[shape_idx](seq)    
# ******************************************


class MimicOcclusionWrapper(object):
    # MimicOcclusionWrapper - Returns an occluded visible part of the video, and the full video, for the specified occlusion type(s)
    def __init__(self, shape_cfgs = []):
        total_prob = 0
        self.prob_cumulative = []
        for s in shape_cfgs:
            #print(s)
            total_prob += s['prob']
            self.prob_cumulative.append(total_prob)
            
        if total_prob > 1:
            raise ValueError('Total probability of transient occlusion shapes is greater than 1')
        if total_prob < 1:
            shape_cfgs.append({'type': 'NoOperation', 'prob': 1-total_prob})
            self.prob_cumulative.append(1.0)
        
        self.occs = [get_transform(s) for s in shape_cfgs]
        self.num_shapes = len(shape_cfgs)
        return
    
    def __call__(self, seq):
        #pdb.set_trace()
        coin_flip = random.uniform(0, 1)
        full_seq = seq.copy()
        for shape_idx in range(self.num_shapes):
            if coin_flip <= self.prob_cumulative[shape_idx]:
                visible_seq = self.occs[shape_idx](seq)
                return np.stack([visible_seq, full_seq], axis=-1)



########### Mimic split transform ############

class MimicSplitTransform(object):
    def __init__(self, portion_range = [0.4,0.6], img_size = 64):
        self.portion_range = portion_range
        self.img_size = img_size
        return
    
    def split_visible_invisible(self, video):
        # Split into visible and invisible video, either top occlusiopn or bottom occlusion
        # video: (f, h, w)
        f,h,_ = video.shape
        occ_portion = random.uniform(self.portion_range[0], self.portion_range[1])
        if random.random() < 0.5:
            # Top video is visible
            visible = video[:, :int((1-occ_portion)*h), :]
            invisible = video[:, int((1-occ_portion)*h):, :]
        else:
            # Bottom video is visible
            visible = video[:, int(occ_portion*h):, :]
            invisible = video[:, :int(occ_portion*h), :]
        return visible, invisible
    
    
    def resize_seq(self, seq):
        max_old = np.max(seq)       #Either 1 or 255.0 - depending on whether mask in binary or uint8
        old_dtype = seq.dtype
        resized_seq = np.array([cv2.resize(frame, dsize=(self.img_size, self.img_size)) for frame in seq])
        resized_seq = max_old*((resized_seq > 0.5*max_old).astype(old_dtype))
        return resized_seq
    
    def cut_resize_seq(self, seq):
        # seq is of shape (T, H, W) - It will have less H than W since it has already been cut
        _, h, w = seq.shape
        seq = seq[:, :, (w-h)//2:-(w-h)//2]     #(cut frames to (h,h) shape)
        resized_seq = self.resize_seq(seq)
        return resized_seq
        
        
    def __call__(self, seq):
        # seq is of shape (T, H, W)
        # We are not using cut_resize_seq, since we are dealing with mre than 50% top/bottom occlusion - Taking out the sides to maintain aspect ratio will crop out useful information in the frames.
        visible, invisible = self.split_visible_invisible(seq)
        visible_resized_seq = self.resize_seq(visible)
        invisible_resized_seq = self.resize_seq(invisible)
        
        return np.stack([visible_resized_seq, invisible_resized_seq], axis=-1)
    

class MimicFullTransform(object):
    # MimicFull - Returns an occluded visible part of the video, and the full video. Visible part is either top or bottom (2 types occlusion)
    def __init__(self, portion_range = [0.4,0.6], img_size = 64):
        self.portion_range = portion_range
        self.img_size = img_size
        return
    
    def split_visible_full(self, video):
        f, h, w = video.shape
        occ_portion = random.uniform(self.portion_range[0], self.portion_range[1])
        
        if random.random() < 0.5:
            # Top video is visible
            visible = video[:, :int((1-occ_portion)*h), :]
            
        else:
            # Bottom video is visible
            visible = video[:, int(occ_portion*h):, :]
        
        full = video
        return visible, full
        
    def resize_seq(self, seq):
        max_old = np.max(seq)       #Either 1 or 255.0 - depending on whether mask in binary or uint8
        old_dtype = seq.dtype
        resized_seq = np.array([cv2.resize(frame, dsize=(self.img_size, self.img_size)) for frame in seq])
        resized_seq = max_old*((resized_seq > 0.5*max_old).astype(old_dtype))
        return resized_seq
    
    def pad_full_seq(self, seq, desired_size):
        # (Pad seq to (desired_size, desired_size) shape)
        f, h, w = seq.shape
        
        assert h == desired_size
        cutting = (desired_size - w)//2
        #print(f"Calculated cutting = {cutting}")
        padded = np.pad(seq, ((0,0), (0,0), (cutting, cutting)), 'constant', constant_values=0)
        #pads cutting number of zeros on both sides of width
        
        return padded
    
    
    def __call__(self, seq):
        # seq is of shape (T, H, W)
        visible, full = self.split_visible_full(seq)
        visible_resized_seq = self.resize_seq(visible)
        full_padded = self.pad_full_seq(full, self.img_size)
        #print(f"visible_resized_seq.shape: {visible_resized_seq.shape}, full_padded.shape: {full_padded.shape}")
        return np.stack([visible_resized_seq, full_padded], axis=-1)



def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform


def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"
