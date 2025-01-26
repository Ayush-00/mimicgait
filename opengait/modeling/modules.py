import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple
from torchvision.ops import RoIAlign
import pdb

class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


class FocalConv2d(nn.Module):
    """
        GaitPart: Temporal Part-based Model for Gait Recognition
        CVPR2020: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
        Github: https://github.com/ChaoFan96/GaitPart
    """
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


class GaitAlign(nn.Module):
    """
        GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality
        ECCV2022: https://arxiv.org/pdf/2203.03972v2.pdf
        Github: https://github.com/ShiqiYu/OpenGait/tree/master/configs/gaitedge
    """
    def __init__(self, H=64, W=44, eps=1, **kwargs):
        super(GaitAlign, self).__init__()
        self.H, self.W, self.eps = H, W, eps
        self.Pad = nn.ZeroPad2d((int(self.W / 2), int(self.W / 2), 0, 0))
        self.RoiPool = RoIAlign((self.H, self.W), 1, sampling_ratio=-1)

    def forward(self, feature_map, binary_mask, w_h_ratio):
        """
           In  sils:         [n, c, h, w]
               w_h_ratio:    [n, 1]
           Out aligned_sils: [n, c, H, W]
        """
        n, c, h, w = feature_map.size()
        # w_h_ratio = w_h_ratio.repeat(1, 1) # [n, 1]
        w_h_ratio = w_h_ratio.view(-1, 1)  # [n, 1]

        h_sum = binary_mask.sum(-1)  # [n, c, h]
        _ = (h_sum >= self.eps).float().cumsum(axis=-1)  # [n, c, h]
        h_top = (_ == 0).float().sum(-1)  # [n, c]
        h_bot = (_ != torch.max(_, dim=-1, keepdim=True)
                 [0]).float().sum(-1) + 1.  # [n, c]

        w_sum = binary_mask.sum(-2)  # [n, c, w]
        w_cumsum = w_sum.cumsum(axis=-1)  # [n, c, w]
        w_h_sum = w_sum.sum(-1).unsqueeze(-1)  # [n, c, 1]
        w_center = (w_cumsum < w_h_sum / 2.).float().sum(-1)  # [n, c]

        p1 = self.W - self.H * w_h_ratio
        p1 = p1 / 2.
        p1 = torch.clamp(p1, min=0)  # [n, c]
        t_w = w_h_ratio * self.H / w
        p2 = p1 / t_w  # [n, c]

        height = h_bot - h_top  # [n, c]
        width = height * w / h  # [n, c]
        width_p = int(self.W / 2)

        feature_map = self.Pad(feature_map)
        w_center = w_center + width_p  # [n, c]

        w_left = w_center - width / 2 - p2  # [n, c]
        w_right = w_center + width / 2 + p2  # [n, c]

        w_left = torch.clamp(w_left, min=0., max=w+2*width_p)
        w_right = torch.clamp(w_right, min=0., max=w+2*width_p)

        boxes = torch.cat([w_left, h_top, w_right, h_bot], dim=-1)
        # index of bbox in batch
        box_index = torch.arange(n, device=feature_map.device)
        rois = torch.cat([box_index.view(-1, 1), boxes], -1)
        crops = self.RoiPool(feature_map, rois)  # [n, c, H, W]
        return crops


class Occ_Detector(torch.nn.Module):

    def __init__(self, parts_num):
        super(Occ_Detector, self).__init__()
        
        keep_prob = 1
        # L1 ImgIn shape=(?, 64, 64, 1)
        # Conv -> (?, 64, 64, 32)
        # Pool -> (?, 32, 32, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 32, 32, 32)
        # Conv      ->(?, 32, 32, 64)
        # Pool      ->(?, 16, 16, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 16, 16, 64)
        # Conv ->(?, 16,16, 128)
        # Pool ->(?, 8, 8, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(128, 64, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        
        # self.fc2 = torch.nn.Linear(64, 3, bias=True)
        # torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        
        
        # self.fc_amount_head = torch.nn.Linear(64, 1, bias=True)
        # torch.nn.init.xavier_uniform_(self.fc_amount_head.weight) # initialize parameters
        
        # self.classification_head = torch.nn.Sequential(
        #     self.fc2, 
        #     torch.nn.Softmax(dim=1)
        # )
        
        # self.layer_amount = torch.nn.Sequential(
        #     self.fc_amount_head,
        #     torch.nn.Sigmoid()
        # )
        
        self.parts_num = parts_num
        
        return 
    
    def forward(self, x, seqL):
        #x: (Batch, frames, c, h, w)    #(c = 1)
        #pdb.set_trace()
        b, c, f, h, w = x.shape
        #pdb.set_trace()
        if seqL is not None:
        
            assert b == 1, "Batch size must be 1"
            x = x.permute(0,2,1,3,4).contiguous()    #(b, f, c, h, w)
            
            out = x.view(b*f, c, h, w)
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            
            out = out.view(out.size(0), -1)   # Flatten them for FC
            out = self.layer4(out)
            
            out = out.view(b, f, -1)
            out = out.squeeze(0) # (f, 64)
            
            #desired_b = seqL.shape[1]
            accumulator = 0
            output_lis = []
            for i,l in enumerate(seqL[0]):
                output_lis.append(torch.mean(out[accumulator:accumulator+l], dim=0))
                accumulator += l
                
            try:
                output_lis = torch.stack(output_lis, dim=0)  #[n,c=64]
                occ_embed = output_lis.unsqueeze(2).repeat(1,1,self.parts_num)   #(n,c,p)  repeated
            except IndexError:
                pdb.set_trace()
                print(f"Error BRUH: {output_lis.shape}")
            return occ_embed

        else: 
            # seqL is None. 
            x = x.permute(0,2,1,3,4).contiguous()    #(b, f, c, h, w)
            
            out = x.view(b*f, c, h, w)
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            
            out = out.view(out.size(0), -1)   # Flatten them for FC
            out = self.layer4(out)
            
            out = out.view(b, f, -1)
            out = torch.mean(out, dim=1) # (n,c=64)
            occ_embed = out.unsqueeze(2).repeat(1,1,self.parts_num)     #(n,c,p)
            
            return occ_embed


class Occ_Detector_Deep(torch.nn.Module):

    def __init__(self, parts_num, multiply=False, return_amt = False):
        super(Occ_Detector_Deep, self).__init__()
        
        #multiply = True if you want to multiply the input by 255.0. Multiplication happens only if the input is in [0,1].

        keep_prob = 1
        # L1 ImgIn shape=(?, 64, 64, 1)
        # Conv -> (?, 64, 64, 32)
        # Pool -> (?, 32, 32, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 32, 32, 32)
        # Conv      ->(?, 32, 32, 64)
        # Pool      ->(?, 16, 16, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 16, 16, 64)
        # Conv ->(?, 16,16, 128)
        # Pool ->(?, 8, 8, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 ImgIn shape=(?, 8, 8, 128)
        # Conv ->(?, 8, 8, 256)
        # Pool ->(?, 4, 4, 256)
        self.layer3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))


        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc0 = torch.nn.Linear(256, 128, bias=True)
        self.fc1 = torch.nn.Linear(128, 64, bias=True)
        # torch.nn.init.xavier_uniform_(self.fc0.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc0,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob),
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        
        
        if return_amt: 
            
            self.fc2 = torch.nn.Linear(64, 5, bias=True)
            # torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
            
            self.fc_amount_head = torch.nn.Linear(64, 1, bias=True)
            # torch.nn.init.xavier_uniform_(self.fc_amount_head.weight) # initialize parameters
            
            self.classification_head = torch.nn.Sequential(
                self.fc2, 
                torch.nn.Softmax(dim=1)
            )
            
            self.layer_amount = torch.nn.Sequential(
                self.fc_amount_head,
                torch.nn.Sigmoid()
            )
        
        self.parts_num = parts_num
        self.multiply = multiply        
        self.return_amt = return_amt
        
        return 
    
    def forward(self, x, seqL):
        #x: (Batch, frames, c, h, w)    #(c = 1)
        #pdb.set_trace()
        b, c, f, h, w = x.shape
        if self.multiply:
            if x.max() == 1.0:
                new_x = x.clone().detach()
                new_x = new_x * 255
            else:
                new_x = x
        else: 
            new_x = x
        if seqL is not None:
        
            assert b == 1, "Batch size must be 1"
            x = new_x.permute(0,2,1,3,4).contiguous()    #(b, f, c, h, w)
            
            out = x.view(b*f, c, h, w)
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            
            out = self.layer3_1(out)
            
            out = self.avg_pool(out)
            
            out = out.view(out.size(0), -1)   # Flatten them for FC
            out = self.layer4(out)

            #pdb.set_trace()
            #amt
            if self.return_amt:
                amt = self.layer_amount(out)    # (f,1)
                pred_occs = self.classification_head(out).argmax(dim=1)  # (f,1)
            
            out = out.view(b, f, -1)
            out = out.squeeze(0) # (f, 64)

            if self.return_amt:
                amt = amt.view(b, f, -1)
                amt = amt.squeeze(0)    # (f,1)
            
            #desired_b = seqL.shape[1]
            accumulator = 0
            output_lis = []
            amt_lis = [] if self.return_amt else None
            occ_lis = [] if self.return_amt else None
            for i,l in enumerate(seqL[0]):
                output_lis.append(torch.mean(out[accumulator:accumulator+l], dim=0))
                if self.return_amt:
                    amt_lis.append(torch.mean(amt[accumulator:accumulator+l], dim=0))
                    occ_lis.append(torch.mode(pred_occs[accumulator:accumulator+l], dim=0)[0])
                accumulator += l
                
            try:
                output_lis = torch.stack(output_lis, dim=0)  #[n,c=64]                
                occ_embed = output_lis.unsqueeze(2).repeat(1,1,self.parts_num)   #(n,c,p)  repeated
                if self.return_amt:
                    amt_lis = torch.stack(amt_lis, dim=0).squeeze()  #(n)
                    occ_lis = torch.stack(occ_lis, dim=0).squeeze()  #(n)
                    
            except IndexError:
                pdb.set_trace()
                print(f"Error BRUH: {output_lis.shape}")
            if self.return_amt:
                return occ_embed, amt_lis, occ_lis
            else:
                return occ_embed

        else: 
            # seqL is None. 
            x = x.permute(0,2,1,3,4).contiguous()    #(b, f, c, h, w)
            if self.return_amt:
                raise NotImplementedError("return_amt is not implemented for seqL=None")
            
            out = x.view(b*f, c, h, w)
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            
            out = self.layer3_1(out)
            
            out = self.avg_pool(out)
            
            out = out.view(out.size(0), -1)   # Flatten them for FC
            out = self.layer4(out)

            #amt
            amt = self.layer_amount(out)
            occ = self.classification_head(out).argmax(dim=1)
            
            out = out.view(b, f, -1)
            amt = amt.view(b, f, -1)
            occ = occ.view(b, f, -1)

            out = torch.mean(out, dim=1) # (n,c=64)
            amt = torch.mean(amt, dim=1) # (n,1)
            occ = torch.mode(occ, dim=1)[0] # (n,1)
            occ_embed = out.unsqueeze(2).repeat(1,1,self.parts_num)     #(n,c,p)
            
            return occ_embed


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False
