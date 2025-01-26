import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks, Occ_Detector
import os
from einops import rearrange



class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class GaitGL_Mimic_Component(nn.Module):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, model_cfg):
        super(GaitGL_Mimic_Component, self).__init__()
        self.build_network(model_cfg)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        #class_num = model_cfg['class_num']
        dataset_name = "BRIAR"

        if dataset_name in ['OUMVLP', 'GREW', 'BRIAR']:
            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = nn.Sequential(
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.Sequential(
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.GLConvB2 = nn.Sequential(
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
        else:
            raise ValueError("Not support dataset: {}!".format(dataset_name))
            
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        # if 'SeparateBNNecks' in model_cfg.keys():
        #     self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        #     self.Bn_head = False
        #else:
        self.Bn = nn.BatchNorm1d(in_c[-1])
        #self.Head1 = SeparateFCs(64, in_c[-1], class_num)
        self.Bn_head = True
        
        return

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        #if self.Bn_head:  # Original GaitGL Head
        bnft = self.Bn(gait)  # [n, c, p]
        #logi = self.Head1(bnft)  # [n, c, p]
        embed = bnft
        # else:  # BNNechk as Head
        #     bnft, logi = self.BNNecks(gait)  # [n, c, p]
        #     embed = gait

        # n, _, s, h, w = sils.size()
        # retval = {
        #     'training_feat': {
        #         'triplet': {'embeddings': embed, 'labels': labs},
        #         'softmax': {'logits': logi, 'labels': labs}
        #     },
        #     'visual_summary': {
        #         'image/sils': sils.view(n*s, 1, h, w)
        #     },
        #     'inference_feat': {
        #         'embeddings': embed
        #     }
        # }
        return embed


class GaitGL_Mimic_Full_Guided(BaseModel):

    def build_network(self, model_cfg):
        self.full_component = GaitGL_Mimic_Component(model_cfg)
        self.mimic_component = GaitGL_Mimic_Component(model_cfg)
        
        class_num = model_cfg['class_num']
        in_c = model_cfg['channels']
        #self.Head1 = SeparateFCs(64, in_c[-1], class_num)
        
        self.full_component.requires_grad_(False)
        
        self.occ_detector = Occ_Detector(parts_num = model_cfg['OccMixerFCs']['parts_num'])
        self.occ_detector.requires_grad_(False)
        
        self.occ_mixer_fc = SeparateFCs(**model_cfg['OccMixerFCs'])
        
        
        
    def init_parameters(self):
        super().init_parameters()
        
        if 'mimic_cfg' in self.cfgs['model_cfg']:
            model_path = os.path.join(
                './output',
                self.cfgs['data_cfg']['dataset_name'], 
                self.cfgs['model_cfg']['mimic_cfg']['teacher_model_name'], 
                self.cfgs['model_cfg']['mimic_cfg']['teacher_save_name'], 'checkpoints', 
                f"{self.cfgs['model_cfg']['mimic_cfg']['teacher_save_name']}-{str(self.cfgs['model_cfg']['mimic_cfg']['teacher_model_iter']).zfill(5)}.pt")
            
            device_rank = torch.distributed.get_rank()
            device=torch.device("cuda", device_rank)
            teacher_model = torch.load(model_path, map_location=device)
            if torch.distributed.get_rank() == 0:
                print(f"\nLoaded teacher model from {model_path}!\n")
        
            full_dict = {}
            
            for k,v in teacher_model['model'].items():
                if k.split('.')[0] == 'Head1':
                    pass 
                else:
                    full_dict[k] = v
            self.full_component.load_state_dict(full_dict)
            
            ################# Load occ detector #################
            
            
            #Removing the module from the state dict
            model_cfg = self.cfgs['model_cfg']
            saved_weights = torch.load(model_cfg['occ_detector_path'], map_location=device)
            new_dict = {}
            for k, v in saved_weights.items():
                if k.startswith('module.'):
                    new_dict[k[7:]] = v
                else:
                    new_dict[k] = v
            
            #new_dict doesnt have the module. prefix. Now, remove unwanted layers
            filter_dict = {}
            for k, v in new_dict.items():
                if k.split('.')[0] in ['fc2', 'fc_amount_head', 'classification_head', 'layer_amount']:
                    #print("Removing: {}".format(k))
                    continue    # These are not needed
                else: 
                    filter_dict[k] = v
            
            
            self.occ_detector.load_state_dict(filter_dict)
            
            if torch.distributed.get_rank() == 0:
                print("\nLoaded Occ Detector from: {}\n".format(model_cfg['occ_detector_path']))
            
        else:
            raise ValueError('The mimic_cfg must be specified in model_cfg for full_mimic training')
        
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        visible_sils = ipts[0][:,:,:,:,0]
        full_sils = ipts[0][:,:,:,:,1]
        
        # assert visible_sils.max() != visible_sils.min(), "Dude no wayy"
        # assert invisible_sils.max() != invisible_sils.min(), "Dude no wayy invisible"
        
        
        visible_inputs = [[visible_sils], labs, _, _, seqL]
        invisible_inputs = [[full_sils], labs, _, _, seqL]
        
        if len(visible_sils.size()) == 4:
            sils = visible_sils.unsqueeze(1)
        else:
            sils = rearrange(visible_sils, 'n s c h w -> n c s h w')
        
        with torch.no_grad():
            occ_embed = self.occ_detector(sils, seqL) #[n, occ_dim, p]
        
        mimic_embed = self.mimic_component(visible_inputs)
        
        mimic_occ = torch.cat([mimic_embed, occ_embed], dim=1)  #(n, c+occ_dim, p)
        mimic_occ_aware = self.occ_mixer_fc(mimic_occ)  #(n, c, p)
        
        if self.training:
            with torch.no_grad():
                full_embed = self.full_component(invisible_inputs)
            train_embed = torch.cat([mimic_occ_aware, full_embed], dim=0)  #(2n, c, p)
            inference_embed=None 
            labs = torch.cat([labs, labs], dim=0)
            
        else:
            visible_embed = self.mimic_component(visible_inputs)
            inference_embed = visible_embed
            # inference_embed = torch.cat([visible_embed, mimic_embed], dim=2)  #(n, c, 2p)
            train_embed=None            
        
        retval = {
            'training_feat': {
                'triplet': {'embeddings': train_embed, 'labels': labs}#,
                #'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': inference_embed
            }
        }
        
        return retval