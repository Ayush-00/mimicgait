import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, Occ_Detector, Occ_Detector_Deep
from einops import rearrange
import pdb
import os

class Baseline_Mimic_Component(nn.Module):
    
    def __init__(self, model_cfg):
        super(Baseline_Mimic_Component, self).__init__()
        self.build_network(model_cfg)
    
    def build_network(self, model_cfg):
        # self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        # self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        #self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        #embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        #embed = embed_1

        # retval = {
        #     'training_feat': {
        #         'triplet': {'embeddings': embed_1, 'labels': labs},
        #         'softmax': {'logits': logits, 'labels': labs}
        #     },
        #     'visual_summary': {
        #         'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
        #     },
        #     'inference_feat': {
        #         'embeddings': embed
        #     }
        # }
        return embed_1
    


class Baseline_Mimic_Full_Guided(BaseModel):
    
    def build_backbones_components(self, model_cfg):
        # self.visible_component.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        # self.visible_component.Backbone = SetBlockWrapper(self.visible_component.Backbone)
        
        self.full_component.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.full_component.Backbone = SetBlockWrapper(self.full_component.Backbone)
        
        self.mimic_component.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.mimic_component.Backbone = SetBlockWrapper(self.mimic_component.Backbone)
        
    
    def build_network(self, model_cfg):
        #self.visible_component = Baseline_Mimic_Component(model_cfg)
        self.full_component = Baseline_Mimic_Component(model_cfg)
        self.mimic_component = Baseline_Mimic_Component(model_cfg)
        
        self.build_backbones_components(model_cfg)
        
        #self.BNNecks_unified = SeparateBNNecks(**model_cfg['SeparateBNNecksUnified'])
        
        #! Should I use BNNECKS_UNIFIED?
        
        #self.visible_component.requires_grad_(False)    #Used only for testing
        self.full_component.requires_grad_(False)  #Used only in inference mode
        
        if 'detector_type' in model_cfg.keys() and model_cfg['detector_type'] == 'deep':
            self.occ_detector = Occ_Detector_Deep(parts_num = model_cfg['bin_num'][0])
        else:
            self.occ_detector = Occ_Detector(parts_num = model_cfg['bin_num'][0])

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
                if k.split('.')[0] == 'BNNecks':
                    pass 
                else:
                    full_dict[k] = v
            self.full_component.load_state_dict(full_dict)
            
            if 'init_mimic' in self.cfgs['model_cfg']['mimic_cfg']:
                if self.cfgs['model_cfg']['mimic_cfg']['init_mimic']:
                    self.mimic_component.load_state_dict(full_dict)
                    
                    if torch.distributed.get_rank() == 0:
                        print(f"\nLoaded Mimic model from {model_path}!\n")
            
            
            ################# Load occ detector #################
            
            
            #Removing the module from the state dict
            model_cfg = self.cfgs['model_cfg']
            saved_weights = torch.load(model_cfg['occ_detector_path'])
            new_dict = {}
            for k, v in saved_weights.items():
                if k.startswith('module.'):
                    new_dict[k[7:]] = v
                else:
                    new_dict[k] = v
            
            #new_dict doesnt have the module. prefix. Now, remove unwanted layers
            filter_dict = {}
            for k, v in new_dict.items():
                if k.split('.')[0] in ['fc2', 'fc_amount_head', 'classification_head', 'layer_amount', 'layer5']:
                    #print("Removing: {}".format(k))
                    continue    # These are not needed
                else: 
                    filter_dict[k] = v
            
            
            self.occ_detector.load_state_dict(filter_dict)
            
            if torch.distributed.get_rank() == 0:
                print("\nLoaded Occ Detector from: {}\n".format(model_cfg['occ_detector_path']))
            
            #Uncomment
            #self.visible_component.load_state_dict(visible_dict)
            # self.full_component.load_state_dict(invisible_dict)
            # self.fcs_unified.load_state_dict(fcs_dict)
            #self.BNNecks_unified.load_state_dict(bnnecks_dict)
        else:
            raise ValueError('The mimic_cfg must be specified in model_cfg for full_mimic training')

        
        
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        
        
        visible_sils = ipts[0][:,:,:,:,0]
        full_sils = ipts[0][:,:,:,:,1]
        
        # assert visible_sils.max() != visible_sils.min(), "Dude no wayy"
        # assert invisible_sils.max() != invisible_sils.min(), "Dude no wayy invisible"
        
        
        visible_inputs = [[visible_sils], labs, _, _, seqL]
        full_inputs = [[full_sils], labs, _, _, seqL]
        
        if len(visible_sils.size()) == 4:
            sils = visible_sils.unsqueeze(1)
        else:
            sils = rearrange(visible_sils, 'n s c h w -> n c s h w')
            
        
        with torch.no_grad():
            occ_embed = self.occ_detector(sils, seqL) #[n, occ_dim, p]
            
        
        mimic_embed = self.mimic_component(visible_inputs)  #(n, c, p)
        
        mimic_occ = torch.cat([mimic_embed, occ_embed], dim=1)  #(n, c+occ_dim, p)
        mimic_occ_aware = self.occ_mixer_fc(mimic_occ)  #(n, c, p)
        
        if self.training:
            with torch.no_grad():
                full_embed = self.full_component(full_inputs)
                
            train_embed = torch.cat([mimic_occ_aware, full_embed], dim=0)  #(2n, c, p)
            
            inference_embed=None 
            labs = torch.cat([labs, labs], dim=0)
            
            # CAN HAVE XE LOSS, FLAT TRIPLET LOSS, AND LOSS WHICH BRINGS MIMIC AND INVISIBLE FEATS CLOSER, ALL AT THE SAME TIME! THINK ABOUT IT
            ###################################################

            # No. Saare losses conflict kar sakte hai. Only have the mimic loss - brings mimic close to invisible
            
            
            
        else:
            visible_embed = self.mimic_component(visible_inputs)
            inference_embed = visible_embed
            # inference_embed = torch.cat([visible_embed, mimic_embed], dim=2)  #(n, c, 2p)
            train_embed=None
        
            # visible_embed = self.visible_component(visible_inputs)
            # invisible_embed = self.invisible_component(invisible_inputs)
            
            #unified_embed = torch.cat([visible_embed, invisible_embed], dim=2) # [n, c, 2p]
            #inference_embed = self.fcs_unified(inference_embed) # [n, c, 2p]
            #embed_2, logits = self.BNNecks_unified(embed_1) # [n, c, 2p]
            
        
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
        
        
        
        
    
    
    
    
    
    
    
    
    
    