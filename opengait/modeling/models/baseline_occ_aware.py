import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, Occ_Detector
import pdb
from einops import rearrange

class Baseline_occ_aware(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.occ_detector = Occ_Detector(parts_num=model_cfg['OccMixerFCs']['parts_num'])
        self.occ_detector.eval()
        self.occ_detector.requires_grad_(False)
        self.occ_mixer_fc = SeparateFCs(**model_cfg['OccMixerFCs'])
        
    def init_parameters(self):
        super(Baseline_occ_aware, self).init_parameters()
        if 'occ_detector_path' in self.cfgs['model_cfg']:
            model_path = self.cfgs['model_cfg']['occ_detector_path']
            
            ## Load model onto correct gpu ##
            rank = torch.distributed.get_rank()
            device = torch.device(rank)
            pretrained_model = torch.load(model_path, map_location=device)
            
            new_dict = {}
            for k,v in pretrained_model.items():
                if k.split('.')[1] in ['fc2', 'fc_amount_head', 'classification_head', 'layer_amount', 'layer5']:
                    continue    #Skip these last layers.
                else:
                    new_key = '.'.join(k.split('.')[1:])
                    new_dict[new_key] = v
            
            self.occ_detector.load_state_dict(new_dict)
            if torch.distributed.get_rank() == 0:
                print(f"\nOCCLUSION DETECTOR LOADED FROM: {model_path}\n")
        else:
            raise ValueError("Specify occ_detector_path in model_cfg!")


    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        
        with torch.no_grad():
            occ_embed = self.occ_detector(sils, seqL)       #[n, occ_dim, p]   
        
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        
        ## Occlusion aware ##
        embed_1 = torch.cat([embed_1, occ_embed], dim=1)  # [n, c+occ_dim, p]
        embed_1 = self.occ_mixer_fc(embed_1)  # [n, c, p]
        #####################
        
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval