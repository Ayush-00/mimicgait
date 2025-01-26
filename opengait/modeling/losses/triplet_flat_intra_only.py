import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper
from pytorch_metric_learning import losses


class TripletLossFlatIntraOnly(BaseLoss):
    # Anchor and positive pair is only from same instance/sequece
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLossFlatIntraOnly, self).__init__(loss_term_weight)
        self.loss_fn = losses.TripletMarginLoss(margin=margin)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [2n, c, p]. Embeddings of mimic and invisible net are concatenated in this loss used only for phase 2.
        # labels: NOT USED.
        
        
        
        # hard_loss = torch.max(loss, -1)[0]
        # loss_avg, loss_num = self.AvgNonZeroReducer(loss)
        n,c,p = embeddings.shape
        flat_embed = embeddings.view(n, -1)
        
        new_labels = torch.arange(n//2).repeat(2).to(labels.device)
        
        loss = self.loss_fn(flat_embed, new_labels)
        loss_avg = loss

        self.info.update({
            'loss': loss_avg.detach().clone()
            # 'hard_loss': hard_loss.detach().clone(),
            # 'loss_num': loss_num.detach().clone(),
            # 'mean_dist': mean_dist.detach().clone()
            })

        return loss_avg, self.info

    # def AvgNonZeroReducer(self, loss):
    #     eps = 1.0e-9
    #     loss_sum = loss.sum(-1)
    #     loss_num = (loss != 0).sum(-1).float()

    #     loss_avg = loss_sum / (loss_num + eps)
    #     loss_avg[loss_num == 0] = 0
    #     return loss_avg, loss_num

    # def ComputeDistance(self, x, y):
    #     """
    #         x: [p, n_x, c]
    #         y: [p, n_y, c]
    #     """
    #     eps = 1.0e-9
    #     x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
    #     y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
    #     inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
    #     dist = x2 + y2 - 2 * inner
    #     dist = torch.sqrt(F.relu(dist) + eps)  # [p, n_x, n_y]
    #     return dist

    # def Convert2Triplets(self, row_labels, clo_label, dist):
    #     """
    #         row_labels: tensor with size [n_r]
    #         clo_label : tensor with size [n_c]
    #     """
    #     matches = (row_labels.unsqueeze(1) ==
    #                clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
    #     diffenc = torch.logical_not(matches)  # [n_r, n_c]
    #     p, n, _ = dist.size()
    #     ap_dist = dist[:, matches].view(p, n, -1, 1)
    #     an_dist = dist[:, diffenc].view(p, n, 1, -1)
    #     return ap_dist, an_dist
