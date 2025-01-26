import torch

from .base import BaseLoss, gather_and_scale_wrapper


class MSE(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(MSE, self).__init__(loss_term_weight)
        self.loss_fn = torch.nn.MSELoss()

    @gather_and_scale_wrapper
    def forward(self, mimic_embeddings, invisible_embeddings):
        # embeddings: [n, c, p] each.
        
        loss = self.loss_fn(mimic_embeddings, invisible_embeddings)

        #print(f"Loss MSE calculated")
        
        self.info.update({
            'loss': loss.detach().clone()
            })

        return loss, self.info
