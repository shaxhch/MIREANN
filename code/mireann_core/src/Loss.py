import torch 
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn=nn.MSELoss(reduction="sum")

    def forward(self, var, ab):
        loss_list = []
        for ivar, iab in zip(var, ab):
            mask = (iab != 0)
            ivar_nonzero = ivar[mask]
            iab_nonzero = iab[mask]
            loss = self.loss_fn(ivar_nonzero, iab_nonzero).view(-1)
            loss_list.append(loss)
        return torch.cat(loss_list)
