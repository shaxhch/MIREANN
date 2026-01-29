import numpy as np
import torch 
import opt_einsum as oe
from torch.autograd.functional import jacobian
from src.MODEL import *
#nnmod定义在MODEL.py中
#============================calculate the energy===================================
class Property(torch.nn.Module):
    def __init__(self,density,nnmodlist):
        super(Property,self).__init__()
        self.density=density
        self.nnmod=nnmodlist[0]
        if len(nnmodlist) > 1:
            self.nnmod1=nnmodlist[1]
            self.nnmod2=nnmodlist[2]

    def forward(self,cart,numatoms,species,atom_index,shifts,create_graph=None):
        species=species.view(-1)
        density = self.density(cart,numatoms,species,atom_index,shifts)
        output=self.nnmod(density,species).view(numatoms.shape[0],-1)
        return output,
