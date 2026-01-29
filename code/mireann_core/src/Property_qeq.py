import torch
from torch import nn
from torch import Tensor
import opt_einsum as oe
import numpy as np

class Property(torch.nn.Module):
    def __init__(self,density,totcharge,nnmodlist,gausswidth,hardness):
        super(Property,self).__init__()
        '''
        gausswidth:tensor[maxnumtype] float element-dependent
        hardness:tensor[maxnumtype] float element-dependent 
        '''
        self.density=density
        self.nnmod=nnmodlist[0]
        self.totcharge=totcharge
        self.gausswidth=nn.parameter.Parameter(gausswidth)
        self.hardness=nn.parameter.Parameter(hardness)
        if len(nnmodlist) > 1:
            self.nnmod1=nnmodlist[1]
            self.nnmod2=nnmodlist[2]

    def forward(self,cart,numatoms,species,atom_index,shifts,create_graph=None):
        '''
        """
        coor[num].append(tmp[1:4])
        ntotpoint=len(coor)
        com_coor=np.zeros((ntotpoint,maxnumatom,3),dtype=scalmatrix.dtype)
        cart=com_coor[ipoint-tmpbatch:ipoint]
        ↓↓↓
        input cart: coordinates (nbatch,numatom,3)        
        input numatoms: number of atoms for each configuration (nbatch)
        input electronegativities: the output of MODEL part (nbatch,numatom)
        """
        
        """
        for ipoint in range(range_rank[1]-range_rank[0]):
            for itype,ele in enumerate(atomtype):
                mask=torch.tensor([m==ele for m in atom_rank[ipoint]])
                ele_index = torch.nonzero(mask).view(-1)
                if ele_index.shape[0] > 0:
                    species_rank[ipoint,ele_index]=itype
        ↓↓↓
        species: indice for element of each atom (nbatch,numatom) example:[[1,3,2],[1,3,2]]
        """
        distances: tensor[nbatch,numatoms,numatoms] distance vector between atom i and j
        gamma: tensor[nbatch,numatoms,numaotms] √(σi^2+σj^2)
        '''
        # torch.linalg.norm 函数用于计算张量的范数（norm）。范数是一个将向量映射到非负值的函数，表示向量的大小。 
        species_=species.view(-1)
        density = self.density(cart,numatoms,species_,atom_index,shifts)
        electronegativities=self.nnmod(density,species_).view(numatoms.shape[0],-1)
        dist_vec=cart.unsqueeze(2)-cart.unsqueeze(1)
        distances=torch.linalg.norm(dist_vec,dim=-1)
        gamma=torch.sqrt(torch.pow(self.gausswidth[species].unsqueeze(2),2)+torch.pow(self.gausswidth[species].unsqueeze(1),2))
        Amatrix=torch.erf(distances/torch.sqrt(torch.tensor(2.0))/gamma)
        Aldgdiag=self.hardness[species]+1.0/self.gausswidth[species]/(torch.pi**0.5)
        Amatrix[:, torch.arange(species.shape[1]), torch.arange(species.shape[1])]=Aldgdiag
        nondiagindex=~torch.eye(species.shape[1], dtype=torch.bool).unsqueeze(0).expand_as(Amatrix)
        Amatrix[nondiagindex]=Amatrix[nondiagindex]/distances[nondiagindex]
        Amatrix=torch.nn.functional.pad(Amatrix,(0,1,0,1),value=1.0)
        Amatrix[:,species.shape[1],species.shape[1]]=0.0
        electronegativities=torch.nn.functional.pad(electronegativities,(0,1),value=self.totcharge)
        charge=torch.matmul(torch.linalg.inv(Amatrix),electronegativities.unsqueeze(-1)).squeeze(-1)
        return charge[:,:-1],
