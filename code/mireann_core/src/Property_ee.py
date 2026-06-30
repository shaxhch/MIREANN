import torch
from torch import nn
from torch import Tensor
import opt_einsum as oe
import numpy as np

class Property(torch.nn.Module):
    def __init__(self,density,totcharge,nnmodlist,gausswidth,hardness,polarizability,slatR,isFreezeDensity=False,ckptPath=None):
        super(Property,self).__init__()
        self.density=density
        self.nnmod=nnmodlist[0]
        self.totcharge=totcharge
        self.gausswidth=nn.parameter.Parameter(gausswidth)
        self.hardness=nn.parameter.Parameter(hardness)
        self.polarizability=nn.parameter.Parameter(polarizability)
        self.slatR=nn.parameter.Parameter(slatR)
        if len(nnmodlist) > 1:
            self.nnmod1=nnmodlist[1]
            self.nnmod2=nnmodlist[2]
        if isFreezeDensity:
            self.prepareFineTuning(ckptPath)

    def prepareFineTuning(self, ckptPath):
        print(f'Loading ckpt file {ckptPath} ...')
        energyModelCkpt = torch.load(ckptPath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        frozenParameterCount = 0 
        for key, value in energyModelCkpt['reannparam'].items():
            print(f'Checkpoint param: {key}, Shape: {value.shape}')
        for name, param in self.named_parameters():
            print(f'Model param: {name}, Shape: {param.shape}')

        for name, param in self.named_parameters():
            if name in energyModelCkpt['reannparam']:
                print(f'Setting parameter: {name}')
                param.data.copy_(energyModelCkpt['reannparam'][name])
                frozenParameterCount += param.nelement()

    def forward(self,cart,bgcart,bgcha,bgcha_real,numatoms,species,atom_index,shifts,create_graph=None):
        species_=species.view(-1)
        density = self.density(cart,bgcart,bgcha,numatoms,species_,atom_index,shifts)
        electronegativities=-self.nnmod(density,species_).view(numatoms.shape[0],-1)
        electronegativities=torch.nn.functional.pad(electronegativities,(0,1),value=0.0)

        dist_vec=cart.unsqueeze(2)-cart.unsqueeze(1)
        distances=torch.linalg.norm(dist_vec,dim=-1)
        gamma=torch.sqrt(torch.pow(self.gausswidth[species].unsqueeze(2),2)+torch.pow(self.gausswidth[species].unsqueeze(1),2))
        Amatrix_part = torch.erf(distances/torch.sqrt(torch.tensor(2.0))/gamma)
        Aldgdiag=self.hardness[species]+1.0/self.gausswidth[species]/(torch.pi**0.5)
        Amatrix = torch.zeros((species.shape[0], species.shape[1], species.shape[1]),dtype=Aldgdiag.dtype,device=Aldgdiag.device)
        Amatrix += torch.eye(species.shape[1], dtype=Aldgdiag.dtype, device=Aldgdiag.device).unsqueeze(0)
        for i in range(len(numatoms)):
            diagindex_ = torch.eye(numatoms[i], dtype=torch.bool,device=Aldgdiag.device)
            nondiagindex_ = ~diagindex_
            diagindex = torch.zeros((species.shape[1], species.shape[1]), dtype=torch.bool, device=Aldgdiag.device)
            nondiagindex = torch.zeros((species.shape[1], species.shape[1]), dtype=torch.bool, device=Aldgdiag.device)
            diagindex[:numatoms[i], :numatoms[i]] = diagindex_
            nondiagindex[:numatoms[i], :numatoms[i]] = nondiagindex_
            Amatrix[i][nondiagindex]=Amatrix_part[i][nondiagindex]/distances[i][nondiagindex]
            Amatrix[i][diagindex] = Aldgdiag[i][:numatoms[i]]
        Amatrix=torch.nn.functional.pad(Amatrix,(0,1,0,1),value=0.0)
        for i in range(len(numatoms)):
            Amatrix[i,:numatoms[i],species.shape[1]]=1.0
            Amatrix[i,species.shape[1],:numatoms[i]]=1.0
        Amatrix[:,species.shape[1],species.shape[1]]=0.0
        electronegativities[:,species.shape[1]]=self.totcharge
        charge=torch.matmul(torch.linalg.inv(Amatrix),electronegativities.unsqueeze(-1)).squeeze(-1)
        qm_charges = charge[:,:-1]

        max_atoms = species.shape[1]
        atom_mask = (torch.arange(max_atoms, device=species.device).unsqueeze(0)<numatoms.unsqueeze(1))
        qm_charges = qm_charges * atom_mask
        delta = cart.unsqueeze(2) - bgcart.unsqueeze(1)  
        dist = torch.norm(delta, dim=-1)
        dist_safe = torch.where(dist < 1e-12,torch.ones_like(dist),dist)
        kernel = (1/dist_safe-torch.exp(-dist_safe/self.polarizability[species].unsqueeze(-1))*(1/dist_safe+ 0.5/self.polarizability[species].unsqueeze(-1)))
        kernel = kernel*atom_mask.unsqueeze(-1)
        potentials = torch.sum(qm_charges.unsqueeze(-1) * kernel,dim=1)*18.2223**2/627.510
        energy = torch.sum(potentials * bgcha_real, dim=-1)*627.510
        return potentials,energy
