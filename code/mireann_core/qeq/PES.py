import torch
import numpy as np
import os
from inference.density import *
from inference.get_neigh import *
from src.MODEL import *
from torch import jit

class PES(torch.nn.Module):
    def __init__(self,gausswidth,hardness,nlinked=1):
        super(PES, self).__init__()
        #========================set the global variable for using the exec=================
        global nblock, nl, dropout_p, table_norm, activate,norbit
        global oc_loop,oc_nblock, oc_nl, oc_dropout_p, oc_table_norm, oc_activate
        global nwave, neigh_atoms, cutoff, nipsin, atomtype,totcharge
        self.gausswidth = nn.parameter.Parameter(gausswidth)
        self.hardness = nn.parameter.Parameter(hardness)
        # global parameters for input_nn
        nblock = 1                    # nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
        nl=[128,128]                # NN structure
        dropout_p=[0.0,0.0]       # dropout probability for each hidden layer
        activate = 'Relu_like'
        table_norm= True
        oc_loop = 1
        oc_nl = [128,128]          # neural network architecture   
        oc_nblock = 1
        oc_dropout_p=[0.0,0.0]
        oc_activate = 'Relu_like'
        #========================queue_size sequence for laod data into gpu
        oc_table_norm=True
        norbit= None
        #======================read input_nn==================================
        with open('para/input_nn','r') as f1:
           while True:
              tmp=f1.readline()
              if not tmp: break
              string=tmp.strip()
              if len(string)!=0:
                  if string[0]=='#':
                     pass
                  else:
                     m=string.split('#')
                     exec(m[0],globals())
        # define the outputneuron of NN
        outputneuron=1
        #======================read input_nn=============================================
        nipsin=2
        cutoff=4.5
        nwave=7
        neigh_atoms=150
        totcharge=0.0
        with open('para/input_density','r') as f1:
           while True:
              tmp=f1.readline()
              if not tmp: break
              string=tmp.strip()
              if len(string)!=0:
                  if string[0]=='#':
                     pass
                  else:
                     m=string.split('#')
                     exec(m[0],globals())

        if activate=='Tanh_like':
            from src.activate import Tanh_like as actfun
        else:
            from src.activate import Relu_like as actfun

        if oc_activate=='Tanh_like':
            from src.activate import Tanh_like as oc_actfun
        else:
            from src.activate import Relu_like as oc_actfun        

        dropout_p=np.array(dropout_p)
        oc_dropout_p=np.array(oc_dropout_p)
        maxnumtype=len(atomtype)
        #========================use for read rs/inta or generate rs/inta================
        if 'rs' in globals().keys():
            rs=torch.from_numpy(np.array(rs))
            inta=torch.from_numpy(np.array(inta))
            nwave=rs.shape[1]
        else:
            inta=torch.ones((maxnumtype,nwave))
            rs=torch.stack([torch.linspace(0,cutoff,nwave) for itype in range(maxnumtype)],dim=0)

        #=========================use for read J/σ or generate J/σ=====================
        if 'hardness' in locals().keys():
           hardness=torch.from_numpy(np.array(hardness))
           gausswidth=torch.from_numpy(np.array(gausswidth))
        else:
           hardness=torch.rand(maxnumtype)*30
           gausswidth=torch.rand(maxnumtype)*10

        #======================for orbital================================
        nipsin+=1
        if not norbit:
            norbit=int(nwave*(nwave+1)/2*nipsin)
        #========================nn structure========================
        nl.insert(0,int(norbit))
        oc_nl.insert(0,int(norbit))
        #================read the periodic boundary condition, element and mass=========
        self.cutoff=cutoff
        self.totcharge = totcharge
        ocmod_list=[]
        for ioc_loop in range(oc_loop):
            ocmod_list.append(NNMod(maxnumtype,nwave,atomtype,oc_nblock,list(oc_nl),\
            oc_dropout_p,oc_actfun,table_norm=oc_table_norm))
        self.density=GetDensity(rs,inta,cutoff,nipsin,norbit,ocmod_list)
        self.nnmod=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
        #================================================nn module==================================================
        self.neigh_list=Neigh_List(cutoff,nlinked)
     
    def forward(self,period_table,cart,cell,species,mass):
        cart = cart.detach().clone()
        neigh_list, shifts = self.neigh_list(period_table, cart, cell, mass)
        density = self.density(cart, neigh_list, shifts, species)
        electronegativities = self.nnmod(density, species)
        dist_vec=cart.unsqueeze(1)-cart.unsqueeze(0)
        distances=torch.linalg.norm(dist_vec,dim=-1)
        species=species.long()
        gamma=torch.sqrt(torch.pow(self.gausswidth[species].unsqueeze(1),2)+torch.pow(self.gausswidth[species].unsqueeze(0),2))
        Amatrix=torch.erf(distances/torch.sqrt(torch.tensor(2.0))/gamma)
        Aldgdiag=self.hardness[species]+1.0/self.gausswidth[species]/(torch.pi**0.5)
        Amatrix[torch.arange(species.shape[0]), torch.arange(species.shape[0])]=Aldgdiag
        nondiagindex=~torch.eye(species.shape[0], dtype=torch.bool).expand_as(Amatrix)
        Amatrix[nondiagindex]=Amatrix[nondiagindex]/distances[nondiagindex]
        Amatrix=torch.nn.functional.pad(Amatrix, (0,1,0,1),value=1.0)
        Amatrix[species.shape[0],species.shape[0]]=0.0
        electronegativities=torch.nn.functional.pad(electronegativities,(0,0,0,1),value=self.totcharge)
        charge = torch.matmul(torch.linalg.inv(Amatrix),electronegativities.squeeze(-1))
        return charge.detach(), 
#        gradsList=[]
#        for i in range(charge.shape[0]-1):
#            varene = charge[i]
#            grad = torch.autograd.grad([varene], [cart], retain_graph=True, allow_unused=True)[0]
#            if grad is not None:
#                gradsList.append(grad)
#            else:
#                gradsList.append(torch.zeros_like(cart))
   
#        if gradsList is not None:
#            grads = torch.stack(gradsList)
#            return charge.detach(), -grads.detach()
