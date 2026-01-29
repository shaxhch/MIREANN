# This is an example script to show how to obtain the energy and force by invoking the potential saved by the training .
# Typically, you can read the structure,mass, lattice parameters(cell) and give the correct periodic boundary condition (pbc) and t    he index of each atom. All the information are required to store in the tensor of torch. Then, you just pass these information to t    he calss "pes" that will output the energy and force.

import time
import numpy as np
import torch
from gpu_sel import *
# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# same as the atomtype in the file input_density
atomtype=['F', 'C', 'O', 'N', 'H']
#load the serilizable model
pes=torch.jit.load("REANN_CHA_DOUBLE.pt")
# FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
pes.to(device).to(torch.double)
# set the eval mode
pes.eval()
pes=torch.jit.optimize_for_inference(pes)
# save the lattic parameters
cell=np.zeros((3,3),dtype=np.float64)
period_table=torch.tensor([0,0,0],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition
npoint=0
rmse=torch.zeros(2,dtype=torch.double,device=device)
with open("configuration",'r') as f1:
    npoint = 0
    while True:
        string=f1.readline()
        if not string: break
        string=f1.readline()
        cell[0]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[1]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[2]=np.array(list(map(float,string.split())))
        string=f1.readline()
        species=[]
        cart=[]
        abforce=[]
        mass=[]
        bgcoor=[]
        bgcharge=[]
        while True:
            string=f1.readline()
            m=string.split()
            if m[0]=="background:":
                while True:
                    string=f1.readline()
                    m=string.split()
                    if m[0]=="abprop:":
                        break
                    tmp=list(map(float,m[:]))
                    bgcoor.append(tmp[0:3])
                    bgcharge.append(tmp[3])
            if m[0]=="abprop:":
                break
            tmp=string.split()
            tmp1=list(map(float,tmp[2:5]))
            cart.append(tmp1[0:3])
            mass.append(float(tmp[1]))
            species.append(atomtype.index(tmp[0]))
        abene=list(map(float,string.split()[1:]))
        abene=torch.from_numpy(np.array([abene])).to(device)
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
        mass=torch.from_numpy(np.array(mass)).to(device).to(torch.double)  # also float32/double
        abforce=torch.from_numpy(np.array(abforce)).to(device).to(torch.double)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
        bgcoor=torch.from_numpy(np.array(bgcoor)).to(device).to(torch.double)
        bgcharge=torch.from_numpy(np.array(bgcharge)).to(device).to(torch.double)
        if len(bgcoor) == 0 or len(bgcharge) == 0:
            bgcoor = torch.zeros((1,3), dtype=torch.double, device=device)
            bgcharge = torch.zeros(1, dtype=torch.double, device=device)

        npoint += 1 

        energy, QMforce, MMforce, QMcharge = pes(period_table, cart, bgcoor, bgcharge, tcell, species, mass)
