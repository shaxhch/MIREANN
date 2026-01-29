import torch
import numpy as np
from src.read_data import *
from src.get_batchneigh import *
from src.com import *

def get_info_of_rank(range_rank,atom,atomtype,mass,numatoms,scalmatrix,period_table,coor,force,bgcor,bgcharge,bgcharge_real,\
start_table,table_coor,neigh_atoms,batchsize,cutoff,device,np_dtype):
    atom_rank=atom[range_rank[0]:range_rank[1]]
    mass_rank=mass[range_rank[0]:range_rank[1]]
    numatoms_rank=numatoms[range_rank[0]:range_rank[1]]
    maxnumatom=max(numatoms_rank)
    cell_rank=np.array(scalmatrix[range_rank[0]:range_rank[1]],dtype=np_dtype)
    coor_rank=coor[range_rank[0]:range_rank[1]]
    force_rank=None
    bgcor_rank=None
    bgcharge_rank=None
    bgcharge_real_rank=None
    if start_table==1: 
        force_rank=force[range_rank[0]:range_rank[1]]
    if start_table==7: 
        bgcor_rank=bgcor[range_rank[0]:range_rank[1]]
        bgcharge_rank=bgcharge[range_rank[0]:range_rank[1]]
        bgcharge_real_rank=bgcharge_real[range_rank[0]:range_rank[1]]
    # get the index of each element
    species_rank=-torch.ones((range_rank[1]-range_rank[0],maxnumatom),dtype=torch.long)
    for ipoint in range(range_rank[1]-range_rank[0]):
        for itype,ele in enumerate(atomtype):
            mask=torch.tensor([m==ele for m in atom_rank[ipoint]])
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0] > 0:
                species_rank[ipoint,ele_index]=itype
    
    com_coor_rank,order_force_rank,com_bgcor_rank=get_com(coor_rank,force_rank,bgcor_rank,mass_rank,cell_rank,numatoms_rank,maxnumatom,table_coor,start_table)
    if start_table==1: 
        order_force_rank=torch.from_numpy(order_force_rank)
    if start_table==7:
        com_bgcor_rank=torch.from_numpy(com_bgcor_rank)
        max_bg = com_bgcor_rank.shape[1]
        bgcharge_padded = []
        bgcharge_real_padded = []
        for charges in bgcharge_rank:
            padded = np.zeros(max_bg, dtype=cell_rank.dtype)
            n_charges = len(charges)
            padded[:n_charges] = charges[:n_charges]
            bgcharge_padded.append(padded)
        for charges_real in bgcharge_real_rank:
            padded_real = np.zeros(max_bg, dtype=cell_rank.dtype)
            n_charges_real = len(charges_real)
            padded_real[:n_charges_real] = charges_real[:n_charges_real]
            bgcharge_real_padded.append(padded_real)
        bgcharge_rank = np.array(bgcharge_padded, dtype=cell_rank.dtype)
        bgcharge_rank = torch.from_numpy(bgcharge_rank)
        bgcharge_real_rank = np.array(bgcharge_real_padded, dtype=cell_rank.dtype)
        bgcharge_real_rank = torch.from_numpy(bgcharge_real_rank)
    com_coor_rank=torch.from_numpy(com_coor_rank)
    cell_rank=torch.from_numpy(cell_rank)
    numatoms_rank=torch.from_numpy(numatoms_rank)
    pbc_rank=torch.from_numpy(np.array(period_table[range_rank[0]:range_rank[1]],dtype=np.int64))
    shifts_rank,atom_index_rank=get_batch_neigh(com_coor_rank,cell_rank,species_rank,pbc_rank,neigh_atoms,batchsize,cutoff,device)
    return com_coor_rank,order_force_rank,com_bgcor_rank,bgcharge_rank,bgcharge_real_rank,numatoms_rank,species_rank,atom_index_rank,shifts_rank
