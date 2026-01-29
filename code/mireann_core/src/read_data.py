import numpy as np
import math

# read system configuration and energy/force
def Read_data(folderlist,nprob,start_table=None):
    coor=[]
    scalmatrix=[]
    abprop=[] 
    force=None
    bgcoor=None
    bgcharge=None
    bgcharge_real=None
    atom=[]
    mass=[]
    numatoms=[]
    period_table=[]
    # tmp variable
    #===================variable for force====================
    if start_table==1:
       force=[]
    if start_table==7:
       bgcoor=[]
       bgcharge=[]
       bgcharge_real=[]
       force=[]
    numpoint=[0 for _ in range(len(folderlist))]
    num=0
    for ifolder,folder in enumerate(folderlist):
        fname2=folder+'configuration'
        with open(fname2,'r') as f1:
            while True:
                string=f1.readline()
                if not string: break
                string=f1.readline()
                scalmatrix.append([])
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()[1:4]))
                period_table.append(m)
                coor.append([])
                mass.append([])
                atom.append([])
                if start_table==7: 
                    bgcoor.append([])
                    bgcharge.append([])
                    bgcharge_real.append([])
                while True:
                    string=f1.readline()
                    m=string.split()
                    if m[0]=="background:":
                        abprop.append([])
                        while True:
                            string=f1.readline()
                            m=string.split()
                            if m[0]=="abprop:":
                                break
                            tmp=list(map(float,m[:]))
                            bgcoor[num].append(tmp[0:3])
                            bgcharge[num].append(tmp[3])
                            abprop[num].append(tmp[4])
                            bgcharge_real[num].append(tmp[5])
                    if m[0]=="abprop:":
                        force.append(float(m[1]))
                        break
                    if start_table==1:
                        atom[num].append(m[0]) 
                        tmp=list(map(float,m[1:]))
                        mass[num].append(tmp[0])
                        coor[num].append(tmp[1:4])
                        force[num].append(tmp[4:7])
                    else:
                        atom[num].append(m[0]) 
                        tmp=list(map(float,m[1:]))
                        mass[num].append(tmp[0])
                        coor[num].append(tmp[1:4])
                numpoint[ifolder]+=1
                numatoms.append(len(atom[num]))
                num+=1
    return numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,abprop,force,bgcoor,bgcharge,bgcharge_real
