import numpy as np

def get_com(coor, force, bgcor, mass, scalmatrix, numatoms, maxnumatom, table_coor, start_table):
    ntotpoint = len(coor)
    maxnumforce = maxnumatom * 3
    order_force = None
    com_bgcor = None
    com_coor = np.zeros((ntotpoint, maxnumatom, 3), dtype=scalmatrix.dtype)
    fcoor = np.zeros((maxnumatom, 3), dtype=scalmatrix.dtype)
    if start_table == 7:
        maxnumbg = max(len(point) for point in bgcor)
        com_bgcor = np.zeros((ntotpoint, maxnumbg,3), dtype=scalmatrix.dtype)
    if start_table == 1:
        order_force = np.zeros((ntotpoint, maxnumforce), dtype=scalmatrix.dtype)
    for ipoint in range(ntotpoint):
        tmpmass = np.array(mass[ipoint], dtype=scalmatrix.dtype)
        natom = numatoms[ipoint]
        matrix = np.linalg.inv(scalmatrix[ipoint])
        fcoor[0:natom] = coor[ipoint]
        
        if start_table == 1:
            order_force[ipoint, 0:natom*3] = np.array(force[ipoint], dtype=scalmatrix.dtype).reshape(-1)
        
        if table_coor == 0:
            fcoor[0:natom] = np.matmul(fcoor[0:natom], matrix)
        
        inv_coor = np.round(fcoor[0:natom] - fcoor[0])
        fcoor[0:natom] -= inv_coor
        fcoor[0:natom] = np.matmul(fcoor[0:natom], scalmatrix[ipoint, :, :])
        com = np.matmul(tmpmass, fcoor[0:natom, :]) / np.sum(tmpmass)
        com_coor[ipoint, 0:natom] = fcoor[0:natom] - com
        
        if start_table == 7:
            numbg = len(bgcor[ipoint])
            bg_fcoor=bgcor[ipoint]
            com_bgcor[ipoint,0:numbg]=bg_fcoor-com
    return com_coor, order_force, com_bgcor
