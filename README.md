# Efficient deep learning framework for multiscale simulations in enzymes via linear response approximation
1. The training data to perform predictions of QM energies in vacuo and QM/MM electrostatic energies via REANN and MIREANN are provided at https://doi.org/10.5281/zenodo.18401525.

2. The code of REANN and MIREANN are provided in "code/"  
It contains the training code of MIREANN package. Details are provided in the sections below.  
A: Prepare the environment  
   　　The REANN Package is built based on PyTorch and uses the "opt_einsum" package for optimizing einsum-like expressions frequently used in the calculation of the embedded density. In order to run the REANN package, users need to install PyTorch (version: 2.0.0) and its dependency environment based on the instructions on the PyTorch official website (https://pytorch.org/get-started/locally/) and the package named opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/). However, in our test, the REANN framework identified an "Error" when using double-float type to train models in the version 2.1.0 of PyTorch.  
B: for REANN  
   B.1 Prepare the "configuration" in the folder "train/" and "test", example input includes:  
   　　Line 1: Comment line  
   　　Line 2-4: Lattice vector defining the unit cell of the system  
   　　Line 5: Enable(1)/disable(0) the periodic boundary conditions in each direction  
   　　Line 6-N+5: Atomic name, relative atomic mass, cooridinates, atomic force vectors  
   　　Last Line: Start with "abprop:" and then follow by the target property  
   B.2 Parameters in "input_nn" and "input_density" file  
   　　Details can be referred in "code/reann_core/manual/REANNPackage_manumal_v_2_0.pdf"  
   B.3 Construct the model  
   　　Start to train the model with the command like "torchrun --nnodes=1 --nproc_per_node=1 --master_port=4161 $REANN_CORE_DIR"  
C: for MIREANN  
   C.1 Prepare the "configuration" in the folder "train/" and "test", example input includes:  
   　　Line 1: Comment line  
   　　Line 2-4: Lattice vector defining the unit cell of the system  
   　　Line 5: Enable(1)/disable(0) the periodic boundary conditions in each direction  
   　　Line 6-N+5: Atomic name, relative atomic mass, cooridinates, atomic force vectors  
   　　Line N+6: Indicate the background charges part with "background:"  
   　　Line N+6-Last Line: coordinates of MM atoms, background charges (0 for the case in vacuo), electrostatic potential, charges in computing electrostatic energy  
   　　Last line: the contributed electrostatic energy calculated from $$\sum_{i=1}^{N_{MM}}q_iV_i$$  
   C.2 Parameters in "input_nn" and "input_density" file  
   　　start_table = 7 to train the electrostatic potential    
   C.3 Construct the model  
   　　Start to train the model with the command like "torchrun --nnodes=1 --nproc_per_node=1 --master_port=4161 $MIEANN_CORE_DIR"  
3. Building an Interface for the Amber Program  
   　　Because Amber is commercial software, the modification of its codes in our scheme are not available unless reasonable request. Here, we only offer its external interface with MIREANN. To build the interface for the Amber program, user needs to build a dynamic-link library to link the JIT inference model of Pytorch trained by MIREANN to the main program sander in Amber, which is written in Fortran. The interface spans four programming languages: PyTorch, C++, C, and Fortran, which is realized via Cmake program.  
   　　The original copy for inferring energy is available at the website https://github.com/junfanxia/proj-reann-cpp2fortran-fixedcell.  
   　　The compilation is relatively complicated and user needs to follow these steps strictly:  
   1.Prepare the environment  
   　　Here are the software requirements:  
   　　　　CMake　　　　　　　　　　　　　　　　 3.19.3  
   　　　　libtorch-CPU/GPU　　　　　　　　　　1.12.1  
   　　　　gcc/g++/gfortran　　　　　　　　　　8.5.0  
   　　　　CUDA(only for GPU)　　　　　　　　　　11.3  
   2.Compiling and Linking  
   　　User needs to change the path of compilers in “build/build.sh” and the path of libtorch in “src/CmakeLists.txt” and “src/interfaces/CmakeLists.txt”. Then, execute Linux commands in the root directory of the interface:  
   　　cd build; sh build.sh; make  
   　　Then we can get the directories named “libs” and “modules” in the path of “build/”, which contains the dynamic-link library and header file in Fortran. User needs to replace original function call of ab initio calculations in the programs of molecular dynamics by the template in src/test.f90.  
   　　With these files, user can link them to Amber or other programs of molecular dynamics through adding the codes listed below in the Makefile like:  
   　　gfortran -fopenmp sander.o -I /home/shaxh/interface/reann-testen/build/modules/ -I /home/shaxh/interface/reann-testee/build/modules/ -L /home/shaxh/interface/reann-testen/build/lib/ -L/home/shaxh/interface/reann-testee/build/lib/ -I /home/shaxh/interface/reann-testen/build/modules/ -I /home/shaxh/interface/reann-testee/build/modules/  
   　　Then the interface is successfully linked to Amber.  
   3.Loading the Model  
   　　User needs to prepare a “input_reann” file to compensate for the missing parameters of the inference model (in the formation of .pt). It contains the information of cell, pbc, number of atoms, their species, maxtype and type species, which is provided in the path “code/MIREANN Fortran interface” as well. With the “input_reann” and .pt file, after the linking process, user can run the simulations with surfaces of MIREANN.  
   　　The example for ML/MM simulations and the inference files in ".pt" format are available in "example/" directory.  

CONTACT INFORMATION  
Yanzi Zhou  
School of Chemistry, Nanjing University,  
Nanjing, Jiangsu, 210023, China  

Email: zhouyz@nju.edu.cn  
