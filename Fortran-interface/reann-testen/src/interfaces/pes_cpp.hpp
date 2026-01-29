#pragma once
#include <torch/torch.h>
#include <torch/script.h> 
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

class Pes {
public:
    Pes(double *cell, int *pbc, int numatoms, char **species, int maxnumtype, char **atomtype, double *mass);
    ~Pes();
    void reann_out(int numatoms, double *cart, double *energy, double *force);
  double *cell;int *pbc;long int numatoms;char **species; int maxnumtype;char **atomtype; double *mass;
  torch::Dtype tensor_type;
  torch::DeviceType tensor_device_type;
  torch::Tensor tensor_device;
  torch::jit::script::Module pes_model;

private:
  torch::Tensor pbc_t;
  torch::Tensor cell_t;
  torch::Tensor species_t;
  torch::Tensor mass_t;
  std::vector<int> arr_pbc;
  std::vector<double> arr_cell;
  std::vector<int> arr_species;
  std::vector<double> arr_mass;
};
