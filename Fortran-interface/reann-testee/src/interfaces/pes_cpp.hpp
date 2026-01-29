#pragma once
#include <torch/torch.h>
#include <torch/script.h> 
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

class chPes {
public:
    chPes(double *chcell, int *chpbc, int chnumatoms, char **chspecies, int chmaxnumtype, char **chatomtype, double *chmass, int chmaxnum_mm);
    ~chPes();
    void reann_chout(int chnumatoms, int chmaxnum_mm, double *chcart, double *chcart_mm, double *chcha_mm, double *chenergy, double *chforce, double *chforce_mm, double *chcharge_qm);
  double *chcell;int *chpbc;long int chnumatoms;char **chspecies; int chmaxnumtype;char **chatomtype; double *chmass;long int chmaxnum_mm;
  torch::Dtype tensor_type;
  torch::DeviceType tensor_device_type;
  torch::Tensor tensor_device;
  torch::jit::script::Module chpes_model;

private:
  torch::Tensor chpbc_t;
  torch::Tensor chcell_t;
  torch::Tensor chspecies_t;
  torch::Tensor chmass_t;
  std::vector<int> arr_chpbc;
  std::vector<double> arr_chcell;
  std::vector<int> arr_chspecies;
  std::vector<double> arr_chmass;
};
