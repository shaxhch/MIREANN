#include "pes_cpp.hpp"
using namespace std;
using namespace torch::indexing;

int select_gpu();
template<typename T>
std::vector<T> trans_new_arr(T* arr,int length) {
  std::vector<T> arr_new;
  for(int i=0;i<length;i++){
    arr_new.push_back(arr[i]);
  }
  return arr_new;
}

//---------------------------------------------------------
Pes::Pes(double *_cell, int *_pbc, int _numatoms, char **_species, int _maxnumtype, char **_atomtype, double *_mass):
cell(_cell), pbc(_pbc), numatoms(_numatoms), species(_species), maxnumtype(_maxnumtype), atomtype(_atomtype),mass(_mass){
  
  //set model precision
  std::string datatype="double"; //float

  tensor_device_type=torch::kCPU;
  tensor_device=torch::empty(1);

  //--------------------- 
  torch::jit::GraphOptimizerEnabledGuard guard{true};
  torch::jit::setGraphExecutorOptimize(true);
  if (datatype=="double") {
    tensor_type = torch::kDouble;
    pes_model = torch::jit::load("REANN_PES_DOUBLE.pt");
  }else{ 
    tensor_type=torch::kFloat;
    pes_model = torch::jit::load("REANN_PES_FLOAT.pt");
  }
  if (torch::cuda::is_available()) {
    int id=select_gpu();
    tensor_device_type=torch::kCUDA;
    auto device=torch::Device(tensor_device_type,id);
    tensor_device=tensor_device.to(device);
    pes_model.to(tensor_device.device(),true);
//    cout << "The simulations are performed on the GPU "<< id << endl;
  }
  else {
//    cout << "The simulations are performed on CPU " << endl;
  }

  pes_model.eval();
  pes_model=torch::jit::optimize_for_inference(pes_model);

  //---------------------
  arr_pbc=trans_new_arr(_pbc,3);
  arr_cell=trans_new_arr(_cell,9);
  auto arr_atom=trans_new_arr(_species,_numatoms);
  auto arr_atomtype=trans_new_arr(_atomtype,_maxnumtype);
  arr_mass=trans_new_arr(_mass,_numatoms);
  this->numatoms=_numatoms;
  pbc_t=torch::from_blob(arr_pbc.data(),{3},torch::kInt).to(tensor_device.device(),true).contiguous();
  cell_t=torch::from_blob(arr_cell.data(),{3,3},torch::kDouble).to(tensor_type).to(tensor_device.device(),true).contiguous();
  std::unordered_map<std::string, int> species_map;
  for(int i=0;i<maxnumtype;i++){
    species_map[atomtype[i]]=i;
  }
  arr_species={};
  for(auto atom: arr_atom) {
    arr_species.push_back(species_map[atom]);
  }
  species_t=torch::from_blob(arr_species.data(),{this->numatoms},torch::kInt).to(tensor_device.device(),true).contiguous();
  mass_t=torch::from_blob(arr_mass.data(),{this->numatoms},torch::kDouble).to(tensor_type).to(tensor_device.device(),true).contiguous();
}

//---------------------------------------------------------
Pes::~Pes(){
}
//---------------------------------------------------------
void Pes::reann_out(int numatoms, double *cart, double *energy, double *force) {
  auto arr_xyz=trans_new_arr(cart,this->numatoms*3);
  torch::Tensor cart_t=torch::from_blob(arr_xyz.data(),{this->numatoms,3},torch::kDouble).to(tensor_type).to(tensor_device.device(),true);
//  cout << "after torch::from_blob" << endl;
       auto outputs = pes_model.forward({pbc_t,cart_t,cell_t,species_t,mass_t}).toTuple()->elements();
// cout << "after pes_model.forward" << endl;
  torch::Tensor energy_t=outputs[0].toTensor().to(torch::kDouble).cpu();
  torch::Tensor force_t=outputs[1].toTensor().to(torch::kDouble).cpu();
  auto force_pd=force_t.data_ptr<double>();
  for (int i=0;i<this->numatoms*3;i++){
    force[i] =force_pd[i];
  }
  *energy=energy_t.item<double>();
 return;
}
//-------------------------------------------------
int select_gpu() {
  int totalnodes, mynode;
  int trap_key = 0;
  system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >gpu_info");
  ifstream gpu_sel("gpu_info");
  string texts;
  vector<double> memalloc;
  while (getline(gpu_sel,texts))
  {
    string tmp_str;
    stringstream allocation(texts);    
    allocation >> tmp_str;
    allocation >> tmp_str;
    allocation >> tmp_str;
    memalloc.push_back(std::stod(tmp_str));
  }
  gpu_sel.close();
  auto smallest=min_element(std::begin(memalloc),std::end(memalloc));
  auto id=distance(std::begin(memalloc), smallest);
  torch::Tensor tensor_device=torch::empty(1000,torch::Device(torch::kCUDA,id));
  system("rm gpu_info");
  return id;
}
