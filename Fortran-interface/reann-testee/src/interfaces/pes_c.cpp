#include <iostream>
#include "pes_cpp.hpp"
#include "pes_c.h"

using namespace std;

chPES* create_chpes(double *chcell, int *chpbc, int chnumatoms, char **chspecies, int chmaxnumtype, char **chatomtype, double *chmass, int chmaxnum_mm) {
  return new chPes(chcell, chpbc, chnumatoms, chspecies, chmaxnumtype, chatomtype, chmass, chmaxnum_mm);
}

void delete_chpes(chPES* chpes){
  delete chpes;
}

void pes_reann_chout(chPES* chpes, int chnumatoms, int chmaxnum_mm, double *chcart, double *chcart_mm, double *chcha_mm, double *chenergy, double *chforce, double *chforce_mm, double *chcharge_qm) {
  chpes->reann_chout(chnumatoms, chmaxnum_mm, chcart, chcart_mm, chcha_mm, chenergy, chforce, chforce_mm, chcharge_qm);
  return;
}
