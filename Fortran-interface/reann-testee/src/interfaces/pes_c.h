#ifdef __cplusplus
extern "C" {
  class chPes;
  typedef chPes chPES;
#else
  typedef struct chPES chPES;
#endif

chPES* create_chpes(double *chcell, int *chpbc, int chnumatoms, char **chspecies, int chmaxnumtype, char **chatomtype, double *chmass, int chmaxnum_mm);

void delete_chpes(chPES* chpes);

void pes_reann_chout(chPES* chpes, int chnumatoms, int chmaxnum_mm, double *chcart, double *chcart_mm, double *chcha_mm, double *chenergy, double *chforce, double *chforce_mm, double *chcharge_qm);
#ifdef __cplusplus
}
#endif
