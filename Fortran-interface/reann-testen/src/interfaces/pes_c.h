#ifdef __cplusplus
extern "C" {
  class Pes;
  typedef Pes PES;
#else
  typedef struct PES PES;
#endif

PES* create_pes(double *cell, int *pbc, int numatoms, char **species, int maxnumtype, char **atomtype, double *mass);

void delete_pes(PES* pes);

void pes_reann_out(PES* pes, int numatoms, double *cart, double *energy, double *force);
#ifdef __cplusplus
}
#endif
