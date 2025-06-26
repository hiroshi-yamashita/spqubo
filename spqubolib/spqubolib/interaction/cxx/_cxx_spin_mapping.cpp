#include <cstdio>
#include <cstdlib>
using namespace std;

namespace SPIN_MAPPING {

void _spinvector_to_spinarr(int Ly, int Lx, int N, float *x, int *pos,
                            float *arr) {
  int i, j;

  for (int k = 0; k < N; k++) {
    i = pos[k * 2];
    j = pos[k * 2 + 1];
    arr[i * Lx + j] = x[k];
  }
}

void _spinarr_to_spinvector(int Ly, int Lx, int N, float *x, int *pos,
                            float *arr) {
  int i, j;
  for (int k = 0; k < N; k++) {
    i = pos[k * 2];
    j = pos[k * 2 + 1];
    x[k] = arr[i * Lx + j];
  }
}
} // namespace SPIN_MAPPING
