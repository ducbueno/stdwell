#ifndef __KERNEL_H_
#define __KERNEL_H_

#include <cuda_runtime.h>

void apply_stdwell(double *Cnnzs, double *Dnnzs, double *Bnnzs, int *Ccols, int *Bcols,
                   double *x, double *y, int *val_pointers, cudaStream_t stream, const unsigned int num_std_wells);

#endif // __KERNEL_H_
