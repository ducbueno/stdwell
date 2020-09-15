#ifndef __KERNEL_H_
#define __KERNEL_H_

#include <cuda_runtime.h>

__global__ void stdwell(const double * __restrict__ Cnnzs,
                        const double * __restrict__ Dnnzs,
                        const double * __restrict__ Bnnzs,
                        const int * __restrict__ Ccols,
                        const int * __restrict__ Bcols,
                        const double * __restrict__ x,
                        double * __restrict__ y,
                        const int dim,
                        const int dim_wells,
                        const int * __restrict__ val_pointers)
{
    const int idx_b = blockIdx.x;
    const int idx_t = threadIdx.x;
    const unsigned int val_size = val_pointers[idx_b + 1] - val_pointers[idx_b];

    const int vals_per_block = dim * dim_wells;        // 12
    const int num_active_threads = (32 / vals_per_block) * vals_per_block; // 24
    const int num_blocks_per_warp = 32 / vals_per_block; // 2
    const int lane = idx_t % 32;
    const int c = lane % dim;                           // col in block
    const int r = (lane / dim) % dim_wells;             // row in block

    extern __shared__ double smem[];
    double * __restrict__ z1 = smem;
    double * __restrict__ z2 = z1 + dim_wells;

    if (idx_t < dim_wells) {
        z1[idx_t] = 0.0;
    }

    __syncthreads();

    // z1 = B * x
    if (idx_t < num_active_threads) {
        // multiply all blocks with x
        double temp = 0.0;
        int b = idx_t / vals_per_block + val_pointers[idx_b];       // block id, val_size indicates number of blocks
        while (b < val_size + val_pointers[idx_b]) {
            int colIdx = Bcols[b];
            temp += Bnnzs[b * dim * dim_wells + r * dim + c] * x[colIdx * dim + c];
            b += num_blocks_per_warp;
        }

        // merge all blocks into 1 dim*dim_wells block
        // since NORNE has only 2 parallel blocks, do not use a loop
        temp += __shfl_down_sync(0x00ffffff, temp, dim * dim_wells);

        b = idx_t / vals_per_block + val_pointers[idx_b];

        // merge all (dim) columns of 1 block, results in a single 1*dim_wells vector, which is used to multiply with invD
        if (idx_t < vals_per_block) {
            // should be a loop as well, now only works for dim == 3
            if (c == 0 || c == 2) {temp += __shfl_down_sync(0x00000B6D, temp, 2);} // add col 2 to col 0
            if (c == 0 || c == 1) {temp += __shfl_down_sync(0x000006DB, temp, 1);} // add col 1 to col 0
        }

        // write 1*dim_wells vector to gmem, could be replaced with shfl broadcast to remove z1 altogether
        if (c == 0 && idx_t < vals_per_block) {
            z1[r] = temp;
        }
    }

    __syncthreads();

    // z2 = D^-1 * B * x = D^-1 * z1
    if (idx_t < dim_wells) {
        double temp = 0.0;
        for (int c = 0; c < dim_wells; ++c) {
            temp += Dnnzs[idx_b * dim_wells * dim_wells + idx_t * dim_wells + c] * z1[c];
        }
        z2[idx_t] = temp;
    }

    __syncthreads();

    // y -= C^T * D^-1 * B * x
    // use dim * val_size threads, each block is assigned 'dim' threads
    if (idx_t < dim * val_size) {
        double temp = 0.0;
        int b = idx_t / dim + val_pointers[idx_b];
        int cc = idx_t % dim;
        int colIdx = Ccols[b];
        for (unsigned int c = 0; c < dim_wells; ++c) {
            temp += Cnnzs[b * dim * dim_wells + c * dim + cc] * z2[c];
        }
        y[colIdx * dim + cc] -= temp;
    }
}

void apply_stdwell(double *Cnnzs, double *Dnnzs, double *Bnnzs, int *Ccols, int *Bcols,
                   double *x, double *y, int *val_pointers, cudaStream_t stream, const unsigned int num_std_wells){
    const unsigned int dim_weqs = 3;
    const unsigned int dim_wells = 4;
    int smem_size = 2 * sizeof(double) * dim_wells;
    stdwell <<< num_std_wells, 32, smem_size, stream>>>(Cnnzs, Dnnzs, Bnnzs, Ccols, Bcols, x, y, dim_weqs, dim_wells, val_pointers);
}

#endif // __KERNEL_H_
