#ifndef __KERNEL_H_
#define __KERNEL_H_

const char* stdwell_s = R"(
__kernel void add_well_contributions(__global const double *valsC,
                                     __global const double *valsD,
                                     __global const double *valsB,
                                     __global const int *colsC,
                                     __global const int *colsB,
                                     __global const double *x,
                                     __global double *y,
                                     const unsigned int blnc,
                                     const unsigned int blnr,
                                     __global const unsigned int *rowptr,
                                     __local double *localSum,
                                     __local double *z1,
                                     __local double *z2){
    int wgId = get_group_id(0);
    int wiId = get_local_id(0);
    int valSize = rowptr[wgId + 1] - rowptr[wgId];
    int valsPerBlock = blnc*blnr;
    int numActiveWorkItems = (32/valsPerBlock)*valsPerBlock;
    int numBlocksPerWarp = 32/valsPerBlock;
    int c = wiId % blnc;
    int r = (wiId/blnc) % blnr;
    double temp;

    localSum[wiId] = 0;
    if(wiId < numActiveWorkItems){
        int b = wiId/valsPerBlock + rowptr[wgId];
        while(b < valSize + rowptr[wgId]){
            int colIdx = colsB[b];
            localSum[wiId] += valsB[b*blnc*blnr + r*blnc + c]*x[colIdx*blnc + c];
            b += numBlocksPerWarp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int stride = valsPerBlock;
    if(wiId < stride){
        localSum[wiId] += localSum[wiId + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(c == 0 && wiId < valsPerBlock){
        for(stride = 2; stride > 0; stride >>= 1){
            localSum[wiId] += localSum[wiId + stride];
        }
        z1[r] = localSum[wiId];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(wiId < blnr){
        temp = 0.0;
        for(unsigned int i = 0; i < blnr; ++i){
            temp += valsD[wgId*blnr*blnr + wiId*blnr + i]*z1[i];
        }
        z2[wiId] = temp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(wiId < blnc*valSize){
        temp = 0.0;
        int bb = wiId/blnc + rowptr[wgId];
        int colIdx = colsC[bb];
        for (unsigned int j = 0; j < blnr; ++j){
            temp += valsC[bb*blnc*blnr + j*blnc + c]*z2[j];
        }
        y[colIdx*blnc + c] -= temp;
    }
}
)";

#endif // __KERNEL_H_
