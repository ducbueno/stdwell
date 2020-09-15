#ifndef __KERNEL_H_
#define __KERNEL_H_

const char* stdwell_s = R"(
__kernel void stdwell(__global const double *valsC,
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

    barrier(CLK_LOCAL_MEM_FENCE);

    localSum[wiId] = 0;
    if(wiId < numActiveWorkItems){
        int b = wiId/valsPerBlock + rowptr[wgId];
        while(b < valSize + rowptr[wgId]){
            int colIdx = colsB[b];
            localSum[wiId] += valsB[b*blnc*blnr + r*blnc + c]*x[colIdx*blnc + c];
            b += numBlocksPerWarp;
        }

        if(wiId < valsPerBlock){
            localSum[wiId] += localSum[wiId + valsPerBlock];
        }

        b = wiId/valsPerBlock + rowptr[wgId];

        if(wiId < valsPerBlock){
            if(c == 0 || c == 2) {localSum[wiId] += localSum[wiId + 2];}
            if(c == 0 || c == 1) {localSum[wiId] += localSum[wiId + 1];}
        }

        if(c == 0 && wiId < valsPerBlock){
            z1[r] = localSum[wiId];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(wiId < blnr){
        double temp = 0.0;
        for(unsigned int i = 0; i < blnr; ++i){
            temp += valsD[wgId*blnr*blnr + wiId*blnr + i]*z1[i];
        }
        z2[wiId] = temp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(wiId < blnc*valSize){
        double temp = 0.0;
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
