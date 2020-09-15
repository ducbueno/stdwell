#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <cuda_runtime.h>
#include "kernel.hpp"

using namespace std;

template<typename T>
void read_vec(string fname, vector<T> &temp){
    T value;
    ifstream input(fname.c_str());

    while(input >> value){
        temp.push_back(value);
    }
    input.close();
}

int main(int, char *argv[]) {
    int deviceID = 0;
    struct cudaDeviceProp props;
    cudaStream_t stream;

    cudaSetDevice(deviceID);
    cudaGetDeviceProperties(&props, deviceID);
    cudaStreamCreate(&stream);

    vector<double> h_Cnnzs;
    vector<double> h_Dnnzs;
    vector<double> h_Bnnzs;
    vector<int> h_Ccols;
    vector<int> h_Bcols;
    vector<double> h_x;
    vector<double> h_y;
    vector<int> h_val_pointers;

    string fpath;
    if(string(argv[1]) == "synth"){
        fpath = "../../data/synth/";
    }
    else if(string(argv[1]) == "real"){
        string model = argv[2];
        fpath = "../../data/real/" + model + "/";
    }
    else{
        cout << "Invalig argument(s)" << endl;
        exit(0);
    }

    read_vec<double>(fpath + "Cnnzs.txt", h_Cnnzs);
    read_vec<double>(fpath + "Dnnzs.txt", h_Dnnzs);
    read_vec<double>(fpath + "Bnnzs.txt", h_Bnnzs);
    read_vec<int>(fpath + "Ccols.txt", h_Ccols);
    read_vec<int>(fpath + "Bcols.txt", h_Bcols);
    read_vec<double>(fpath + "x.txt", h_x);
    read_vec<double>(fpath + "y.txt", h_y);
    read_vec<int>(fpath + "val_pointers.txt", h_val_pointers);

    double *d_Cnnzs = nullptr;
    double *d_Dnnzs = nullptr;
    double *d_Bnnzs = nullptr;
    int *d_Ccols = nullptr;
    int *d_Bcols = nullptr;
    double *d_x = nullptr;
    double *d_y = nullptr;
    double *d_z1 = nullptr;
    double *d_z2 = nullptr;
    int *d_val_pointers = nullptr;

    cudaMalloc((void**)&d_Cnnzs, sizeof(double) * h_Cnnzs.size());
    cudaMalloc((void**)&d_Dnnzs, sizeof(double) * h_Dnnzs.size());
    cudaMalloc((void**)&d_Bnnzs, sizeof(double) * h_Bnnzs.size());
    cudaMalloc((void**)&d_Ccols, sizeof(int) * h_Ccols.size());
    cudaMalloc((void**)&d_Bcols, sizeof(int) * h_Bcols.size());
    cudaMalloc((void**)&d_val_pointers, sizeof(int) * h_val_pointers.size());
    cudaMalloc((void**)&d_x, sizeof(double) * h_x.size());
    cudaMalloc((void**)&d_y, sizeof(double) * h_y.size());

    cudaMemcpyAsync(d_Cnnzs, h_Cnnzs.data(), sizeof(double) * h_Cnnzs.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Dnnzs, h_Dnnzs.data(), sizeof(double) * h_Dnnzs.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Bnnzs, h_Bnnzs.data(), sizeof(double) * h_Bnnzs.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Ccols, h_Ccols.data(), sizeof(int) * h_Ccols.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Bcols, h_Bcols.data(), sizeof(int) * h_Bcols.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_val_pointers, h_val_pointers.data(), sizeof(int) * h_val_pointers.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, h_x.data(), sizeof(double) * h_x.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y, h_y.data(), sizeof(double) * h_y.size(), cudaMemcpyHostToDevice, stream);

    const unsigned int num_std_wells = h_val_pointers.size() - 1;
    apply_stdwell(d_Cnnzs, d_Dnnzs, d_Bnnzs, d_Ccols, d_Bcols, d_x, d_y, d_val_pointers, stream, num_std_wells);

    cudaMemcpyAsync(h_y.data(), d_y, sizeof(double) * h_y.size(), cudaMemcpyDeviceToHost, stream);

    for(double y: h_y){
        cout << y << endl;
    }

    string opath = fpath + "y_-cuda.txt";
    ofstream output_file(opath.c_str());
    ostream_iterator<double> output_iterator(output_file, "\n");
    copy(h_y.begin(), h_y.end(), output_iterator);

    cudaFree(d_Cnnzs);
    cudaFree(d_Dnnzs);
    cudaFree(d_Bnnzs);
    cudaFree(d_Ccols);
    cudaFree(d_Bcols);
    cudaFree(d_val_pointers);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaStreamDestroy(stream);

    return 0;
}
