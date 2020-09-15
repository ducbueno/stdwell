#include <memory>
#include <fstream>
#include <iostream>
#include <iterator>
#include "kernel.hpp"
#include "opencl.hpp"

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
    int platformID = 0;
    int deviceID = 0;

    cl_int err = CL_SUCCESS;
    unique_ptr<cl::Context> context;
    unique_ptr<cl::CommandQueue> queue;
    unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::Buffer&, cl::LocalSpaceArg, cl::LocalSpaceArg, cl::LocalSpaceArg> > stdwell_k;

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platformID])(), 0};
    context.reset(new cl::Context(CL_DEVICE_TYPE_GPU, properties));

    vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();

    cl::Program::Sources source(1, make_pair(stdwell_s, strlen(stdwell_s)));
    cl::Program program = cl::Program(*context, source);
    program.build(devices);

    queue.reset(new cl::CommandQueue(*context, devices[deviceID], 0, &err));

    vector<double> h_Cnnzs;
    vector<double> h_Dnnzs;
    vector<double> h_Bnnzs;
    vector<unsigned int> h_Ccols;
    vector<unsigned int> h_Bcols;
    vector<double> h_x;
    vector<double> h_y;
    vector<unsigned int> h_val_pointers;

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
    read_vec<unsigned int>(fpath + "Ccols.txt", h_Ccols);
    read_vec<unsigned int>(fpath + "Bcols.txt", h_Bcols);
    read_vec<double>(fpath + "x.txt", h_x);
    read_vec<double>(fpath + "y.txt", h_y);
    read_vec<unsigned int>(fpath + "val_pointers.txt", h_val_pointers);

    for(double i: h_Cnnzs){
        cout << i << endl;
    }

    cl::Buffer d_Cnnzs = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * h_Cnnzs.size());
    cl::Buffer d_Dnnzs = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * h_Dnnzs.size());
    cl::Buffer d_Bnnzs = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * h_Bnnzs.size());
    cl::Buffer d_Ccols = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(unsigned int) * h_Ccols.size());
    cl::Buffer d_Bcols = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(unsigned int) * h_Bcols.size());
    cl::Buffer d_x = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * h_x.size());
    cl::Buffer d_y = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * h_y.size());
    cl::Buffer d_val_pointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(unsigned int) * h_val_pointers.size());

    queue->enqueueWriteBuffer(d_Cnnzs, CL_TRUE, 0, sizeof(double) * h_Cnnzs.size(), h_Cnnzs.data());
    queue->enqueueWriteBuffer(d_Dnnzs, CL_TRUE, 0, sizeof(double) * h_Dnnzs.size(), h_Dnnzs.data());
    queue->enqueueWriteBuffer(d_Bnnzs, CL_TRUE, 0, sizeof(double) * h_Bnnzs.size(), h_Bnnzs.data());
    queue->enqueueWriteBuffer(d_Ccols, CL_TRUE, 0, sizeof(unsigned int) * h_Ccols.size(), h_Ccols.data());
    queue->enqueueWriteBuffer(d_Bcols, CL_TRUE, 0, sizeof(unsigned int) * h_Bcols.size(), h_Bcols.data());
    queue->enqueueWriteBuffer(d_x, CL_TRUE, 0, sizeof(double) * h_x.size(), h_x.data());
    queue->enqueueWriteBuffer(d_y, CL_TRUE, 0, sizeof(double) * h_y.size(), h_y.data());
    queue->enqueueWriteBuffer(d_val_pointers, CL_TRUE, 0, sizeof(unsigned int) * h_val_pointers.size(), h_val_pointers.data());

    const unsigned int num_std_wells = h_val_pointers.size() - 1;
    const unsigned int dim_weqs = 3;
    const unsigned int dim_wells = 4;
    const unsigned int work_group_size = 32;
    const unsigned int total_work_items = num_std_wells * work_group_size;
    const unsigned int lmem1 = sizeof(double) * work_group_size;
    const unsigned int lmem2 = sizeof(double) * dim_wells;

    stdwell_k.reset(new cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                    cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int,
                    const unsigned int, cl::Buffer&, cl::LocalSpaceArg, cl::LocalSpaceArg,
                    cl::LocalSpaceArg>(cl::Kernel(program, "stdwell")));
    cl::Event event = (*stdwell_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                   d_Cnnzs, d_Dnnzs, d_Bnnzs, d_Ccols, d_Bcols, d_x, d_y, dim_weqs, dim_wells, d_val_pointers,
                                   cl::Local(lmem1), cl::Local(lmem2), cl::Local(lmem2));

    queue->enqueueReadBuffer(d_y, CL_TRUE, 0, sizeof(double) * h_y.size(), h_y.data());

    for(double y: h_y){
        cout << y << endl;
    }

    string opath = fpath + "y_-opencl.txt";
    ofstream output_file(opath.c_str());
    ostream_iterator<double> output_iterator(output_file, "\n");
    copy(h_y.begin(), h_y.end(), output_iterator);

    return 0;
}
